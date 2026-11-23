"""
OpenAI-compatible LLM interface.

By default it talks to a local Ollama server's OpenAI-compatible endpoint
(http://127.0.0.1:11434/v1). Any OpenAI-compatible server works: set
OPENAI_API_BASE and OPENAI_API_KEY.

Why this file differs from upstream:
  * MiniRAG always calls ``llm_model_func(..., hashing_kv=<JsonKVStorage>)``.
    The previous version forwarded ``**kwargs`` straight into
    ``client.chat.completions.create()``, so every single call died with
    ``TypeError: got an unexpected keyword argument 'hashing_kv'``.
  * The model name was hard-coded to ``gpt-4o-mini``, so Ollama returned
    "model not found" even when the base URL was right. The name now comes
    from ``hashing_kv.global_config["llm_model_name"]`` (i.e. MiniRAG's
    ``llm_model_name=`` argument), like every other connector in this repo.
  * Responses are cached in llm_response_cache, which matters because
    ``ainsert()`` re-runs entity extraction over *all* processed chunks on
    every call - without a cache that is O(n^2) LLM requests.
"""

import os
from typing import List, Optional

import asyncio
from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minirag.utils import compute_args_hash, logger


class GPTKeywordExtractionFormat(BaseModel):
    high_level_keywords: List[str]
    low_level_keywords: List[str]


DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"

# phi3.5 (3.8B) will happily ramble for 10k tokens of nonsense when it is
# given only a bare user turn: with the entity-extraction prompt it sometimes
# answers with an unrelated essay instead of the requested records. A short
# system turn pins it down (measured: 5.4s + correct format, vs. 278s of
# garbage without it). MiniRAG never passes a system prompt itself.
DEFAULT_SYSTEM_PROMPT = (
    "You are a precise assistant that follows instructions exactly. "
    "Answer only what is asked, in exactly the requested format, "
    "with no extra commentary, no repetition of the instructions."
)

# Hard cap so a runaway generation cannot eat minutes per chunk.
DEFAULT_MAX_TOKENS = 768

# Ollama defaults to 0.8; lower is far more repeatable for extraction.
DEFAULT_TEMPERATURE = 0.2

# Models that "think" before answering. On Ollama's /v1 endpoint the switches
# for this are all ignored: "think": false, a "/no_think" suffix and
# chat_template_kwargs.enable_thinking all leave the reasoning block running,
# and it eats the whole max_tokens budget - the caller then gets en empty
# string instead of the extraction records (measured: 0 chars of content,
# 3600 chars of thinking). Talking to the NATIVE endpoint with "think": false
# really does disable it (0 chars thinking, records returned, done_reason=stop).
THINKING_MODEL_TAGS = ("qwen3", "deepseek-r1", "qwq", "reason")


def _is_thinking_model(model: str) -> bool:
    return any(tag in model.lower() for tag in THINKING_MODEL_TAGS)


def _native_base() -> str:
    base = _base_url().rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


async def _ollama_native_chat(messages, model, max_tokens, temperature) -> str:
    """POST /api/chat, which is the only Ollama endpoint that honours think=false."""
    import httpx

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            # Ollama defaults to a 4096 token window and silently truncates
            # whatever is longer than that. A QA prompt over a 490-chunk index
            # is far bigger, and the truncation eats the retrieved sources.
            "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "4096")),
        },
    }
    async with httpx.AsyncClient(timeout=900.0) as client:
        resp = await client.post(_native_base() + "/api/chat", json=payload)
        resp.raise_for_status()
        return (resp.json().get("message") or {}).get("content") or ""

_clients: dict[tuple[str, str], AsyncOpenAI] = {}


def _base_url() -> str:
    return (
        os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("LLM_BASE_URL")
        or DEFAULT_BASE_URL
    )


def _api_key() -> str:
    return (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or "ollama"
    )


def _get_client() -> AsyncOpenAI:
    """One shared async client per (base_url, api_key) - keeps connection
    pooling instead of building an httpx client for every request."""
    key = (_base_url(), _api_key())
    client = _clients.get(key)
    if client is None:
        client = AsyncOpenAI(
            api_key=key[1],
            base_url=key[0],
            timeout=600.0,
            max_retries=0,  # tenacity below is our retry policy
        )
        _clients[key] = client
    return client


def _model_name(hashing_kv, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    if hashing_kv is not None:
        try:
            name = (getattr(hashing_kv, "global_config", None) or {}).get(
                "llm_model_name"
            )
            if name:
                return name
        except Exception:  # pragma: no cover - defensive
            pass
    return os.environ.get("LLM_MODEL") or "gpt-4o-mini"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,)),
)
async def gpt_4o_mini_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    model_name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    **kwargs,
) -> str:
    hashing_kv = kwargs.pop("hashing_kv", None)
    # MiniRAG / LightRAG specific flags that must never reach the API call.
    for _flag in ("keyword_extraction", "mode", "encoding_format", "response_format"):
        kwargs.pop(_flag, None)

    model = _model_name(hashing_kv, model_name or kwargs.pop("model", None))

    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_messages or [])
    messages.append({"role": "user", "content": prompt})

    args_hash = compute_args_hash(model, messages, max_tokens)

    if hashing_kv is not None:
        try:
            cached = await hashing_kv.get_by_id(args_hash)
        except Exception:  # pragma: no cover
            cached = None
        if isinstance(cached, dict) and cached.get("return"):
            return cached["return"]

    max_out = max_tokens or DEFAULT_MAX_TOKENS

    # Thinking models go through Ollama's native endpoint, otherwise every call
    # comes back empty (see THINKING_MODEL_TAGS above).
    if _is_thinking_model(model):
        for attempt in range(3):
            try:
                content = await _ollama_native_chat(
                    messages, model, max_out, DEFAULT_TEMPERATURE
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(2 * (attempt + 1))
        else:
            logger.error(f"native ollama call failed ({model}): {last_error}")
            return ""
    else:
        content = await _openai_compatible(messages, model, max_out, kwargs)
        if content is None:
            return ""

    if hashing_kv is not None:
        try:
            await hashing_kv.upsert({args_hash: {"return": content, "model": model}})
        except Exception as exc:  # pragma: no cover
            logger.debug(f"llm response cache write failed: {exc}")

    return content


async def _openai_compatible(messages, model, max_tokens, kwargs):
    """The /v1 path, for models that do not need thinking disabled."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": DEFAULT_TEMPERATURE,
    }
    payload.update(kwargs)

    resp = None
    last_error = None
    for attempt in range(3):
        try:
            resp = await _get_client().chat.completions.create(**payload)
            break
        except TypeError:
            # Model/server does not know one of the optional fields.
            raise
        except Exception as exc:
            last_error = exc
            text = str(exc)
            # Ollama aborts a generation that falls into a repetition loop with
            # "500 prediction aborted, token repeat limit reached". Retrying the
            # same prompt just burns another minute and then kills the whole
            # insert() - so give up on this chunk and keep the run alive.
            if "token repeat limit" in text or "prediction aborted" in text:
                logger.warning(f"LLM gave up on a repeating response ({model}); "
                               f"chunk skipped, re-run to retry it")
                return None
            if attempt < 2:
                await asyncio.sleep(2 * (attempt + 1))

    if resp is None:
        logger.error(f"LLM call failed after 3 attempts ({model}): "
                     f"{type(last_error).__name__}: {last_error}")
        return None

    return resp.choices[0].message.content or ""

