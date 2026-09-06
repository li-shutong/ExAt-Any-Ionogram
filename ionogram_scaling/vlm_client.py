"""VLM client for the ionogram-scaling skill.

Uses raw httpx for the OpenAI Responses API (more reliable with aggregator
backends that reject SDK-injected parameters on reasoning models) and falls
back to the openai SDK for the Chat Completions API.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Union

import httpx

from ionogram_scaling.config import Config

logger = logging.getLogger("ionogram_scaling.vlm")


def _parse_headers(raw: str) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if not raw:
        return headers
    for sep in (";", "\n", ","):
        if sep in raw:
            items = raw.split(sep)
            break
    else:
        items = [raw]
    for item in items:
        if ":" in item:
            k, v = item.split(":", 1)
            headers[k.strip()] = v.strip()
    return headers


class VLMClient:
    def __init__(self, config: Config):
        self.config = config
        self.headers = _parse_headers(config.http_headers)
        self.headers.setdefault("Content-Type", "application/json")
        self.headers.setdefault("Authorization", f"Bearer {config.api_key}")
        self.model = config.model
        self.wire_api = config.wire_api

    # ------------------------------------------------------------------
    def chat(
        self,
        system_prompt: str,
        user_text: str,
        image_b64: Optional[str] = None,
        image_mime: str = "image/jpeg",
    ) -> str:
        if self.wire_api == "responses":
            return self._call_responses(system_prompt, user_text, image_b64, image_mime)
        return self._call_chat(system_prompt, user_text, image_b64, image_mime)

    # ------------------------------------------------------------------
    def _call_responses(
        self,
        system_prompt: str,
        user_text: str,
        image_b64: Optional[str],
        image_mime: str,
    ) -> str:
        user_content: List[Dict[str, Any]] = [{"type": "input_text", "text": user_text}]
        if image_b64:
            user_content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{image_mime};base64,{image_b64}",
                }
            )

        payload: Dict[str, Any] = {
            "model": self.model,
            "store": False,
            "reasoning": {"effort": self.config.reasoning_effort},
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        if self.config.max_tokens:
            payload["max_output_tokens"] = self.config.max_tokens

        url = f"{self.config.base_url.rstrip('/')}/responses"
        return _post_with_retry(url, self.headers, payload, self.config.api_retries)

    # ------------------------------------------------------------------
    def _call_chat(
        self,
        system_prompt: str,
        user_text: str,
        image_b64: Optional[str],
        image_mime: str,
    ) -> str:
        import openai

        sdk = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            default_headers=self.headers or None,
        )
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if image_b64:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{image_mime};base64,{image_b64}"},
                        },
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": user_text})

        r = sdk.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
        )
        return r.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_responses_text(data: dict) -> str:
    """Pull the assistant text out of a Responses API JSON body."""
    parts: List[str] = []
    for item in data.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text" and c.get("text"):
                parts.append(c["text"])
    return "".join(parts)


def _post_with_retry(url: str, headers: dict, payload: dict, max_retries: int) -> str:
    """POST with retry on 5xx / transient errors. Returns extracted text."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=300)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            last_err = e
            wait = min(10 * attempt, 60)
            logger.warning("Attempt %d/%d network error: %s (retry in %ds)", attempt, max_retries, e, wait)
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            return _extract_responses_text(resp.json())

        last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
        if 500 <= resp.status_code < 600:
            wait = min(10 * attempt, 60)
            logger.warning(
                "Attempt %d/%d server error %d, retrying in %ds",
                attempt, max_retries, resp.status_code, wait,
            )
            time.sleep(wait)
            continue
        raise last_err

    raise RuntimeError(f"All {max_retries} retries failed: {last_err}")


def extract_json(text: str) -> Optional[Union[dict, list]]:
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError as e:
                        logger.warning("JSON parse failed: %s", e)
                        return None
    return None
