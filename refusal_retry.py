"""In-turn refusal retry: walk REFUSAL_CHAIN until a non-refusal answer."""
from __future__ import annotations
import json, os, urllib.request
from pathlib import Path
from typing import Any, Optional

REFUSAL_CHAIN = [m.strip() for m in os.environ.get(
    "REFUSAL_CHAIN",
    "dotop_deepseek-v4.1-flash,dotop_qwen3.8-27b,glm-4.7-flash,glm-4-flash",
).split(",") if m.strip()]
BASE_URL = os.environ.get("REFUSAL_BASE_URL", "http://localhost:4000/v1").rstrip("/")
TIMEOUT = float(os.environ.get("REFUSAL_RETRY_TIMEOUT", "90"))
MAX_HISTORY = 20


def _api_key() -> str:
    k = os.environ.get("LITELLM_API_KEY", "")
    if k:
        return k
    try:
        for line in Path("/opt/data/.env").read_text().splitlines():
            if line.startswith("LITELLM_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def build_messages(agent: Any) -> list[dict[str, str]]:
    msgs = list(getattr(agent, "_session_messages", None) or [])
    out: list[dict[str, str]] = []
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            continue
        t = _text(m.get("content")).strip()
        if t:
            out.append({"role": m["role"], "content": t})
    # drop trailing assistant (the refused answer) so the last turn is the user's
    while out and out[-1]["role"] == "assistant":
        out.pop()
    return out[-MAX_HISTORY:]


def _call(model: str, messages: list[dict[str, str]]) -> str:
    body = json.dumps({"model": model, "messages": messages, "max_tokens": 4096}).encode()
    req = urllib.request.Request(BASE_URL + "/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + _api_key()})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        data = json.load(r)
    ch = (data.get("choices") or [{}])[0]
    if ch.get("finish_reason") == "content_filter":
        return ""
    return _text((ch.get("message") or {}).get("content")).strip()


def retry_chain(messages, is_refusal, refused_model: str = "") -> dict[str, Any]:
    tried = []
    for model in REFUSAL_CHAIN:
        if model == refused_model:
            continue
        try:
            text = _call(model, messages)
        except Exception as exc:
            tried.append({"model": model, "result": f"error: {exc.__class__.__name__}"})
            continue
        if not text:
            tried.append({"model": model, "result": "empty"})
            continue
        if is_refusal(text):
            tried.append({"model": model, "result": "refusal"})
            continue
        tried.append({"model": model, "result": "ok"})
        return {"ok": True, "model": model, "text": text, "tried": tried}
    return {"ok": False, "tried": tried}
