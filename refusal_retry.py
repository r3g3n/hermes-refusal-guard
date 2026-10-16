"""In-turn refusal retry: walk REFUSAL_CHAIN until a non-refusal answer.

Optional per-model pre-prompt injection
----------------------------------------
Before calling a retry model you may prepend an extra ``system`` message.
Configure it in ``preprompts.json`` next to this file (created empty by default):

    {
      "default": "",                       # applied to every retry model
      "fallback-a": "",     # applied only to this model (overrides default)
      "fallback-b": ""
    }

An empty string = no injection. Content is supplied by the operator; this module
only provides the plumbing to carry it through.

endpoints.json (optional)
-------------------------
Per-model OpenAI-compatible base URL. Missing models use REFUSAL_BASE_URL.

    {
      "fallback-b": "http://127.0.0.1:8080/v1"
    }

Parallel race
-------------
When REFUSAL_PARALLEL=1 (default), all chain models are called concurrently;
the first non-refusal wins. Sequential mode: REFUSAL_PARALLEL=0.
"""
from __future__ import annotations
import base64, json, os, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

REFUSAL_CHAIN = [m.strip() for m in os.environ.get(
    "REFUSAL_CHAIN",
    "fallback-a,fallback-b",
).split(",") if m.strip()]
BASE_URL = os.environ.get("REFUSAL_BASE_URL", "http://127.0.0.1:4000/v1").rstrip("/")
TIMEOUT = float(os.environ.get("REFUSAL_RETRY_TIMEOUT", "600"))
PARALLEL = os.environ.get("REFUSAL_PARALLEL", "0").strip() not in ("0", "false", "False", "no")
MAX_HISTORY = 20
DEFAULT_MAX_TOKENS = int(os.environ.get("REFUSAL_MAX_TOKENS", "16384"))
MAX_TOKENS_FILE = Path(__file__).with_name("max_tokens.json")
ENDPOINTS_FILE = Path(__file__).with_name("endpoints.json")
PREPROMPT_FILE = Path(__file__).with_name("preprompts.json")
LOG_PATH = os.environ.get("REFUSAL_LOG", "/opt/data/.omh/runtime/refusal-hook.log")


def _max_tokens_for(model):
    try:
        data = json.loads(MAX_TOKENS_FILE.read_text())
        if isinstance(data, dict):
            if model in data:
                return int(data[model])
            if "default" in data:
                return int(data["default"])
    except Exception:
        pass
    return DEFAULT_MAX_TOKENS


def _endpoints() -> dict[str, str]:
    try:
        data = json.loads(ENDPOINTS_FILE.read_text())
        if isinstance(data, dict):
            return {str(k): str(v).rstrip("/") for k, v in data.items() if v}
    except Exception:
        pass
    return {}


def _base_url_for(model: str) -> str:
    return _endpoints().get(model) or BASE_URL


def _needs_auth(base_url: str) -> bool:
    # local llama.cpp / tunneled endpoints usually need no key
    u = base_url.lower()
    if "127.0.0.1" in u or "127.0.0.1" in u or "localhost" in u:
        return False
    return True


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


def _preprompts() -> dict[str, str]:
    try:
        data = json.loads(PREPROMPT_FILE.read_text())
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _decode_preprompt(val):
    """Value may be plain text, or base64 when prefixed with "base64:"."""
    if not isinstance(val, str):
        return ""
    s = val.strip()
    if s.startswith("base64:"):
        try:
            return base64.b64decode(s[len("base64:"):]).decode("utf-8")
        except Exception:
            return ""
    return s


def _preprompt_for(model: str) -> str:
    pp = _preprompts()
    val = pp.get(model)
    if val is None:
        val = pp.get("default", "")
    return _decode_preprompt(val or "").strip()


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
    while out and out[-1]["role"] == "assistant":
        out.pop()
    return out[-MAX_HISTORY:]


def _wire_model(model: str, base_url: str) -> str:
    """llama.cpp ignores model id; keep a stable local name for direct calls."""
    if "127.0.0.1" in base_url or "127.0.0.1" in base_url or "localhost" in base_url:
        return os.environ.get("REFUSAL_LOCAL_MODEL", "fallback-b")
    return model


def _call(model: str, messages: list[dict[str, str]]) -> str:
    pre = _preprompt_for(model)
    payload_msgs = ([{"role": "system", "content": pre}] + messages) if pre else messages
    base = _base_url_for(model)
    body = json.dumps({
        "model": _wire_model(model, base),
        "messages": payload_msgs,
        "max_tokens": _max_tokens_for(model),
    }).encode()
    headers = {"Content-Type": "application/json"}
    if _needs_auth(base):
        headers["Authorization"] = "Bearer " + _api_key()
    req = urllib.request.Request(base + "/chat/completions", data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"HTTPError {e.code}: {err_body}") from e
    ch = (data.get("choices") or [{}])[0]
    if ch.get("finish_reason") == "content_filter":
        return ""
    return _text((ch.get("message") or {}).get("content")).strip()


def _step_log(obj):
    try:
        import time as _t
        obj = {"ts": _t.strftime("%Y-%m-%dT%H:%M:%S%z"), **obj}
        with open(LOG_PATH, "a") as _f:
            _f.write(json.dumps(obj) + "\n")
    except Exception:
        pass


def _eval_one(model, messages, is_refusal):
    import time as _t
    _step_log({"event": "chain_step_start", "model": model, "base": _base_url_for(model)})
    _t0 = _t.time()
    try:
        text = _call(model, messages)
    except Exception as exc:
        result = f"error: {exc.__class__.__name__}"
        detail = str(exc)[:200]
        _step_log({"event": "chain_step", "model": model, "result": result, "detail": detail, "elapsed_s": round(_t.time()-_t0, 1)})
        return {"model": model, "result": result, "text": "", "ok": False}
    if not text:
        _step_log({"event": "chain_step", "model": model, "result": "empty", "elapsed_s": round(_t.time()-_t0, 1)})
        return {"model": model, "result": "empty", "text": "", "ok": False}
    if is_refusal(text):
        _step_log({"event": "chain_step", "model": model, "result": "refusal", "elapsed_s": round(_t.time()-_t0, 1)})
        return {"model": model, "result": "refusal", "text": "", "ok": False}
    _step_log({"event": "chain_step", "model": model, "result": "ok", "elapsed_s": round(_t.time()-_t0, 1), "chars": len(text)})
    return {"model": model, "result": "ok", "text": text, "ok": True}


def retry_chain(messages, is_refusal, refused_model: str = "") -> dict[str, Any]:
    models = [m for m in REFUSAL_CHAIN if m != refused_model]
    if not models:
        return {"ok": False, "tried": []}

    if not PARALLEL or len(models) == 1:
        tried = []
        for model in models:
            r = _eval_one(model, messages, is_refusal)
            tried.append({"model": r["model"], "result": r["result"]})
            if r["ok"]:
                return {"ok": True, "model": r["model"], "text": r["text"], "tried": tried}
        return {"ok": False, "tried": tried}

    # Parallel race: first non-refusal wins; others may still finish in background.
    _step_log({"event": "chain_race_start", "models": models, "parallel": True})
    tried = []
    winner = None
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        futs = {pool.submit(_eval_one, m, messages, is_refusal): m for m in models}
        for fut in as_completed(futs):
            r = fut.result()
            tried.append({"model": r["model"], "result": r["result"]})
            if r["ok"] and winner is None:
                winner = r
                # keep draining briefly so tried[] is mostly complete; don't wait forever
                break
        if winner is not None:
            # collect any already-finished siblings for the trace
            for fut, m in futs.items():
                if fut.done() and not any(t["model"] == m for t in tried):
                    try:
                        r = fut.result()
                        tried.append({"model": r["model"], "result": r["result"]})
                    except Exception as exc:
                        tried.append({"model": m, "result": f"error: {exc.__class__.__name__}"})
            _step_log({"event": "chain_race_winner", "model": winner["model"], "tried": tried})
            return {"ok": True, "model": winner["model"], "text": winner["text"], "tried": tried}
        # all finished without ok
        for fut, m in futs.items():
            if not any(t["model"] == m for t in tried):
                try:
                    r = fut.result()
                    tried.append({"model": r["model"], "result": r["result"]})
                except Exception as exc:
                    tried.append({"model": m, "result": f"error: {exc.__class__.__name__}"})
    _step_log({"event": "chain_race_fail", "tried": tried})
    return {"ok": False, "tried": tried}
