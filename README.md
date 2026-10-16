# refusal-guard

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that detects model content refusals and retries through a configurable model chain until it gets a non-refusal answer. Works alongside [oh-my-hermes (OMH)](https://github.com/rlaope/oh-my-hermes).

## What it does

- **Detection.** English and Russian refusal phrases (LiteLLM rejection phrases + NVIDIA garak style), plus `finish_reason=content_filter`.
- **Main-turn retry.** On a refusal, the `transform_llm_output` hook re-asks the conversation through an OpenAI-compatible gateway. It walks `REFUSAL_CHAIN` (sequential by default) and returns the first non-refusal answer, with a TUI footer showing the chain trace.
- **Sticky session switch.** After a successful in-turn retry, the session is switched to the head of Hermes `fallback_providers` (same path as `/model`) so later turns stay on that model instead of reverting to the primary.
- **Per-model endpoints.** Optional `endpoints.json` can route a chain model to a direct OpenAI-compatible URL (for example a local server), bypassing the gateway.
- **Per-model max tokens / preprompts.** Optional `max_tokens.json` and `preprompts.json` (plain text or `base64:`-prefixed). Preprompt injection applies to **retry-chain models only**, not the primary turn.
- **OMH child tasks.** A refused `delegate_task` result advances the OMH mixture chain (`omh_delegate_route action=fallback`).
- **Tool.** `refusal_fallback` (`mode=main|omh`) triggers a fallback by hand.
- Live step logs go to `$OMH_HOME/runtime/refusal-hook.log`.

## Install

```bash
cp -r hermes-refusal-guard ~/.hermes/plugins/refusal-guard
# add "refusal-guard" to plugins.enabled in config.yaml, then restart Hermes
```

Optional configs (copy next to the plugin or mount via your vault):

```bash
cp preprompts.example.json preprompts.json
cp endpoints.example.json endpoints.json
cp max_tokens.example.json max_tokens.json
```

## Configuration (env)

| Variable | Default |
|---|---|
| `REFUSAL_CHAIN` | `fallback-a,fallback-b` |
| `REFUSAL_BASE_URL` | `http://127.0.0.1:4000/v1` |
| `LITELLM_API_KEY` | gateway key (or from `$HERMES_HOME/.env`) |
| `REFUSAL_RETRY_TIMEOUT` | `600` seconds per model |
| `REFUSAL_PARALLEL` | `0` (sequential; set `1` for race) |
| `REFUSAL_LOCAL_MODEL` | wire model id for local endpoints |
| `REFUSAL_GUARD_DISABLED` | set to `1` to disable |

Also set Hermes top-level `fallback_providers` to the model you want the **session** to stick to after the first refusal (see sticky session switch above). Raise `plugins.hook_callback_timeout` if the chain can run long.

## Preprompts (`base64:`)

`preprompts.json` values may be plain text or Base64 with a `base64:` prefix (handy when the prompt has newlines/quotes). The plugin decodes on every retry; no restart needed.

Generate from a file:

```bash
# macOS / Linux
printf 'base64:' > preprompts.b64.txt
base64 < your-preprompt.txt | tr -d '\n' >> preprompts.b64.txt
# then paste the single line into preprompts.json, e.g.:
# { "default": "", "fallback-a": "base64:...." }
```

Or one-shot with Python:

```bash
python3 -c 'import base64,sys; print("base64:"+base64.b64encode(open(sys.argv[1],"rb").read()).decode())' your-preprompt.txt
```

Put only placeholders in git. Keep real `preprompts.json` private (it is gitignored).


## Sensitive data

This repository ships **placeholders only**. Put real model ids, gateway URLs, API keys, and any preprompt text in your private env / vault files — never commit them here.

## Limitations

- The main-turn retry sends user/assistant text history without tools, so the fallback answer is plain text.
- Hooks can't re-run `delegate_task` themselves. Upstream request: rlaope/oh-my-hermes#2159.
- Preprompts in `preprompts.json` apply only to retry-chain calls, not the primary model turn (use Hermes `SOUL.md` / system prompt for that).

## Tests

```bash
python3 test_detector.py && python3 test_main_fallback.py
```

## License

MIT
