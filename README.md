# refusal-guard

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that detects model content refusals and falls back to other models until it gets an answer that isn't a refusal. It works alongside [oh-my-hermes (OMH)](https://github.com/rlaope/oh-my-hermes).

## What it does

- **Detection.** About 86 English and Russian refusal phrases, drawn from LiteLLM's rejection phrases and NVIDIA garak, plus `finish_reason=content_filter`.
- **Main-turn retry.** On a refusal, the `transform_llm_output` hook re-asks the conversation through an OpenAI-compatible gateway (for example LiteLLM). It walks `REFUSAL_CHAIN` in order and returns the first answer that isn't a refusal, labeled with the model that gave it.
- **OMH child tasks.** A refused `delegate_task` result moves the OMH mixture chain forward (`omh_delegate_route action=fallback`) and asks the parent to re-dispatch.
- **Tool.** `refusal_fallback` (`mode=main|omh`) triggers a fallback by hand.
- Logs go to `$OMH_HOME/runtime/refusal-hook.log`.

## Install

```bash
cp -r refusal-guard ~/.hermes/plugins/   # or $HERMES_HOME/plugins/
# add "refusal-guard" to plugins.enabled in config.yaml, then restart Hermes
```

## Configuration (env)

| Variable | Default |
|---|---|
| `REFUSAL_CHAIN` | `dotop_deepseek-v4.1-flash,dotop_qwen3.8-27b,glm-4.7-flash,glm-4-flash` |
| `REFUSAL_BASE_URL` | `http://localhost:4000/v1` |
| `LITELLM_API_KEY` | gateway key (or read from `$HERMES_HOME/.env`) |
| `REFUSAL_RETRY_TIMEOUT` | `90` seconds per model |
| `REFUSAL_GUARD_DISABLED` | set to `1` to disable |

## Limitations

- The main-turn retry sends only the user and assistant text history, without tools, so the fallback answer is plain text.
- Hooks can't re-run `delegate_task` themselves. Upstream request: rlaope/oh-my-hermes#2159.

## Tests

```bash
python3 test_detector.py && python3 test_main_fallback.py
```

## License

MIT
