# Inference Deployment

Dakota1890 inference is remote-first. Do not require adopters to load a 35B model locally.

## Current Surfaces

| Surface | Script | Status |
|---|---|---|
| Thinking Machines / Tinker sampler | `run_inference.py` | Verified working |
| Hugging Face dedicated endpoint | `hf_inference_standalone.py --endpoint-url ...` | Ready once an endpoint is provisioned |
| Hugging Face shared Inference API | `hf_inference_standalone.py` | Adapter artifacts are public, but no shared provider mapping is currently available |

## Primary Endpoint

The verified sampler is:

```text
tinker://1f23df9c-5d88-59d9-a7e8-dd4e169ea7d0:train:0/sampler_weights/final
```

Run:

```powershell
python run_inference.py `
  --prompt "Translate 'my elder brother' to Dakota. Return only the answer." `
  --max-tokens 32 `
  --temperature 0 `
  --json
```

Verified output shape:

```json
{
  "backend": "tinker",
  "responses": ["waŋbluŋiŋ"],
  "stop_reasons": ["stop"]
}
```

## Hugging Face Deployment

Published adapter:

```text
HarleyCooper/Qwen3.6-35B-A3B-Dakota1890-GRPO
```

Base model:

```text
Qwen/Qwen3.6-35B-A3B
```

Use a dedicated HF Inference Endpoint:

```powershell
python hf_inference_standalone.py `
  --endpoint-url "https://YOUR-ENDPOINT.endpoints.huggingface.cloud" `
  --prompt "Translate 'my elder brother' to Dakota. Return only the answer." `
  --max-tokens 32 `
  --json
```

The script uses the base tokenizer chat template and disables Qwen thinking by default.

## Adoption Guide

Use [REMOTE_INFERENCE.md](REMOTE_INFERENCE.md) as the canonical guide for external users.
