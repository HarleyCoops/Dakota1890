# Hugging Face Inference

This repo publishes the Qwen3.6 Dakota adapter at:

```text
HarleyCooper/Qwen3.6-35B-A3B-Dakota1890-GRPO
```

Use it with base tokenizer:

```text
Qwen/Qwen3.6-35B-A3B
```

The maintained HF client is [hf_inference_standalone.py](../hf_inference_standalone.py). It supports both the shared Hugging Face Inference API and dedicated HF Inference Endpoints, but the shared provider surface currently does not serve this PEFT adapter directly. A dedicated endpoint is the expected HF production path.

## Install

```powershell
python -m pip install -r requirements_hf_inference.txt
```

## Authenticate

```powershell
$env:HF_TOKEN = "your_hf_token"
```

`HUGGINGFACE_TOKEN` is also accepted.

## Dedicated Endpoint

```powershell
python hf_inference_standalone.py `
  --endpoint-url "https://YOUR-ENDPOINT.endpoints.huggingface.cloud" `
  --prompt "Translate 'my elder brother' to Dakota. Return only the answer." `
  --max-tokens 32 `
  --temperature 0.1 `
  --json
```

## Shared Inference API Check

```powershell
python hf_inference_standalone.py `
  --prompt "Translate 'my elder brother' to Dakota. Return only the answer." `
  --json `
  --no-login
```

Current expected result: this may fail with no provider mapping for `text-generation`. That means HF has the adapter artifacts, but the shared Inference API is not serving the PEFT model. Use a dedicated endpoint or the verified Tinker sampler.

## Python Use

```python
from hf_inference_standalone import DakotaInferenceClient

client = DakotaInferenceClient(
    model_id="HarleyCooper/Qwen3.6-35B-A3B-Dakota1890-GRPO",
    base_model_id="Qwen/Qwen3.6-35B-A3B",
    endpoint_url="https://YOUR-ENDPOINT.endpoints.huggingface.cloud",
)

result = client.generate(
    prompt="Translate 'my elder brother' to Dakota. Return only the answer.",
    max_new_tokens=32,
    temperature=0.1,
)

print(result["response"])
```

## Related Guide

See [REMOTE_INFERENCE.md](REMOTE_INFERENCE.md) for the side-by-side Tinker and HF adoption path.
