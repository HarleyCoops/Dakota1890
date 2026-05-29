# Remote Inference

Dakota1890 has two hosted inference paths:

- **Thinking Machines / Tinker sampler**: working now, using the final Qwen3.6 sampler checkpoint from the May 27, 2026 run.
- **Hugging Face infrastructure**: model artifacts are published, but the public shared Inference API does not currently expose a text-generation provider for the PEFT adapter. Use a dedicated HF Inference Endpoint or another HF serving deployment.

No local model loading is required for either path.

## Models

| Surface | Value |
|---|---|
| Base model | `Qwen/Qwen3.6-35B-A3B` |
| HF adapter repo | `HarleyCooper/Qwen3.6-35B-A3B-Dakota1890-GRPO` |
| Tinker sampler | `tinker://1f23df9c-5d88-59d9-a7e8-dd4e169ea7d0:train:0/sampler_weights/final` |
| W&B run | `christian-cooper-us/dakota-rl-grammar/owf98569` |

## Install

For Tinker remote inference:

```powershell
python -m pip install -r requirements-tinker.txt
```

For Hugging Face hosted inference:

```powershell
python -m pip install -r requirements_hf_inference.txt
```

## Credentials

Tinker requires:

```powershell
$env:TINKER_API_KEY = "..."
```

Hugging Face endpoints usually require:

```powershell
$env:HF_TOKEN = "..."
```

`HUGGINGFACE_TOKEN` is also accepted by `hf_inference_standalone.py`.

## Tinker Sampler

The primary remote endpoint is `run_inference.py`.

```powershell
python run_inference.py `
  --prompt "Translate 'my elder brother' to Dakota. Return only the answer." `
  --max-tokens 32 `
  --temperature 0 `
  --json
```

Expected shape:

```json
{
  "backend": "tinker",
  "responses": ["waŋbluŋiŋ"],
  "stop_reasons": ["stop"]
}
```

The script disables Qwen thinking by default. Use `--enable-thinking` only when you explicitly want reasoning-style output.

Useful options:

- `--model-path`: override the Tinker sampler URI.
- `--system-prompt`: override the concise answer-only system prompt.
- `--num-samples`: request multiple completions.
- `--json`: emit machine-readable output for apps and notebooks.

## Hugging Face Endpoint

The HF script is `hf_inference_standalone.py`.

For the shared public Inference API:

```powershell
python hf_inference_standalone.py `
  --prompt "Translate 'my elder brother' to Dakota. Return only the answer." `
  --json `
  --no-login
```

Current status: this fails on the shared HF provider surface because the PEFT adapter repo has no `text-generation` provider mapping. That is expected until HF serves the adapter through a compatible provider.

For a dedicated HF Inference Endpoint:

```powershell
python hf_inference_standalone.py `
  --endpoint-url "https://YOUR-ENDPOINT.endpoints.huggingface.cloud" `
  --prompt "Translate 'my elder brother' to Dakota. Return only the answer." `
  --max-tokens 32 `
  --temperature 0.1 `
  --json
```

The script formats prompts with the base model tokenizer `Qwen/Qwen3.6-35B-A3B` and sends text generation to the configured HF endpoint. It also disables Qwen thinking in the chat template.

## Python Use

Tinker:

```python
import tinker

client = tinker.ServiceClient()
sampler = client.create_sampling_client(
    model_path="tinker://1f23df9c-5d88-59d9-a7e8-dd4e169ea7d0:train:0/sampler_weights/final"
)
tokenizer = sampler.get_tokenizer()
```

Hugging Face:

```python
from hf_inference_standalone import DakotaInferenceClient

client = DakotaInferenceClient(
    model_id="HarleyCooper/Qwen3.6-35B-A3B-Dakota1890-GRPO",
    base_model_id="Qwen/Qwen3.6-35B-A3B",
    endpoint_url="https://YOUR-ENDPOINT.endpoints.huggingface.cloud",
)
result = client.generate("Translate 'my elder brother' to Dakota. Return only the answer.")
```

## Adoption Notes

- Treat Tinker as the currently verified hosted runtime.
- Treat HF as the public artifact and deployment target; it needs a dedicated endpoint until the shared provider surface can serve the PEFT adapter.
- Keep prompts short and explicit. The training run was tuned toward concise grammar-task answers.
- Always preserve UTF-8 output because Dakota orthography uses characters such as `ŋ`, `š`, and `ć`.
