# GPU Requirements for HF Inference

## No Local GPU Needed! 

The `hf_inference_standalone.py` script uses **Hugging Face's Inference API**, which runs entirely on HF's cloud infrastructure. You don't need a local GPU at all.

## How It Works

1. **Tokenization (Local)**: The script downloads the tokenizer locally to format your prompt using the chat template. This is just text processing - no GPU needed.

2. **Inference (Cloud)**: The formatted prompt is sent to Hugging Face's servers, which run the model on their GPUs.

3. **Response**: The generated text is returned to you.

## What Gets Downloaded Locally

- Tokenizer files (~15MB total):
  - `tokenizer_config.json`
  - `vocab.json`
  - `merges.txt`
  - `tokenizer.json`
  - `chat_template.jinja`
  - `special_tokens_map.json`

These are just configuration files - no model weights, no GPU needed.

## Model Weights Stay on HF Servers

The actual model weights (~0.8B parameters) stay on Hugging Face's servers. You never download them locally.

## If You Want Local Inference

If you want to run inference locally (requires GPU), use `scripts/inference/test_model_inference.py` instead. But for the HF infrastructure script, no GPU is needed!

