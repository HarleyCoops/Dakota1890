# Model Not Available on Inference API - Solutions

## The Issue

Your model `HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL` is not currently available on Hugging Face's Inference API. This is common for newly published models.

## Solutions (in order of recommendation)

### Option 1: Use HuggingFace Space (Recommended - Already Set Up!)

You already have a complete Space setup in `huggingface_space/`:

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Select "Gradio" SDK
4. Name it: `HarleyCooper/dakota-grammar-rl-inference`
5. Upload files from `huggingface_space/`:
   - `app.py`
   - `requirements.txt`
   - `README.md`
6. The Space will automatically build and deploy with free GPU

**Advantages:**
- Free GPU inference (T4)
- Interactive web interface
- No local setup needed
- Public demo ready

### Option 2: Enable Model for Inference API

1. Go to your model page: https://huggingface.co/HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL
2. Go to Settings → Inference
3. Enable "Inference API" if available
4. Some models may need manual approval from HF

### Option 3: Use Inference Endpoints

Create a dedicated Inference Endpoint:

1. Go to https://huggingface.co/inference-endpoints
2. Create new endpoint
3. Select your model: `HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL`
4. Choose instance type (GPU)
5. Get the endpoint URL
6. Use with script: `python hf_inference_standalone.py --prompt "..." --endpoint-url "https://your-endpoint.hf.space"`

**Note:** Inference Endpoints are a paid service.

### Option 4: Local Inference (Requires GPU)

Use the existing `scripts/inference/test_model_inference.py` script:

```powershell
python scripts/inference/test_model_inference.py
```

**Requirements:**
- Local GPU (CUDA)
- ~2GB VRAM for 0.6B model
- PyTorch with CUDA support

## Quick Fix: Use the Space

The easiest solution is to deploy your existing Space. It's already configured and ready to go!

See `huggingface_space/DEPLOYMENT.md` for detailed instructions.

