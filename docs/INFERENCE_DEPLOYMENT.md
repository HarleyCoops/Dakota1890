# Inference & Deployment Guide for Dakota Grammar RL Model

## Model Published
 **Model**: [HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL](https://huggingface.co/HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL)

## Inference Options

### 1. **HuggingFace Spaces** (Recommended for Public Demo)
**Status**:  Ready to deploy

I've created a complete HuggingFace Space setup in `huggingface_space/`:
- `app.py` - Gradio interface
- `requirements.txt` - Dependencies
- `README.md` - Space description

**To deploy:**
1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Select "Gradio" SDK
4. Name it: `HarleyCooper/dakota-grammar-rl-inference`
5. Upload the files from `huggingface_space/`
6. The Space will automatically build and deploy

**Features:**
- Interactive web interface
- Adjustable temperature and max tokens
- Example prompts included
- Free GPU inference (T4) for public spaces

### 2. **Prime Intellect Infrastructure**
**Status**:  Check availability

Prime Intellect may offer:
- **Inference endpoints** - Check their dashboard/docs for API endpoints
- **Model serving** - They may have infrastructure for serving RL-trained models
- **Custom deployments** - Contact Prime Intellect support

**To check:**
- Look for "Inference" or "Serving" sections in Prime Intellect dashboard
- Check their documentation for inference APIs
- The `infer_30b.toml` config suggests they have inference infrastructure

### 3. **Local/Remote Inference**
**Status**:  Working (needs parameter tuning)

The model loads successfully but generation parameters need tuning:
- Current issue: Repetitive outputs
- Solution: Adjust `temperature`, `top_p`, `repetition_penalty`, and stopping criteria

**Quick Test Script**: `scripts/inference/test_model_inference.py`

**To improve:**
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.3,  # Lower = more focused
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.2,  # Add this to reduce repetition
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    stop_strings=["Human:", "User:", "\n\n"],  # Add stopping criteria
)
```

### 4. **HuggingFace Inference API**
**Status**:  Available

You can use HuggingFace's Inference API:
```python
from huggingface_hub import InferenceClient

client = InferenceClient()
response = client.text_generation(
    "HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL",
    prompt="Translate to Dakota: Hello",
    max_new_tokens=64
)
```

**Pricing**: Free tier available, paid tiers for higher throughput

### 5. **Custom API Server**
**Status**: Can be built

You can deploy your own API server using:
- **FastAPI** + **vLLM** (for efficient serving)
- **Flask** + **transformers** (simpler setup)
- **TGI** (Text Generation Inference) - HuggingFace's production server

## Current Inference Test Results

 **Model loads successfully** from HuggingFace Hub
 **Chat format works** with system prompts
 **Generation needs tuning** - outputs are repetitive

**Next Steps for Better Inference:**
1. Experiment with `repetition_penalty` parameter
2. Add proper stopping criteria
3. Fine-tune temperature and top_p values
4. Consider post-processing to extract clean responses

## Recommended Deployment Path

1. **Short-term**: Deploy HuggingFace Space for public demo
2. **Medium-term**: Tune generation parameters and update Space
3. **Long-term**: 
   - Check Prime Intellect inference options
   - Or deploy custom API server if needed
   - Consider using vLLM for production serving

## Files Created

- `scripts/inference/test_model_inference.py` - Local inference test script
- `huggingface_space/app.py` - Gradio interface for Spaces
- `huggingface_space/requirements.txt` - Dependencies
- `huggingface_space/README.md` - Space description

## Notes

- The model was trained with RL on Dakota grammar tasks
- It expects chat format with system prompts
- Generation parameters may need adjustment for best results
- The model preserves Dakota orthography (special characters)

