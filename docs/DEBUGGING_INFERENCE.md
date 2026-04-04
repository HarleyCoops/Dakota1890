# Debugging Inference Issues

## The Problem

You're getting garbled output like "NS Ｎ ＮＳ Ｎ Ｎ" which suggests:
1. **Token extraction issue** - Wrong tokens being decoded
2. **Repetition problem** - Model stuck in a loop
3. **Decoding issue** - Special tokens not handled correctly
4. **Response extraction bug** - Cutting off at wrong point

## How to Diagnose

### Option 1: Enable Debug Mode (Recommended)

Add this to your Space's environment variables:
- Go to your Space Settings → Variables
- Add: `DEBUG_INFERENCE=true`
- Restart the Space

This will print debug info to the logs showing:
- What tokens were generated
- How the response was extracted
- The full decoded text before cleaning

### Option 2: Check Space Logs

1. Go to your Space → "Logs" tab
2. Look for error messages or warnings
3. Check if the model is loading correctly
4. See if there are tokenization errors

### Option 3: Test Locally First

Run the test script locally to see if it works:
```powershell
python scripts/inference/test_model_inference.py
```

If it works locally but not in Space, it's a Space-specific issue.

## Common Issues & Fixes

### Issue 1: Repetition Penalty Too Low
**Symptom**: Repetitive tokens like "NS Ｎ Ｎ"
**Fix**: Increase `repetition_penalty` to 1.2 or 1.3

### Issue 2: Wrong Token Extraction
**Symptom**: Garbled characters, partial words
**Fix**: The new code extracts tokens directly instead of string slicing

### Issue 3: Model Not Generating Properly
**Symptom**: Very short or empty responses
**Fix**: Check if model loaded correctly, verify GPU is available

### Issue 4: Chat Template Mismatch
**Symptom**: Model generates prompt instead of response
**Fix**: Verify `apply_chat_template` is working correctly

## Quick Test

Try this minimal prompt first:
```
Translate to Dakota: Hello
```

Expected: `Háu` or similar greeting

If this fails, the issue is fundamental (model loading, tokenization, etc.)
If this works but longer prompts fail, it's a generation/extraction issue.

## What the Debug Output Shows

When `DEBUG_INFERENCE=true` is set, you'll see:
```
DEBUG - Formatted prompt length: 245
DEBUG - Full decoded (with special tokens): <|im_start|>system...
DEBUG - Decoded (no special tokens): You are a Dakota...
DEBUG - Output shape: torch.Size([1, 312])
DEBUG - Input length: 248
DEBUG - Output length: 312
DEBUG - Extracted generated text: 'Háu'
DEBUG - Final cleaned text: 'Háu'
```

This helps identify where the problem occurs:
- If "Extracted generated text" is wrong → extraction bug
- If "Full decoded" is wrong → decoding bug
- If output length is same as input → model didn't generate

