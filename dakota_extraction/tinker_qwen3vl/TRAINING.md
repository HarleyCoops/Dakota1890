# Training options (LoRA / FT) for Qwen3-VL on dictionary pages

This repo’s extraction prompt (`dakota_extraction/core/extraction_prompt.py`) is already a strong **supervised target**: image → structured JSON (`DictionaryEntry` fields).

## What to fine-tune

### Recommended: LoRA (SFT) on the extraction task

- **Goal**: maximize character-level fidelity (Dakota diacritics/special glyphs) + schema compliance.
- **Why LoRA**: cheapest + safest way to adapt very large Qwen3-VL checkpoints.
- **Data**: `(page_image, extraction_prompt) -> gold_json`.

Practical recipe:
1. Run extraction on N pages.
2. Manually correct the JSON (especially the Dakota headwords and inflected forms).
3. SFT on corrected examples (LoRA rank 16–64).
4. Re-run extraction on held-out pages; iterate.

### Full fine-tune

Only consider after LoRA plateaus and you have substantial curated data (and budget). It’s higher risk for:
- catastrophic forgetting of general OCR/vision priors
- overfitting to this one book’s typography

## Tinker-specific notes

- Tinker provides LoRA training clients via `ServiceClient.create_lora_training_client(...)`.
- Vision inputs use `ModelInput(chunks=[ImageChunk(...), EncodedTextChunk(...)])`.
- The current `tinker_cookbook` RL utilities in this repo explicitly note multimodal support is TODO; for supervised vision SFT you’ll likely build a small custom training loop that creates `tinker.types.Datum` for `loss_fn="cross_entropy"` and uses the `TrainingClient.forward_backward(...)` + `TrainingClient.optim_step(...)` APIs.

## Unicode fidelity strategy

If the goal is **exact glyph reproduction**:
- Keep `ensure_ascii=False` everywhere (already done in the extraction pipeline).
- Add a verifier pass that flags any headword/inflected form containing suspicious substitutions (e.g., missing carons/eng/glottal stop).
- Consider augmenting supervision with a **character-level “must preserve” list** per page (the prompt can require the model to echo detected special glyphs).

## After OCR/extraction: Q/A pair development

Once you have stable page→JSON extraction:
- Build downstream text-only datasets from the extracted entries (translation pairs, morphology prompts, etc.).
- Fine-tune a smaller text model for interactive Q/A (cheaper than keeping a vision model in the loop).

