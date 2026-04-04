# Dakota1890 Pipeline

This is the canonical step-0 pipeline after the repository audit. It separates the primary Dakota RL chain from the secondary SFT comparison path.

## Primary Path

```mermaid
flowchart TD
    A["Riggs 1890 source<br/>grammardictionar00riggrich.pdf<br/>Dictionary/*.jp2"] --> B["JP2 -> JPEG<br/>scripts/extraction/convert_all_images.py<br/>dakota_extraction.tools.image_converter"]
    B --> C["Grammar extraction<br/>scripts/extraction/extract_grammar_pages.py"]
    B --> D["Dictionary extraction<br/>dakota_extraction.run_extraction<br/>or extract_dakota_dictionary_v2.py"]
    C --> E["Organize grammar rules<br/>scripts/rl/organize_grammar_for_rl.py"]
    E --> F["Generate RL tasks<br/>scripts/conversion/convert_rules_to_primeintellect.py"]
    F --> G["Packaged Dakota environment<br/>environments/dakota_grammar_translation"]
    G --> H["Local RL checks<br/>dakota_rl_training/train.py --check-only"]
    G --> I["Remote RL path<br/>dakota_rl_training/tinker_train.py"]
    I --> J["Published adapter<br/>HarleyCooper/Qwen3-30B-ThinkingMachines-Dakota1890"]
    J --> K["Inference surfaces<br/>run_inference.py<br/>hf_inference_standalone.py<br/>huggingface_space/"]
    D --> L["data/extracted/*.json"]
    L --> M["training_dataset_builder / provenance checks"]
```

## Secondary Path

This path stays in the repo as a baseline and educational comparison, not the main story:

```mermaid
flowchart TD
    D["Dictionary extraction<br/>data/extracted/*.json"] --> N["Gemini synthetic QA<br/>scripts/conversion/generate_synthetic_dakota.py"]
    N --> O["data/bilingual_training_set.jsonl"]
    O --> P["OpenAI chat conversion<br/>scripts/conversion/convert_extracted_to_chat.py"]
    P --> Q["OpenAIFineTune/dakota_train.jsonl<br/>OpenAIFineTune/dakota_valid.jsonl"]
    Q --> R["OpenAI readiness / baseline submission<br/>scripts/rl/dakota_openai_finetune.py"]
```

## Canonical Entry Points

- Extraction: `python -m dakota_extraction.run_extraction`
- Grammar extraction: `python scripts/extraction/extract_grammar_pages.py --pages 31-92 --yes`
- RL task generation: `python scripts/conversion/convert_rules_to_primeintellect.py`
- Packaged environment: `from dakota_grammar_translation import load_environment`
- Local RL check: `python dakota_rl_training/train.py --check-only`
- Remote RL path: `python dakota_rl_training/tinker_train.py ...`
- Local inference: `python run_inference.py --prompt "..."`
- HF inference: `python hf_inference_standalone.py --prompt "..."`

## Current Artifact Counts

- Organized grammar rules: `1,497`
- RL tasks: `10,576`
- OpenAI baseline train split: `980`
- OpenAI baseline validation split: `245`

## Notes

- The packaged environment is the Dakota Grammar Gym. It wraps the RL task dataset plus the reward rubric and exposes `load_environment()`.
- The published adapter metadata currently points to `Qwen/Qwen3-30B-A3B-Instruct-2507` with LoRA rank `32`.
- The local RL check path uses `Qwen/Qwen3-0.6B` for consumer-hardware validation.
