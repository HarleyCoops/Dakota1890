# Dakota1890

**A language model that read the 1890 Dakota grammar cover to cover — and is now ready for the community to teach it the rest.**

![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Method](https://img.shields.io/badge/RL-GRPO-black.svg)
![Source](https://img.shields.io/badge/source-Riggs%201890-brightgreen.svg)

To our knowledge, this is the first time a model of this size has been trained from a **single historical source document** for an **endangered language**.

The model is not always right. That is not the point.

This is the working endpoint that modern descendants of Dakota speakers will correct — the way you correct a toddler learning to speak. The toddler has read the 1890 grammar cover to cover. It knows the orthography, the affixes, the morphology of the book. Now the community teaches it the rest. That community-in-the-loop second stage is the idea carried over from the companion **StoneyNakoda** project, where contemporary speakers correct and extend the model's output.

The whole pipeline runs from one public-domain book: Stephen Return Riggs' 1890 *Grammar and Dictionary of the Dakota Language*. Book → extraction → grammar rules → RL tasks → GRPO training → community correction.

---

## November 2026 update: the Lexicon Gym

This section describes the current direction of the project. The sections after it describe the 2025 to mid-2026 runs and are kept as the record of that work.

### Why the experiment changed

We audited the reward function, closed its leaks, and ran the clean version again on Tinker. The reward is now honest. The honest result showed a structural problem.

| Run (Qwen3-30B-A3B, LoRA 32, about 200 steps) | Held-out exact match, start → end | English→Dakota exact match |
|---|---|---|
| `tinker_qwen30b_grant_clean` | 0.06 → 0.24 | 0.00 → 0.03 |
| `dakota1890-grant-clean-v3d` | 0.07 → 0.11 | 0.00 → 0.00 |

Policy-gradient RL can only reinforce an answer that the model samples. A base model that does not know a Dakota word does not sample it, so the lexical tasks give no gradient. The gains in these runs are on Dakota→English, where the model answers in English.

The audit also found two data problems:

- The RL environment used only the 1,497 grammar rules. The 24,224 structured dictionary entries that we had already extracted from the scans were not in the environment.
- 1,307 of the 10,576 grammar tasks contained unfilled template text, and the task set had only 5,638 unique answers.

### What changed in the repository

A new package, `lexgym/`, builds everything from the per-page extraction files in `data/extracted/`. It does not depend on the older task files.

| File | Function |
|---|---|
| `lexgym/orthography.py` | Riggs alphabet, mapping of variant glyphs to one form, headword parser (syllables and accent) |
| `lexgym/build_graph.py` | Extraction JSON → lexicon graph with page-level provenance, word families, and splits |
| `lexgym/tasks.py` | Graph → supervised rows, RL tasks, held-out and recall evaluation sets, lookup index |
| `lexgym/scoring.py` | One deterministic verifier per task type and one reward ledger per rollout |
| `lexgym/tinker_env.py` | `tinker-cookbook` `Env`, `EnvGroupBuilder`, and `RLDatasetBuilder`, including the multi-turn lookup tool |
| `lexgym/train_sft.py` | Stage 1: supervised warm start on Tinker |
| `lexgym/train_rl.py` | Stage 2: RL on Tinker, started from the stage 1 checkpoint |
| `tests/test_lexgym.py` | Offline tests: scoring, reward-hack probes, gold answers, split leakage |

```bash
python -m lexgym.build_graph      # data/extracted -> data/lexicon_graph
python -m lexgym.tasks            # data/lexicon_graph -> data/lexgym_tasks
python -m pytest tests/test_lexgym.py
```

`requirements-tinker.txt` now pins `tinker==0.39.0` and `tinker-cookbook==0.6.1`. The Tinker server no longer accepts the SDK version that the earlier runs used.

### Dataset design: one volume, two witnesses

Riggs printed the dictionary in two halves. Scan pages 95 to 370 are Dakota→English and pages 371 to 430 are English→Dakota. The graph keeps both halves and uses each half to check the other.

| Item | Count |
|---|---|
| Extracted entries (336 scan pages) | 24,224 |
| Dakota→English entries | 16,980 |
| – headword with a trusted syllable division and accent | 16,227 |
| – with inflected forms | 5,583 |
| – with a derivation link that resolves to another headword | 3,657 links |
| English→Dakota entries | 7,244 |
| Dakota forms in the English half | 10,779 |
| – that match a headword in the Dakota half exactly | 5,279 |
| – that differ from a headword only by n / ŋ | 1,494 |
| Word families (a headword and all words derived from it) | 12,135 |

Design rules:

- **Provenance.** Every entry, edge, and task carries `entry_id`, the scan page number, and the image file name. Each fact can be checked against the page.
- **Canonical orthography.** The vision model wrote one printed letter with several Unicode characters (four different characters for ŋ, and `é` where the book prints `ć`). `orthography.py` maps these variants to one form and lists each mapping as a hypothesis about the extraction, not about the language. A form that still contains a character outside the Riggs alphabet is kept in the graph but is never used as a gold answer.
- **Two-witness gold.** For English→Dakota, RL and evaluation use only answers that appear in both halves of the book. The 1,494 near matches are a work list for a targeted second reading of those pages.
- **Splits by word family.** One family in ten is held out. A held-out headword and its derived forms do not appear in any training row. The test suite checks that the family overlap is zero.

### The RL environment

| Task | Input → answer | Verifier | RL train | Held-out |
|---|---|---|---|---|
| `syllabify` | plain word → Riggs syllable division and accent | exact; partial credit for correct boundaries and stress; zero if a letter changes | 13,990 | 1,446 |
| `pos` | word → Riggs part-of-speech abbreviation | exact, closed set | 13,985 | 1,443 |
| `gloss` | Dakota word → English definition | content-token F1 | 13,562 | recall set |
| `inflect` | word → the inflected forms that Riggs lists | set F1 | 4,936 | 482 |
| `root` | derived word → the word it is derived from | set F1 | 3,060 | 287 |
| `en_to_dak` | English word → Dakota word | member of the accepted set; 0.7 if only n / ŋ differs | 3,369 | recall set |
| `lookup_gloss` | multi-turn, 1 lookup needed | content-token F1 on the final answer | 13,562 | 1,401 |
| `lookup_root_gloss` | multi-turn, 2 lookups needed | content-token F1 on the final answer | 2,447 | 225 |

- **Supervised set:** 56,120 chat rows from the six single-turn task types.
- **RL set:** 68,911 tasks.
- **Two evaluation sets with different meanings.** `eval_heldout_rule` (5,284 tasks) uses word families that training does not see, and contains only tasks that a rule or the lookup tool can answer. It measures generalization. `eval_recall` (600 tasks) uses training vocabulary. A new word cannot be guessed, so this set measures memory of the volume, and we report it under that name.
- **Lookup tool.** In the lookup tasks the model can reply `LOOKUP: <word>` up to three times. The environment returns the printed entry as the next message. Reward is paid only on the final `ANSWER:` line. The ledger records the number of lookups and whether the model reached the target entry. These tasks can be learned without prior knowledge of Dakota, and they train the behaviour that a speaker-correction tool needs: read the source, then answer.
- **Reward rules.** No LLM judge. Only the final answer span is scored, so gold text inside reasoning earns nothing. Reward is in [0, 1] with a length guard and no difficulty multiplier. Every component goes to the ledger, so each reward can be reconstructed by hand. The tests include probes for word stuffing, prompt echo, and gold-in-reasoning.

### How we use Thinking Machines Tinker and the cookbook

Tinker is the only training infrastructure in the current plan. The training loop runs on a CPU machine, and Tinker does the forward and backward passes.

1. **Stage 1, supervised warm start.** `lexgym/train_sft.py` uses `tinker_cookbook.supervised.train` with `FromConversationFileBuilder` on `sft_train.jsonl`. LoRA rank 32, cross-entropy on the assistant message, the `qwen3_5_disable_thinking` renderer so that the target is one short line. This stage gives the model the vocabulary that RL cannot find by exploration.
2. **Stage 2, RL.** `lexgym/train_rl.py` uses `tinker_cookbook.rl.train` with our own `Env`, `EnvGroupBuilder`, and `RLDatasetBuilder`. It starts from the stage 1 state through `load_checkpoint_path`. Each task is sampled as a group of rollouts and the loss is the cookbook's importance-sampling policy gradient with group-centred advantages. The multi-turn lookup environment follows the pattern of the cookbook's `search_tool` recipe.
3. **Evaluation.** The dataset builder returns the held-out and recall sets as the test dataset, so the cookbook's evaluation loop reports each task and each split separately at a fixed interval.
4. **Monitoring.** The cookbook logger writes `metrics.jsonl` and sends the same metrics, including every ledger channel, to Weights & Biases.
5. **Publication.** We download the trained LoRA weights from Tinker, convert them to a PEFT adapter, and publish the adapter, the dataset, and the rollout traces on Hugging Face. The environment will also be published in `verifiers` format for the Prime Intellect Environments Hub.

Model plan: `Qwen/Qwen3.5-4B` for pilots and ablations, then `Qwen/Qwen3.6-35B-A3B` for the main run, which permits a direct comparison with our earlier 35B adapter.

### Plan

1. **Pilot (in progress).** Short supervised and RL runs on the 4B model to verify the loop, the ledger, and the cost per step.
2. **Main experiment.** Three arms on the same base model and the same evaluation sets: RL only, supervised only, supervised then RL. The question is how much RL adds after the model knows the lexicon, and on which task types.
3. **Ablations.** Lookup tool on and off; two-witness gold against all gold; curriculum order.
4. **Data repair.** A second reading of the pages behind the 1,494 near matches and the headwords that failed the parser, then a new graph version.
5. **Grammar section.** Rebuild the grammar tasks from the 1,497 rules with the same verifier standard and link each rule to the dictionary entries that show it.
6. **Second volume.** Apply the same pipeline to Watkins' *A Dictionary of the Cree Language* (1865), which has the same two-half structure (see `docs/cree1865_source_dossier/`), and test whether a model trained on one volume learns a second volume faster.
7. **Cookbooks.** One short runnable guide per stage (extract, graph, supervised, RL, evaluate), written so that another group can apply the method to its own source volume.
8. **Speaker correction.** A model that is correct for the 1890 book is the starting point for correction by present-day Dakota speakers. That stage needs community authority and review, and the book-trained model does not replace it.

### Limits

- The source is a missionary dictionary from 1890. It records one written variety and is not a description of Dakota as it is spoken now.
- The extraction has errors. The orthography mapping removes some of them and hides none: the raw form stays in every row.
- The pilot results are not in this document yet. The numbers in the first table are from the earlier clean runs.

---

## The model progression

Each model is published on Hugging Face. The arc runs from a 0.6B proof of concept to a 35B GRPO run.

| Model | Params | Method | Infra | Notes |
|---|---|---|---|---|
| [`Qwen3-0.6B-Dakota-Grammar-RL`](https://huggingface.co/HarleyCooper) | 0.6B | GRPO | PrimeIntellect | First proof: grammar rules as a reward signal |
| [`Qwen3-0.6B-Dakota-Grammar-RL-400`](https://huggingface.co/HarleyCooper) | 0.6B | GRPO | PrimeIntellect | 400-step run; 150% reward improvement |
| [`Qwen3.6-35B-A3B-Dakota1890-GRPO`](https://huggingface.co/HarleyCooper/Qwen3.6-35B-A3B-Dakota1890-GRPO) | **35B** | **GRPO** | Tinker | **Latest** — published most recently |

The latest model, **Qwen3.6-35B-A3B-Dakota1890-GRPO**, is the current working endpoint — the toddler that has read the book. It is a live research checkpoint, not an authoritative Dakota assistant.

---

## The pipeline

One book becomes a self-contained training loop. No parallel corpus, no separate grammar documentation, no OCR training.

```
1890 Riggs grammar (PDF, public domain)
        │
        ▼
  VLM extraction          Claude reads the scans; preserves ŋ š ć ḣ exactly
        │
        ▼
  Grammar rules           1,497 rules across morphology, syntax, phonology,
        │                 conjugation, particles, translation
        ▼
  RL tasks                10,576 verifiable tasks (≈7.1 per rule):
        │                 morphology, translation, reverse translation,
        │                 syntax analysis, pattern ID
        ▼
  GRPO training           PrimeIntellect / Tinker; the reward IS the grammar
        │
        ▼
  Community correction     descendants of Dakota speakers correct the output —
                           the StoneyNakoda community-in-the-loop stage
```

The key move: grammar rules stop being static documentation and become **executable feedback**. Instead of asking a model to imitate text, the RL loop scores whether each output satisfies the orthographic, morphological, and task-level constraints pulled from the book.

---

## The reward function

The reward is **deterministic**. There is no LLM judge. Every component is independently verifiable, so the gradient is honest and you can see exactly what the model got wrong.

It decomposes a qualitative linguistic task into three measurable primitives:

```python
reward = (
    0.4 * character_preservation +   # orthography: ŋ š ć ḣ ṡ á é í ó ú preserved?
    0.4 * affix_accuracy +           # morphology: correct prefixes/suffixes applied?
    0.2 * semantic_correctness       # semantics: meaning preserved vs. ground truth?
) * difficulty_multiplier            # curriculum weight, 1.0x → 2.0x
```

- **Orthography** — recall of the required special Unicode characters against the source.
- **Morphology** — regex / pattern checks against specific extracted grammar rules (affix presence, possessives like `-ku`, `-ću`, `-tku`, plural `-pi`).
- **Semantics** — similarity to the ground-truth translation or dictionary lookup.

Because each piece is checkable by code rather than judgment, GRPO gets dense, multi-dimensional feedback on **structure**, not just imitation — which is why an RL approach works on a task usually considered too qualitative for it.

---

## See the proof

Everything above is the pitch. Below is what it actually looks like — the source book, the training curves, and the reward signal running live.

### The one book

It all starts with a single public-domain scan: Riggs' 1890 *Grammar and Dictionary of the Dakota Language*.

<img src="Public/grammar.jpg" alt="Grammar of the Dakota Language — 1890 title page" width="100%">

The grammar section (pages 31–92) and the dictionary section (pages 93–440) are the entire training corpus. No parallel text, no modern annotations.

| | |
|---|---|
| Dictionary entries → `{dakota:english}` pairs | ![Dakota dictionary scan](Public/Dictionary.jpeg) |
| Morphology and prepositions → extracted rules | ![Dakota prepositions scan](Public/Prepositions.jpg) |

![Dakota1890 dictionary collage](hf_model_card_work/Qwen3.6-35B-A3B-Dakota1890-GRPO/assets/dakota1890_dictionary_collage.png)

### The latest run — Qwen3.6-35B-A3B GRPO

The reward channels of the most recent 35B run, restored end to end: composite reward climbs, the pattern channel goes live after the schema fix, char-overlap rises while affix accuracy stays high, and the ledger audit confirms `composite_diff` stayed at zero.

![Qwen3.6-35B full run dashboard](wandb_analysis/qwen36_35b_full_rerun_20260527/qwen36_dakota_full_run_dashboard.png)

| Composite reward progression | Reward components |
|---|---|
| ![Composite reward progression](wandb_analysis/qwen36_35b_full_rerun_20260527/qwen36_reward_progression.png) | ![Reward components](wandb_analysis/qwen36_35b_full_rerun_20260527/qwen36_reward_components.png) |

![Pattern reward channel](wandb_analysis/qwen36_35b_full_rerun_20260527/qwen36_pattern_channel.png)

### The reward, running live

The deterministic reward is not a black box. Every component — character overlap, affix accuracy, semantic match, difficulty multiplier — is logged per step. Here is a single Tinker training step with the full ledger broken out:

<img src="Public/TrainScreenShot.jpg" alt="Live Tinker training metrics, single step with full reward ledger" width="70%">

![Reward ledger reconciliation](wandb_analysis/reward_ledger.png)

### The method, end to end

How grammar rules become an executable reward function, file by file:

![Methodology overview](Public/overview.jpg)

![Dakota grammar gym methodology diagram](docs/diagram.png)

---

## Source material

Everything derives from one public-domain book:

> **Riggs, S. R. (1890).** *A Grammar and Dictionary of the Dakota Language.* Washington: Government Printing Office.

- ~665 page scans from the Internet Archive
- Grammar section (pages 31–92) → 1,497 rules
- Dictionary section (pages 93–440) → ~10,000 `{dakota:english}` pairs

Dakota is a Siouan language with a rich orthography — special consonants (`ć`, `š`, `ŋ`, `ḣ`), pitch accents (`á é í ó ú`), and agglutinative morphology. The special characters are what make the reward *verifiable*: they give an unambiguous signal of whether the model preserved the language's structure.

The code is Apache-2.0. The 1890 text is public domain (see `DATA_LICENSE.md`).

### Datasets

- [`HarleyCooper/adaption-dakota-english-qa`](https://huggingface.co/datasets/HarleyCooper/adaption-dakota-english-qa) — 1,950 bilingual Dakota–English QA pairs.
- **Stoney10kRL** — 10K entries from the companion Stoney Nakoda work, applying the same method to a related Siouan language.

---

## How to use

The latest adapter loads on top of its base model with PEFT:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "Qwen/Qwen3.6-35B-A3B"
adapter = "HarleyCooper/Qwen3.6-35B-A3B-Dakota1890-GRPO"

model = AutoModelForCausalLM.from_pretrained(base, device_map="auto", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(base)
model = PeftModel.from_pretrained(model, adapter)

messages = [
    {"role": "system", "content": "You are a Dakota language expert."},
    {"role": "user", "content": "Translate 'my elder brother' to Dakota using the correct possessive suffix."},
]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
out = model.generate(**tok(text, return_tensors="pt").to(model.device), max_new_tokens=128)
print(tok.decode(out[0], skip_special_tokens=True))
```

Treat the output as a toddler's first attempt — a starting point for correction, not a final answer.

**Live demo:** the **StoneyApp** Space on Hugging Face Spaces shows the community-in-the-loop idea running on the related Stoney Nakoda model.

---

## Companion projects

- **StoneyNakoda** — the project this method came from. It runs the community-in-the-loop correction stage: contemporary Stoney Nakoda speakers correct and extend model output, turning a book-trained model into a living one. Dakota1890 reuses that pattern.
- **StoneyApp** — the live Hugging Face Space demo for the Stoney Nakoda model and its correction loop.

The larger claim is methodological: if a low-resource language has a usable historical source and a community willing to run the second stage, this pipeline can be reused rather than rebuilt.

---

## Links

- **Latest model:** https://huggingface.co/HarleyCooper/Qwen3.6-35B-A3B-Dakota1890-GRPO
- **Dataset (Dakota QA):** https://huggingface.co/datasets/HarleyCooper/adaption-dakota-english-qa
- **W&B training logs:** https://wandb.ai/christian-cooper-us
- **All Hugging Face models:** https://huggingface.co/HarleyCooper
- **Source book (Internet Archive):** Riggs 1890, *A Grammar and Dictionary of the Dakota Language* (public domain)

---

## Acknowledgments

- **Stephen Return Riggs** — the 1890 grammar and dictionary.
- **Internet Archive** — the scanned source.
- **PrimeIntellect** and **Thinking Machines (Tinker)** — RL training infrastructure.
- **Anthropic** — VLM extraction.
- **The Dakota and Stoney Nakoda language communities** — who do the part that actually matters: teaching the model the rest.

## License

Code: **Apache-2.0** (`LICENSE`). Historical Dakota text (Riggs, 1890): **public domain** (`DATA_LICENSE.md`).
