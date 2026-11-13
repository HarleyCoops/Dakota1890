---
title: Know It or Look It Up
emoji: 📖
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
license: apache-2.0
short_description: Inkling 2 Small tuned on an 1890 Dakota dictionary
tags:
  - dakota
  - low-resource-language
  - tinker
  - inkling
  - fine-tuning
  - tool-use
language:
  - en
  - dak
datasets:
  - HarleyCooper/riggs1890-lexicon-graph
  - HarleyCooper/riggs1890-lexgym-traces
---

# Know It or Look It Up

Live comparison of `thinkingmachines/Inkling-2-Small` before and after supervised fine-tuning on
Stephen R. Riggs, *A Dakota-English Dictionary* (1890).

- **From memory:** the model answers with no tool.
- **With the dictionary:** the model can read up to 3 entries with a lookup tool, then answers.

The first column shows the printed entry from the lookup index, so each answer can be checked.

## How it runs

- `app.py` is a FastAPI server. It calls Tinker's OpenAI-compatible endpoint for the base model and for the
  tuned checkpoint. The key is the Space secret `TINKER_API_KEY`.
- `engine.py` holds the prompts and the lookup tool. They are the same as in training (`lexgym/tasks.py`).
- `examples.json` holds 48 recorded evaluation examples. `build_examples.py` makes it from the public traces.
  The examples work when live answers are off.
- Limits: 15 live questions per visitor per hour and 600 per day (`LEXGYM_VISITOR_LIMIT`, `LEXGYM_DAY_LIMIT`).
- `LEXGYM_TUNED` sets the checkpoint path. Tinker checkpoints expire, so set this variable to a new path
  when the checkpoint is replaced.

## Result (exact match, 1,500 tasks)

| Measure | Base | After 3 passes |
|---|---|---|
| Recall from memory | 0.030 | 0.835 |
| Held-out word families | 0.443 | 0.816 |
| Lookup, one hop | 0.767 | 1.000 |
| Lookup, two hops | 0.893 | 1.000 |

Training cost $33.70 on Tinker. The method is in the
[cookbook](https://github.com/HarleyCoops/Dakota1890/tree/main/cookbook).

An independent project by Christian H. Cooper. It is not a publication of Thinking Machines Lab.
A correct answer means agreement with one book from 1890. Dakota speakers are the authority on the language.
