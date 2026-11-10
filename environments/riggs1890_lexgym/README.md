# riggs1890-lexgym

One dictionary volume, compiled into a reinforcement-learning environment.

- **Source:** Stephen R. Riggs, *A Dakota-English Dictionary* (1890), public domain.
- **Tasks:** 68,911 training tasks and 5,884 evaluation tasks, in 8 task types.
- **Rewards:** computed by code from the final answer line. No LLM judge.
- **Provenance:** every task carries the entry id, the scan page number, and the page image name.
- **Code and data pipeline:** https://github.com/HarleyCoops/Dakota1890 (`lexgym/`)

## Design

### 1. From scans to a lexicon graph

A vision model read 336 scan pages into 24,224 structured entries. Riggs printed the dictionary in two halves, and the graph keeps both:

| Half | Scan pages | Entries |
|---|---|---|
| Dakota → English | 95–370 | 16,980 |
| English → Dakota | 371–430 | 7,244 |

The graph builder applies four rules.

1. **Canonical orthography.** The vision model wrote one printed letter with several Unicode characters. A fixed table maps the variants to one form. A form that still has a character outside the Riggs alphabet stays in the graph, but it is never a gold answer.
2. **Trusted headwords only.** A headword is used for the syllable task only if its printed syllable division and its single accent mark parse without error (16,227 of 16,980).
3. **Two witnesses.** A Dakota form in the English half that is also a headword in the Dakota half has two independent readings. English→Dakota tasks accept only these forms.
4. **Word families.** A headword and all words derived from it are one family (12,135 families). One family in ten is held out. No held-out family appears in a training row.

### 2. Task types

| Task | Turns | The model gets | The model must give | Reward |
|---|---|---|---|---|
| `syllabify` | 1 | a plain word | Riggs' syllable division and accent (`a-hi'-yo-ka-sin`) | 1.0 if exact. Otherwise 0.6 × boundary F1 + 0.2 if the stress position is right. 0 if any letter changes. |
| `pos` | 1 | a word | Riggs' part-of-speech abbreviation | 1.0 if it matches, else 0 |
| `inflect` | 1 | a word and its part of speech | the inflected forms that Riggs lists | F1 over the set of forms |
| `root` | 1 | a derived word | the word or words it is derived from | F1 over the set of forms |
| `gloss` | 1 | a Dakota word | the English definition | F1 over content words |
| `en_to_dak` | 1 | an English word and its part of speech | a Dakota word | 1.0 if in the accepted set. 0.7 if only n / ŋ differs. Otherwise 0.25 × character-bigram overlap. |
| `lookup_gloss` | up to 4 | a Dakota word and a dictionary tool | the English definition | F1 over content words, on the final answer only |
| `lookup_root_gloss` | up to 4 | a derived word and a dictionary tool | the English definition of the word it is derived from | F1 over content words, on the final answer only |

Counts:

| Task | Train | Held-out families | Recall |
|---|---|---|---|
| `syllabify` | 13,990 | 1,446 | |
| `pos` | 13,985 | 1,443 | |
| `gloss` | 13,562 | | 300 |
| `lookup_gloss` | 13,562 | 1,401 | |
| `inflect` | 4,936 | 482 | |
| `en_to_dak` | 3,369 | | 300 |
| `root` | 3,060 | 287 | |
| `lookup_root_gloss` | 2,447 | 225 | |

### 3. The lookup tool

In the two lookup task types the model may reply with one line, `LOOKUP: <word>`. The environment returns the printed entry as the next user message:

```
ENTRY: ya-źuŋ', v. a., to pull up by the roots with the mouth, as birds do corn [forms: mdaźuŋ] (scan page 355)
```

The model has 3 lookups. `lookup_root_gloss` needs two: the first entry names the root, and the second entry holds the answer. An episode that ends on a `LOOKUP` line has no answer and scores 0. A base model can learn these tasks with no knowledge of Dakota, and they also work for word families that were held out of training.

### 4. Reward rules that hold for every task

- Only the final answer span is scored: the text after the last `ANSWER:`, or the last non-empty line if there is none. Gold text inside reasoning earns nothing. An unclosed `<think>` block has no answer.
- The reward is in [0, 1]. Nothing multiplies it except the length guard: an answer span more than 3 times the length of the gold answer (and over 24 characters) is scaled by 3 ÷ ratio. This stops word stuffing.
- There is no difficulty multiplier and no hint channel.

### 5. Metrics

| Metric | Weight | Meaning |
|---|---|---|
| `lexgym_reward` | 1.0 | the training reward |
| `exact_match` | 0 | 1.0 if the answer is exactly correct |
| `parse_ok` | 0 | 1.0 if the model wrote an `ANSWER:` line |
| `length_mult` | 0 | the length guard multiplier |
| `lookups_used` | 0 | number of dictionary reads |
| `target_entry_seen` | 0 | 1.0 if the model read the entry that holds the answer |

### 6. Two evaluation splits with different meanings

- `heldout_rule` uses word families that training does not contain, and only task types that a rule or the lookup tool can answer. It measures generalization.
- `recall` uses training vocabulary. A new word cannot be guessed, so this split measures memory of the volume.

## Use

```bash
# from a checkout of the repository
uv pip install -e environments/riggs1890_lexgym
# from the Environments Hub, after publication
prime env install <owner>/riggs1890-lexgym
```

```python
import verifiers as vf

env = vf.load_environment("riggs1890-lexgym")                       # all tasks
env = vf.load_environment("riggs1890-lexgym", tasks=["lookup_gloss", "lookup_root_gloss"])
```

| Argument | Default | Meaning |
|---|---|---|
| `tasks` | all eight | task types to keep |
| `max_train_per_task` | -1 | cap on training tasks per type |
| `max_eval_per_task` | 50 | cap on evaluation tasks per type and split |
| `eval_splits` | `("heldout_rule", "recall")` | evaluation splits to load |
| `seed` | 42 | shuffle seed |

## First results

One pilot on `Qwen/Qwen3.5-4B` (LoRA rank 32, Thinking Machines Tinker), 40 evaluation tasks per type, one seed. Exact-match rate:

| Task | Split | Base model | After supervised stage | After supervised + 30 RL steps |
|---|---|---|---|---|
| `syllabify` | held-out | 0.00 | 0.83 | 0.83 |
| `pos` | held-out | 0.05 | 0.38 | 0.73 |
| `root` | held-out | 0.00 | 0.48 | 0.58 |
| `inflect` | held-out | 0.00 | 0.30 | 0.38 |
| `lookup_gloss` | held-out | 0.50 | 0.00 | 0.85 |
| `lookup_root_gloss` | held-out | 0.10 | 0.00 | 0.63 |

RL from a base model does not learn the single-turn lexical tasks, because the model cannot sample a word that it does not know. Use a supervised warm start on the same volume first (`lexgym/train_sft.py` in the repository), then RL.

## Tests

`tests/e2e_scripted_model.py` runs the environment end to end against a scripted OpenAI-compatible server with four policies: an oracle (reward 1.0 on all eight task types, with 1 and 2 lookups on the lookup tasks), a wrong answer, the gold answer hidden in a thinking block (reward 0), and a model that only sends `LOOKUP` (reward 0 after 3 lookups).

## Limits

- The source is a missionary dictionary from 1890. It records one written variety and is not a description of Dakota as it is spoken now.
- The extraction has errors. 1,494 forms in the English half differ from a headword only by n / ŋ, which is likely a reading error. The scorer pays 0.7 for that difference in `en_to_dak` and records it.
- A model that is correct for this book is a starting point for correction by Dakota speakers. It does not replace them.
