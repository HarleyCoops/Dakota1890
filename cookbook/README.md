# Cookbook 01: teach a model one dictionary, and keep its habit of looking things up

Supervised fine-tuning of `thinkingmachines/Inkling-2-Small` on Stephen R. Riggs, *A Dakota-English Dictionary* (1890),
with [Tinker](https://tinker-docs.thinkingmachines.ai/). Three passes over 69,461 rows took recall from 3% to 84%.
Use of the dictionary lookup tool stayed at 100%. Training cost $33.70.

- **Live demo:** https://huggingface.co/spaces/HarleyCooper/riggs1890-lexgym (source in [`space/`](space/))
- **Tasks:** https://huggingface.co/datasets/HarleyCooper/riggs1890-lexicon-graph
- **Traces and results:** https://huggingface.co/datasets/HarleyCooper/riggs1890-lexgym-traces/tree/main/sft_mixed_thinking_off_2026_11_12/inkling-2-small
- **Training charts:** https://forge.coreweave.com/wandb/christian-cooper-us/riggs1890-lexgym

## Result

Exact match on the same 1,500 tasks for each checkpoint. The numbers are in
[`results_inkling-2-small.json`](results_inkling-2-small.json).

| Measure | Base | Pass 1 | Pass 2 | Pass 3 |
|---|---|---|---|---|
| Recall from memory (600) | 0.030 | 0.295 | 0.637 | **0.835** |
| – Dakota word to English definition | 0.007 | 0.223 | 0.510 | 0.817 |
| – English word to Dakota word | 0.053 | 0.367 | 0.763 | 0.853 |
| Held-out word families (900) | 0.443 | 0.814 | 0.820 | **0.816** |
| – Lookup, one hop | 0.767 | 1.000 | 1.000 | 1.000 |
| – Lookup, two hops | 0.893 | 1.000 | 1.000 | 1.000 |
| – Syllables and accent | 0.327 | 0.867 | 0.913 | 0.907 |
| – Root word | 0.380 | 0.820 | 0.740 | 0.773 |
| – Part of speech | 0.293 | 0.693 | 0.753 | 0.720 |
| – Inflected forms | 0.000 | 0.507 | 0.513 | 0.493 |
| All 1,500 tasks | 0.278 | 0.607 | 0.747 | **0.823** |

How to read this:

- Recall rises with each pass, because recall is memory of the book.
- The held-out scores are flat after pass 1. Those tasks use word families the model never saw and depend on rules.
- The base model already called the tool correctly. Its lookup errors were answers in its own words. Training taught
  it to give the entry text as written.
- On earlier runs with Qwen3.5-4B and Qwen3.6-35B, training on word rows only removed tool use: the one-hop lookup
  score fell from 0.73 and 0.79 to 0.02 and 0.03.

Two things differ between this run and the Qwen runs: the model and the training file. This run does not show which
one protects tool use. A Qwen run on the mixed file is the control.

## Files

| File | Purpose |
|---|---|
| [`build_mixed_sft.py`](build_mixed_sft.py) | Builds the training file: word rows plus solved lookup conversations |
| [`launch_sft_curve_job.py`](launch_sft_curve_job.py) | Starts the whole run as one Hugging Face Job |
| [`serve_check.py`](serve_check.py) | Samples from a checkpoint through the OpenAI-compatible endpoint |
| [`space/`](space/) | Source of the live demo |
| [`earlier_runs/`](earlier_runs/) | Launchers for the earlier RL experiments, kept for the record |

The training and evaluation code is in [`lexgym/`](../lexgym/): `tasks.py`, `train_sft.py`, `evaluate.py`,
`recall_curve.py`, `scoring.py`.

## Step 1: turn the dictionary into tasks

Each entry of the scan was parsed into a headword, syllable division, part of speech, definition, listed forms, and
the word it is derived from. `lexgym/tasks.py` makes eight task types and one lookup index from these fields.

| Task | Question | Train | Evaluation |
|---|---|---|---|
| `gloss` | English definition of a Dakota word | 13,562 | 300 recall |
| `en_to_dak` | Dakota word for an English word | 3,369 | 300 recall |
| `syllabify` | Syllable division and accent mark | 13,990 | 150 held-out |
| `pos` | Part-of-speech abbreviation | 13,985 | 150 held-out |
| `inflect` | Inflected forms listed in the entry | 4,936 | 150 held-out |
| `root` | Word that this word is derived from | 3,060 | 150 held-out |
| `lookup_gloss` | Read the entry, give the definition | 13,562 | 150 held-out |
| `lookup_root_gloss` | Read the entry, read the root's entry, give the root's definition | 2,447 | 150 held-out |

The split is by word family. One family in ten is held out with all its derived words. Recall tasks are different by
design: they ask about words that were in training, so they measure memory.

## Step 2: add solved lookup conversations to the training file

The first training file had 56,120 rows of one question and one answer. Training on it taught the model to answer at
once in every case. The fix is to put the tool in the training data.

```
python cookbook/build_mixed_sft.py --one-hop 3000
```

The builder writes each lookup task as a solved conversation. It needs no teacher model, because the lookup index
holds the entry text and the gold answer.

```
system     You can read the dictionary. To read the entry for a Dakota word, reply with exactly
           one line: 'LOOKUP: <word>'. ... You have 3 lookups.
user       Read the dictionary. The Dakota word aska is derived from another word. What is the
           English definition of that other word?
assistant  LOOKUP: aska
user       ENTRY: a-ska', v. n., to become white on [derived from: ska]
assistant  LOOKUP: ska
user       ENTRY: ska', adj., white; clear, as, mini ska, clear water: clean, new ...
assistant  ANSWER: white; clear, as, mini ska, clear water: clean, new ...
```

Each conversation is replayed against the rules of the evaluation environment and kept only if it scores 1.0. Then
it is written as one row for each assistant turn, so the loss is always on the last assistant message of a row. Each
turn then has the same token prefix in training that the model sees when it generates.

The result is 69,461 rows: 56,120 word rows and 13,341 lookup rows (3,000 one-hop conversations and all 2,447
two-hop conversations).

## Step 3: train on Tinker

Tinker runs the model. Your code runs on a small CPU machine and sends batches to the service. This run used a
Hugging Face Job as that machine. Put `TINKER_API_KEY`, `WANDB_API_KEY` and `HF_TOKEN` in a local `.env` file, then:

```
python cookbook/launch_sft_curve_job.py --model-name thinkingmachines/Inkling-2-Small
```

The job evaluates the base model, trains LoRA for three passes with a checkpoint at the end of each pass, evaluates
each checkpoint, and uploads traces and results.

| Setting | Value | Note |
|---|---|---|
| LoRA rank | 32 | |
| Learning rate | 2e-4, linear decay | Set by hand. `get_lr` in the Tinker cookbook is not calibrated for Inkling yet. |
| Batch size | 128 rows | 540 steps for each pass |
| Loss | Last assistant message | One row for each assistant turn (step 2) |
| Packages | `tinker 0.39.0`, `tinker-cookbook 0.6.1` | Repository commit `a281807` |

### Reasoning off for Inkling

Inkling has one renderer (`tml_v0`) and a reasoning effort value in a system message. The job sets the default
effort to 0 before training and evaluation, so the two use the same prompt:

```
TML=$(python -c "import importlib.util as u; print(u.find_spec('tinker_cookbook.renderers.tml_v0').origin)")
sed -i 's/^DEFAULT_EFFORT: float = 0.9$/DEFAULT_EFFORT: float = 0.0/' "$TML"
```

Effort 0 does not fully stop reasoning in the base model. The base model sampled 1.27 million tokens on the 1,500
evaluation tasks. The tuned model sampled 23,000 tokens on the same tasks.

## Step 4: evaluate each checkpoint

The evaluation set is 150 held-out tasks for each of six types and 600 recall tasks. Temperature is 0. The score is
exact match after the normalisation in `lexgym/scoring.py`. The answer limit in the repository is 96 tokens; this run
raised it to 2,048 so that a base model that still reasons is not cut off before its answer.

## Step 5: sample from the checkpoint

```
python cookbook/serve_check.py "Give the English definition of the Dakota word: iwiśtaŋ"
# ANSWER: to treat well, as a sick person
```

The endpoint is in beta. Checkpoints on this account expire seven days after they are made. Download a checkpoint or
push it to the Hub with the `tinker checkpoint` commands to keep it.

## Cost

| Item | Tokens | Cost |
|---|---|---|
| Training, 1,620 steps | 22.47 M trained | $33.70 (logged) |
| Base evaluation | 1.27 M sampled | about $1.35 (estimate) |
| Three checkpoint evaluations | about 0.07 M sampled | about $0.30 (estimate) |
| Total | | about $35 |

## What did not work

- **Word rows only.** Qwen3.5-4B and Qwen3.6-35B reached 0.392 and 0.528 recall after three passes, and both lost
  the tool.
- **Reinforcement learning with reasoning on.** A 30-step test on Inkling with a 25,000-token limit cost $13.13 and
  showed no learning. A full pass over the tasks was priced at about $4,000.
- **One row for a whole conversation.** Training on all assistant messages of a multi-turn row is wrong for a
  renderer that does not keep the same token prefix between turns.

## Limits

- Recall measures memory of words that were in training. It says nothing about new words.
- The score is exact match against one book in its 1890 spelling. The book has errors and the views of its time.
  A correct score means agreement with Riggs. Dakota speakers are the authority on the language.
- The model never chooses between memory and the tool. The task decides if the tool is available.
- Every number is from one run with one seed.

## Use it for a different language

1. Parse a dictionary into entries with a headword, a definition, and any cross-references.
2. Write word tasks and lookup tasks from the entries, and hold out one word family in ten.
3. Build solved lookup conversations from the index and add them to the word rows, one row for each assistant turn.
4. Train LoRA for three passes and save a checkpoint after each pass.
5. Score each checkpoint on recall, held-out rules, and tool use, and read the three curves together.

An independent project by Christian H. Cooper. It is not a publication of Thinking Machines Lab.
