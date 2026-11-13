# The Verifier Loop: executable grammar rules as the RL environment

**Spec for the next Dakota1890 experiment.** Version 1.0, 2026-11-12.
Codebase it extends: [`HarleyCoops/Dakota1890`](https://github.com/HarleyCoops/Dakota1890) (`lexgym/`).
Data it reads: [`HarleyCooper/riggs1890-lexicon-graph`](https://huggingface.co/datasets/HarleyCooper/riggs1890-lexicon-graph).

---

## 0. What changed since the first framing

The first framing of this experiment assumed most of the pipeline had to be
built. Reading the repos shows otherwise. Three parts of the design **already
exist** and should not be rebuilt:

| Design element | Already implemented as | Where |
|---|---|---|
| Reverse clues, answers anchored | `en_to_dak` task: English word → Dakota form, scored against the accepted set | `lexgym/tasks.py`, reward in `lexgym/scoring.py` |
| Accepted set = every headword sharing the gloss | **Two-witness gold**: only forms attested in both halves of the book are accepted answers | graph builder rule 3 |
| "May vary the question surface, must not synthesize a new answer" (the Cree contract) | The Cree anchored-expansion run: 490,280 balanced rows, answers dictionary-bound | Cree1865, run `cree1865-synthetic-expansion-v1` |
| Frozen measures by split | `heldout_rule` (5,284 tasks, unseen families) and `recall` (600 tasks, training vocabulary) | `rl_tasks` config |
| Lookup / abstain behavior | `LOOKUP: <word>` tool, reward only on final `ANSWER:` | `lexgym/tinker_env.py` |

What does **not** exist yet, and is the whole delta of this experiment:

1. An **executable morphology checker** — code that can say *"this form is
   well-built by rule X"* and *"this form breaks rule Y"*. The current Dakota
   grammar scoring is soft overlap (character F1, affix presence); it cannot
   reject a form.
2. **Three-label triage** of every synthetic row: attested / rule-licensed /
   rejected, with a promotion rule that keeps rule-licensed rows out of gold.
3. The **generate → verify → retrain loop** that grows the dataset to the
   ceiling of the book, with policy-gradient RL reserved for behavior
   (lookup/abstain) only — the 2026-11-10 runs showed RL adds no facts
   (English→Dakota exact match 0.00→0.03 after ~200 GRPO steps).
4. The **unverified control** that prices the verifier: the Stoney-RL setup
   (model writes the answers, reward placeholder returns 0.0) run as an arm,
   not as an accident.

Also new since the first framing: the Inkling-2-Small SFT run lifted
memory recall from 3% → 83.5% in three passes. The binding constraint on
"the maximum of what is in the book" is now (a) the quality of the extracted
gold and (b) how much surface area the training rows cover — exactly the two
things this loop works on.

---

## 1. The claim under test

> Grammar rules compiled into executable checks can (a) clean the extracted
> gold, (b) label synthetic rows as licensed or rejected, and (c) raise
> held-out recall when rule-licensed rows are added to supervised training —
> without ever promoting an unattested form to gold.

Falsified if: checker-approved rows do not change held-out recall, or if the
rule-violation rate in free output does not fall relative to the baseline arm.

---

## 2. The checker

### 2.1 Rule schema

Every compiled rule is a pure function over a canonical form plus a rule file
that admits it:

```python
@dataclass
class MorphRule:
    name: str            # "pronoun-affix-insertion"
    scope: str           # "v." | "n." | "adj." | ... (Riggs POS abbreviations)
    inventory: list[str] # surface strings, e.g. ["wa", "mda", "ya", ...]
    apply: Callable[[str, str], list[str]]   # (stem, affix) -> candidate surface forms
    notes: str           # citation: Riggs page / grammar section

@dataclass
class Admission:
    rule: str
    n_attested: int      # attested forms the rule explains
    n_explained_total: int
    coverage: float      # n_explained_total / n_attested_in_scope
    false_positives: int # rule-accepted forms NOT attested, sampled and read by hand
    admitted: bool       # coverage >= threshold AND FP rate <= threshold
```

### 2.2 The admission test

A rule is **admitted** only after it passes the book's own attested forms:

- **Coverage gate.** On the attested inflected forms of in-scope entries,
  the rule must explain ≥ 60% (the feasibility probe's simplest rule already
  explains 63% of 6,989 forms; stricter per-rule thresholds are set at
  admission time).
- **Precision gate.** Every form the rule *accepts* that is *not* attested is
  a candidate false positive. A sample is read against the page images; a rule
  whose accepted-but-unattested sample is > 10% wrong builds is not admitted.
- **Family holdout.** Coverage and precision are computed on train families
  first; the rule is then frozen and re-scored on held-out families. A rule
  that only works on families it was tuned on is not admitted.

### 2.3 What v0 checks without any compiled morphology (already shippable)

From `lexgym/orthography.py`, which is grounded and tested:

1. **Alphabet check** — every character in the Riggs alphabet
   (`ALPHABET` set); anything else is `is_suspect` and is never gold.
2. **Shape check** — the form parses as headword syllables + one accent mark
   (`parse_headword`); accent mark terminates its syllable; at most one accent.
3. **Attested check** — the form is in the entry's `inflected_forms`, or is a
   headword, or is in the family's attested set. This is the gold criterion.
4. **Near-match report** — the form differs from an attested form only by
   n/ŋ (1,494 known extraction suspects). Reported, never scored correct.

### 2.4 Milestone 1: run the checker against the gold (first build, no GPU)

The checker's first job is cleaning the gold. The feasibility probe's rule
("inflected form = headword with a pronoun affix inserted, optionally ± `-pi`")
explains 63% of the 6,989 inflected forms; the unexplained forms are a
**triage list**, and the sampled misses are mostly extraction errors (e.g.
`abawamdeca` for a headword printed with ć) or forms attached to the wrong
entry. Output of M1:

- `triage_unexplained.parquet` — one row per unexplained form: entry_id, page,
  source_image, candidate correction, suspected error class.
- `rule_admission.json` — per-rule coverage and precision on train and
  held-out families.
- Every fix flows back into `lexgym/build_graph.py` inputs, which raises the
  ceiling the last SFT run hit.

**v0 audit results (2026-11-13, published):** 16,980 entries, 5,583 with
inflected forms, 7,864 inflected-form instances, **0 alphabet violations**.
Of the instances: 231 attested-as-headword, 39 n/ŋ variants, 1,769
licensed-pending (prefix/suffix/contained), 5,825 unexplained. Relation
distribution over the 7,594 non-attested: other 5,621, suffix 1,228,
contained 485, prefix 56, no-stem 204. Simple contiguity matching explains
only ~23%; the 63% probe figure requires modeling **affix insertion that
splits the stem** — including stem-change classes like `y`→`md` (1st person).

**Affix inventory is M2, not M1.** The 1,497 extracted grammar rules live in
LFS and the pronoun/paradigm tables are Riggs grammar pp. 40–57 (scan pages,
already imaged in `docs/collages/tinker_grammar_tables.jpg`). The inventory
must be read from those sources, not invented. A rule can license a form but
not choose it: the affix is inserted word-internally ~60% of the time and
prefixed ~40%, and which one is a fact about each verb — so licensed forms are
a superset, and only attestation picks the winner.

### 2.5 Labels

| Label | Meaning | Use |
|---|---|---|
| `attested` | in the book, passes alphabet+shape | gold, always |
| `licensed` | rejected by no admitted rule, not attested | training only in the licensed arm; never gold |
| `rejected` | breaks an admitted rule or the alphabet | excluded; logged as the model's error signal |

**Promotion rule:** a `licensed` row becomes `attested` only with a speaker
confirmation or a second source. The loop amplifies its own errors the moment
this is skipped — this is the failure mode the Stoney arm is designed to
measure.

---

## 3. The loop

```
        lexicon graph (dakota_entries)
              │
   ┌──────────▼──────────┐        ┌─────────────────────┐
   │ generator           │        │ executable checker  │
   │ (clue templates +   │───────▶│ alphabet, shape,    │
   │  model rephrasings, │        │ attested, rules     │
   │  form proposals)    │        └─────────┬───────────┘
   └──────────────────────┘                  │ 3-label triage
                                            ▼
                              attested → SFT gold (pass 2)
                              licensed → licensed arm only
                              rejected → generator feedback
                                            │
                                ┌───────────▼───────────┐
                                │ SFT pass 2 (Tinker)   │
                                │ + RL for lookup/abstain│
                                └───────────┬───────────┘
                                            ▼
                              frozen eval: rephrased-clue recall,
                              held-out families, violation rate
```

Rejected rows go back to the generator as *negative* signal (the proposal that
broke rule X is not retried verbatim), which is what makes this a loop and not
a filter.

---

## 4. Arms

| Arm | Adds to pass-2 SFT | Tests |
|---|---|---|
| A. Baseline | current mixed file, no new rows | reference |
| B. Rephrased clues | model-varied clue wording, answers anchored (Cree contract) | is knowledge tied to prompt wording? |
| C. Rule-licensed forms | checker-approved unattested forms, labeled as such | does licensed surface area help or pollute? |
| D. Unverified (Stoney-style) | model-written answers, no checker | what is the verifier worth? |

C vs A is the main read; D vs C prices the checker. All arms share the same
base model, base SFT recipe (LoRA rank 32 on Tinker), and frozen eval.

## 5. Measures (frozen before any arm runs)

1. **Recall under unseen phrasings** — held-out clue templates written after
   training data is fixed; answers scored against the two-witness accepted set.
2. **Held-out families** — the existing `heldout_rule` split, unchanged.
3. **Rule-violation rate in free output** — sample free generation, run the
   checker, report % rejected. Arm D is predicted to be highest.
4. **Neither-attested-nor-licensed rate** — forms the checker can't place.
   This is the error-amplification detector.

## 6. Compute shape

- M1 (checker + triage): CPU only. No training. **Done 2026-11-13** — triage
  parquet published; no GPU was used.
- M2 (inventory compile): CPU + page-image reading.
- M3–M4 (generation + arms B/C/D + baseline): Tinker LoRA, ~4 passes
  comparable to the $11.23/pass Inkling run; generation rollouts via the
  existing Tinker sampling path.
- Eval: the frozen sets, temperature 0, 1,500 tasks — same harness as the
  2026-11-12 run.

## 7. Limits

- A licensed form is a hypothesis about 1890 orthography rules, not about
  Dakota as spoken. It never enters gold without a speaker or a second source.
- The book is one witness with missionary bias; the checker verifies
  well-formedness *per Riggs*, nothing more.
- One seed per arm to start; the D arm exists partly because a single noisy
  run of the Stoney pipeline already misled once.
