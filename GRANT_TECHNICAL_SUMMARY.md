# Grant Technical Summary

## What Dakota1890 Does

Dakota1890 turns a single historical learning source into a runnable language-model training ecosystem. The source in this case is Stephen Return Riggs' 1890 *Dakota Grammar, Texts, and Ethnography*. The repository ingests scans and page images, extracts structured grammatical and lexical knowledge with vision-language models, converts that knowledge into verifiable training tasks, and uses reinforcement learning to train a model that can produce Dakota-language outputs while preserving orthography and morphology.

The central technical claim is that endangered-language documentation can be converted into executable feedback. Instead of treating a grammar book as static reference material, Dakota1890 treats each extracted rule as a constraint function. That produces a reward environment rather than only a corpus. The result is a pipeline that can move from source pages to a verifier-backed language model even when parallel corpora and native-speaker-labeled training sets are scarce.

## Why This Matters

Low-resource language revitalization is often blocked by three engineering problems:

1. source material exists only in historical scans or books
2. modern supervised datasets are too small or too expensive to create
3. evaluation is difficult because native-speaker time is precious and should not be wasted on routine data formatting work

Dakota1890 addresses those constraints by combining three ideas:

- VLM extraction from historical documents
- synthetic data generation for a comparison SFT baseline
- reinforcement learning from grammar-derived verifiers for the main training path

In the current repository state, the Dakota path contains:

- `1,497` organized grammar rules
- `10,576` RL training tasks
- an importable Dakota grammar environment with a reward ledger
- a preserved SFT baseline with `980` training and `245` validation chat examples
- a published adapter surface for Dakota inference

## Technical Architecture

The maintained path now looks like this:

```text
Riggs source pages
  -> grammar extraction
  -> rule organization
  -> RL task generation
  -> packaged Dakota verifier environment
  -> RL training (local checks or remote Tinker / PrimeIntellect path)
  -> published model / inference
```

Alongside that, the repo preserves a secondary educational branch:

```text
dictionary extraction
  -> Gemini synthetic QA generation
  -> OpenAI chat-format conversion
  -> OpenAI fine-tune readiness baseline
```

This secondary branch is useful for comparison because it shows what the project can learn from straightforward supervision before the grammar-gym RL stage.

## What Is Novel Here

The novelty is not only “OCR a dictionary and fine-tune a model.” The more important contribution is the conversion of grammar into machine-checkable rewards. Dakota orthography, affix behavior, and response structure are scored by a rubric instead of being left entirely to informal judgment. That makes reinforcement learning viable in a domain usually considered too qualitative for verifiable feedback.

This matters for generalization. Many endangered languages do not have modern instruction datasets, but they do have grammars, dictionaries, missionary texts, readers, ethnographies, or pedagogical materials. If those materials can be extracted reliably enough, then they can seed a verifier environment. The resulting training loop is portable in principle even when the surface rules, orthography, and cultural constraints differ language by language.

## What Is Dakota-Specific vs Reusable

Reusable components already present in the repo:

- source-page ingestion and image conversion
- page-level structured extraction pattern
- rule-to-task transformation pattern
- packaged environment architecture
- synthetic-QA-to-chat-format baseline path

Dakota-specific components that would need parameterization:

- extraction prompts and schema details
- special-character preservation rules
- affix and morphology heuristics
- reward-weight tuning
- community constraints around restricted content and acceptable outputs

The next generalization step should therefore be a language configuration layer rather than a rewrite. A future version should externalize orthography rules, source-page ranges, extraction prompts, reward weights, and community-governance settings into a per-language config.

## Limits and Responsible Extension

This repository does not solve pragmatics, ceremonial register, or the difference between 1890 Dakota and living Dakota as spoken today. Those are not bugs to hide; they are precisely why the next phase must be community-in-the-loop. The grant narrative should state this directly.

Operationally, community-in-the-loop should mean:

- source review before extraction at scale
- validation of orthography and meaning after extraction
- explicit handling of temporal drift between historical and living usage
- protected pathways for culturally sensitive or restricted material
- post-training review and correction before broader deployment

This is also why the first BC descendant-language target should remain unnamed in the current repo-facing summary unless that community relationship is already established and consented to. The engineering can be described now; the community-specific claim should be made only when the partnership is concrete.

## Step-0 Outcome

After the repository audit and cleanup, Dakota1890 is in a better position to serve as a grant artifact because:

- the Dakota core path is clearer
- stale or misleading docs have been archived
- the test harness is now useful
- the model lineage is consistent across inference and model-card surfaces
- the step-0 docs now explain what is active, what is preserved, and what still blocks full live reproducibility

That is the right basis for a fellowship application: not an overclaimed “finished product,” but a technically serious, inspectable pipeline whose strengths, limitations, and generalization strategy are now legible.
