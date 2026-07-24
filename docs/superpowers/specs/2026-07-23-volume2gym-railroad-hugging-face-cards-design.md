# Volume2Gym Railroad 1959 — Hugging Face Release Design

Date: 2026-07-23

Status: Locked for review

Visual direction: Living Blueprint × Archivist's Light Table

## 1. Outcome

Publish a paired Hugging Face release that turns the 1959 *Consolidated Code of Operating Rules* into an auditable case study for Volume2Gym:

- The dataset card explains the archive, transformation pipeline, environment interface, provenance, and broader research idea.
- The model card explains the Qwen3-4B adapter experiment, reports the actual Tinker run, and links every headline result to raw artifacts.
- Both cards use original source pages, archival collages, a generated cinematic hero, purpose-built diagrams, and regenerated charts while staying inside Hugging Face's supported Markdown rendering.

The release should feel like a living engineering blueprint laid across an archivist's light table: dramatic at first glance, source-literate on inspection, and reproducible underneath.

## 2. Existing Work and Reuse Boundary

No repository named exactly `HarleyCoops/railroad` was found. The existing project is:

- GitHub: `HarleyCoops/Qwen3-RailroadEngineer1959-RL`
- Hugging Face dataset: `HarleyCooper/volume2gym-railroad-1959`
- Hugging Face model: `HarleyCooper/Qwen3-4B-RailRoadEngineer1959`

The GitHub repository is the upstream project record and a source of extraction code, environment code, documentation, and historical assets. It will be linked from both cards and reused selectively, but it will not be edited in this release unless separately requested.

Its prose is not safe to copy wholesale:

- The root README is dominated by Dakota 1890 material rather than the railroad experiment.
- Some pipeline prose describes speculative or unfinished Gemini/SAM 3D work that is not part of the published training run.
- The checked-in legacy environment uses an Anthropic LLM judge with separately weighted safety, procedure, and terminology channels, while the published Tinker run used a deterministic proxy scorer.

The Hugging Face cards must distinguish these systems explicitly and must not blend their claims.

## 3. Audience and Communication Goals

The primary audience is reinforcement-learning researchers, dataset builders, digital humanists, low-resource-language technologists, and technically curious readers.

After reading the cards, a reader should understand:

1. What the 1959 railroad rulebook is and why it is structurally useful.
2. How a fixed technical volume becomes an RL environment.
3. What the agent observes, produces, and receives as reward.
4. What was actually trained and measured.
5. Why this pattern matters directly for document-grounded RL.
6. Why it may matter indirectly for low-resource-language work without implying that this railroad adapter itself improves any language.
7. Where to inspect source pages, transformed data, run configuration, and raw metrics.

## 4. Scientific Story

### 4.1 Direct claim: document-grounded reinforcement learning

Volume2Gym is the idea that structured text can be compiled into an interactive learning environment. A volume supplies:

- a bounded source of truth;
- addressable rules, definitions, and procedures;
- task-generating structure;
- verifiable response targets; and
- a reward surface that can be audited against the source.

For this experiment, the transformation is presented as:

`117 scanned pages → 536 extracted rules → 2,708 scenario tasks → grouped policy updates`

These counts must be verified against the release artifacts before publication and amended if the audit differs.

### 4.2 Indirect claim: a pattern relevant to low-resource languages

The card will frame low-resource-language relevance as a transferable systems pattern, not as a measured outcome:

- Community grammars, dictionaries, pedagogical texts, oral-history transcripts, and annotated corpora are often structured but too small for conventional pretraining.
- Volume2Gym offers a way to turn such bounded materials into tasks, constraints, feedback, and evaluation surfaces.
- This can make scarce, community-governed knowledge operational in training without pretending that data volume has ceased to matter.

The cards will state plainly that the railroad run is an English historical-technical demonstration. It does not evaluate a low-resource language, establish cross-lingual transfer, or authorize reuse of culturally governed materials. Any future language application would require community authority, appropriate licensing, native-speaker review, and culturally specific evaluation.

## 5. Paired Card Architecture

### 5.1 Shared opening

Both cards begin with the same visual family:

- a card-column-wide hero image;
- a compact one-sentence thesis;
- reciprocal links between dataset, model, and GitHub project;
- a small status strip identifying source year, pages, rules, tasks, base model, and run type.

The hero combines a clearly artistic railroad scene with authentic scans and diagram fragments. Generated imagery will be labeled as an interpretive illustration; original pages will be captioned as archival source material.

### 5.2 Dataset card

Proposed sequence:

1. **Hero — The Rulebook Becomes a World**
2. **The Volume** — title page, ownership page, railroad roster, and publication context
3. **Inside the Archive** — contact sheet and selected full-page reproductions
4. **One Volume, One Environment** — visual pipeline from scan to rule to scenario to reward
5. **Dataset Anatomy** — configs, splits, row schema, source references, answer schema, reward ledger
6. **Environment Contract** — observation, action, verifier, reward, termination, and evaluation
7. **Why Railroad Rules** — boundedness, procedural structure, terminology, safety hierarchy
8. **Volume2Gym Beyond Railroads** — direct RL significance and careful low-resource-language bridge
9. **Provenance and Rights** — source identity, custody, user rights assertion, transformations, image manifest
10. **Limitations and Intended Use** — historical research only; not current operating guidance
11. **Reproducibility and Links** — model, code, raw data, checksums, citation

### 5.3 Model card

Proposed sequence:

1. **Hero — Learning the Rulebook**
2. **Experiment at a Glance** — Qwen3-4B adapter, Tinker GRPO, task count, steps, batch/group sizes
3. **The Learning Loop** — source-grounded scenario, sampled answers, deterministic score, grouped update
4. **Actual Run Results** — training trajectory, evaluation gates, answer length, parse compliance
5. **What Improved** — exact start/end/peak results with denominators and definitions
6. **What the Metrics Mean** — scorer mechanics, reward interpretation, and caveats
7. **Run Audit** — deterministic Tinker scorer versus legacy Anthropic-judge environment
8. **Examples from the Ledger** — selected source-linked tasks and outputs, after privacy/safety review
9. **Reproduce the Run** — config, checkpoints, metrics, adapter usage
10. **Limitations and Safety** — historical model, no operational railroad use, no present-day compliance claim
11. **Citation, License, and Links**

## 6. Visual System

### 6.1 Art direction

The visual language combines:

- blueprint navy and cyan for mechanisms, schemas, and learning loops;
- warm paper, amber light, graphite, and red editorial marks for archival evidence;
- full-bleed-looking compositions constrained to the Hugging Face card column;
- restrained industrial typography baked into pre-rendered graphics;
- dense source detail at high resolution with generous breathing room in Markdown.

The work should feel cinematic, not nostalgic kitsch. Scans remain legible and visibly material: folds, halftones, line weight, stamps, margins, and typographic hierarchy are treated as evidence.

### 6.2 Planned assets

Shared or coordinated assets:

1. `hero-railroad-volume2gym.webp` — generated cinematic scene composited with authentic scan fragments; approximately 1600×900.
2. `archive-light-table.webp` — collage of title, ownership, roster, rule, sign, and diagram pages.
3. `volume-to-gym-blueprint.webp` — scan → extraction → structured task → reward → update.
4. `archive-contact-sheet.webp` — selected source pages with page identifiers.

Dataset-specific:

5. `dataset-anatomy.webp` — record schema and relationships.
6. `reward-environment-contract.webp` — observation/action/verifier/reward diagram.

Model-specific:

7. `training-reward-trajectory.webp` — raw training reward with start, peak, and final annotated separately.
8. `eval-gates.webp` — evaluation reward at steps 0, 20, 40, and 60 with episode counts.
9. `efficiency-and-format.webp` — tokens per turn and parse compliance.
10. `run-facts.webp` — compact configuration plate.

Original source pages used in a collage will also be published individually at readable resolution, with descriptive alt text and a manifest mapping filenames to scan page numbers.

### 6.3 Image-generation boundary

The cinematic hero may use image generation for atmosphere and composition, but generated pixels must not be presented as a historical photograph. Documentary claims, typography, numbers, diagrams, and source pages will be added from verified assets in a deterministic layout. Charts will never be generated by an image model.

## 7. Hugging Face-Native Implementation Constraints

The cards will use only features supported by Hugging Face repository cards:

- `README.md` with valid YAML metadata;
- standard Markdown headings, paragraphs, lists, blockquotes, tables, links, and images;
- repository-hosted static PNG, JPEG, or WebP assets;
- optional supported HTML image markup only where necessary for width or light/dark variants;
- KaTeX only if a formula materially improves explanation.

The cards will not depend on custom CSS, JavaScript, iframes, animation, CSS grid, or viewport-wide positioning. “Full width” means the available card-content column. Complex layouts will be precomposed into single accessible images rather than assembled with unsupported browser styling.

All images will have meaningful alt text. Text essential to comprehension will also appear as real Markdown rather than only inside an image.

## 8. Run Data and Truthfulness Rules

The model card will be generated from the raw local Tinker artifacts:

- `metrics.jsonl`
- `config.json`
- `checkpoints.jsonl`
- `railroad_reward_ledger_tinker.csv`

Artifacts will be scanned for credentials, private URLs, personal data, and unstable local paths before upload. Full rollout HTML logs are excluded by default because they are large and require a separate content review.

The current audit found:

| Measurement | Start | End / final gate | Peak |
|---|---:|---:|---:|
| Training reward | 0.2465 at step 0 | 0.3338 at step 76 | 0.3983 at step 57 |
| Evaluation reward | 0.2493 at step 0 | 0.3753 at step 60 | 0.3753 at step 60 |
| Training tokens/turn | 253.64 | 84.56 | n/a |
| Evaluation tokens/turn | 255.86 | 90.08 | n/a |
| Training parse compliance | 5.86% | 100% | 100% |
| Evaluation parse compliance | 0.37% | 100% | 100% |

Evaluation gates contain 270 episodes each. The card may summarize evaluation reward as approximately +50.5%, evaluation tokens per turn as approximately −64.8%, and parse compliance as 0.37% → 100%, provided those figures are regenerated and tested from the uploaded raw file.

Training peak and training final must remain distinct. The hero or headline must not substitute the peak for the final value.

The existing rendered charts will not be reused because they contain Dakota labels and blank component panels. All result graphics will be regenerated from raw metrics.

The Tinker ledger currently reports identical values for safety, procedure, and terminology proxy columns. The card will not visualize these as three independently measured improvements. It will explain that the deterministic run mapped the same proxy signal into those ledger fields, while the legacy judge-based environment defined genuinely separate weighted channels.

## 9. Metadata Design

### 9.1 Dataset

The dataset card will replace the incorrect video modality metadata with verified dataset metadata. Expected fields include:

- English language;
- custom/other license metadata with a human-readable rights note and link where supported;
- task categories and tags appropriate to text generation, question answering, reinforcement learning, synthetic tasks, historical documents, and structured text;
- accurate configs/splits/features only if they match loadable published data;
- source, model, and code links;
- a thumbnail derived from the shared hero.

If the current repository is not loadable through Dataset Viewer, the publication step will either normalize its data layout or describe it as an artifact repository without inventing viewer statistics. Normalizing the dataset is permitted only if it preserves existing records and identifiers.

### 9.2 Model

Expected fields include:

- `language: en`
- appropriate license metadata
- `library_name: peft`, if verified against the adapter
- `base_model: Qwen/Qwen3-4B-Instruct-2507`
- adapter/base-model relationship metadata where supported
- `pipeline_tag: text-generation`
- linked dataset
- tags for reinforcement learning, GRPO, LoRA, historical documents, and Volume2Gym
- a repository-hosted thumbnail
- structured model-index/custom results where valid

Custom metrics will be defined clearly. The release will not fabricate a registered Hugging Face benchmark or use the new evaluation-results schema unless the custom task and dataset satisfy its registration requirements.

## 10. Provenance, Rights, and Safety

The dataset card will identify the source as the 1959 revised edition of *The Consolidated Code of Operating Rules*, effective December 1, 1959, and will include a source manifest for reproduced pages.

The rights statement will record that the repository owner asserts ownership or authorization for the reproduced source images. It will avoid an unsupported public-domain claim. Any license label selected in metadata must match the actual grant accompanying the repository.

Both cards will state:

- The source is historical and may not reflect current rules, terminology, law, technology, or safety practice.
- The dataset and adapter are for research, education, and archival experimentation.
- They must not be used to control trains, train operating personnel for present-day service, or make operational safety decisions.

## 11. Publication Layout

The implementation will use a local staging area with separate dataset and model repository directories. Changes will be committed and pushed to each Hugging Face repository independently so their histories remain legible.

Suggested repository layout:

```text
README.md
assets/
  hero/
  archive/
  diagrams/
  charts/
  source-pages/
run/
  config.json
  metrics.jsonl
  checkpoints.jsonl
analysis/
  run-summary.json
  railroad-reward-ledger-tinker.csv
  source-manifest.csv
```

The dataset repository may additionally contain normalized data files and a dataset-loading configuration. The model repository will not duplicate the full dataset; it will link to it.

## 12. Verification

Before either push:

1. Parse the YAML front matter using Hugging Face card tooling or an equivalent strict YAML parser.
2. Recompute every reported metric from the staged `metrics.jsonl`.
3. Compare the regenerated summary against the card tables and chart annotations.
4. Check that the source-page manifest resolves to existing files.
5. Check all relative links and image paths.
6. Confirm no image contains stale Dakota branding or blank data panels.
7. Scan staged artifacts for secrets, access tokens, local user paths, and personal data.
8. Validate image dimensions, file sizes, legibility, and alt text.
9. Confirm the adapter metadata matches `adapter_config.json`.
10. Review the complete staged diff for unintended changes.

After each push:

1. Confirm the remote commit and authenticated owner.
2. Open the rendered card and verify YAML, hero, tables, charts, anchors, and cross-links.
3. Confirm the Dataset Viewer status or state its limitation accurately.
4. Record the final Hugging Face commit identifiers in the handoff.

## 13. Scope Boundaries

Included:

- research and source audit;
- paired card copy;
- original-page selections and archival collages;
- one generated interpretive hero;
- deterministic diagrams and charts;
- safe raw run artifacts;
- metadata correction;
- optional non-destructive dataset-layout normalization;
- publication to both existing Hugging Face repositories.

Excluded:

- retraining the model;
- modifying adapter weights;
- registering a new Hugging Face benchmark;
- asserting present-day railroad validity;
- claiming demonstrated low-resource-language performance;
- rewriting or publishing changes to the GitHub repository;
- publishing unreviewed rollout HTML logs.

## 14. Acceptance Criteria

The release is complete when:

- both Hugging Face cards are live and cross-linked;
- the final visual direction is evident without unsupported web features;
- authentic source pages are readable, captioned, and mapped to a manifest;
- all charts are regenerated from uploaded raw data;
- training start, peak, and final values are not conflated;
- evaluation results state four gates and 270 episodes per gate;
- the deterministic Tinker scorer and legacy LLM judge are clearly separated;
- the low-resource-language section is ambitious but explicitly indirect;
- incorrect video metadata and stale Dakota labels are absent;
- the source, task, reward, run, limitations, rights, and safety story is understandable without relying on images alone;
- remote rendering and repository commits are verified.
