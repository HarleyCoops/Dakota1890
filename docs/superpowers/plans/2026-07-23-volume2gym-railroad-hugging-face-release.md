# Volume2Gym Railroad 1959 Hugging Face Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate, and publish an archival-quality dataset card and an evidence-backed model card for the Volume2Gym Railroad 1959 experiment.

**Architecture:** Reproducible Python builders transform the verified source scans, structured tasks, and raw Tinker metrics into two self-contained Hugging Face upload bundles. The dataset bundle owns the complete 117-page image archive, normalized train/test JSONL, source manifest, archival graphics, and Volume2Gym explanation; the model bundle owns the adapter card, run artifacts, regenerated charts, and scorer audit. A validator treats Markdown metadata, local assets, numerical claims, source checksums, and prohibited stale labels as release gates before the two bundles are uploaded independently.

**Tech Stack:** Python 3.12, standard library, Pillow, Matplotlib, PyYAML, pytest, Hugging Face Hub CLI/SDK, Hugging Face Dataset Viewer API, Markdown, WebP/PNG.

## Global Constraints

- Visual direction is **Living Blueprint × Archivist's Light Table**.
- Cards must render using Hugging Face-supported YAML, Markdown, repository-hosted static images, tables, links, and optional supported image HTML only.
- No custom CSS, JavaScript, iframe, animation, CSS grid, or viewport-wide positioning.
- The complete 117-page scan set may be published because the repository owner asserts ownership or authorization, but the card must not make an unsupported public-domain claim.
- The 1959 source is historical research material, not present-day railroad operating guidance.
- The railroad run is an English technical demonstration; low-resource-language relevance is an indirect systems hypothesis, not an evaluated result.
- The deterministic Tinker scorer and the legacy Anthropic-judge environment must be described separately.
- Charts must be regenerated from `metrics.jsonl`; existing Dakota-labeled or blank-panel charts are prohibited.
- Training start, peak, and final values must remain distinct.
- Evaluation uses four gates at steps 0, 20, 40, and 60 with 270 episodes at each gate.
- Essential explanations and numbers must remain available as Markdown text rather than only inside images.
- Model weights and adapter files already on the Hub must not be replaced or deleted.
- Full rollout HTML logs are excluded from publication.

---

## File Structure

Create or modify these project files:

```text
scripts/railroad_release/
  __init__.py                         Package marker.
  metrics.py                          Parse raw Tinker logs and compute summaries.
  dataset.py                          Reproduce the seed-42 train/test split and JSONL.
  visuals.py                          Build collages, diagrams, charts, and final hero composite.
  validate.py                         Validate bundles, metadata, links, assets, metrics, and bans.
tests/railroad_release/
  test_metrics.py                     Numerical summary and scorer-channel tests.
  test_dataset.py                     Record conversion and deterministic split tests.
  test_visuals.py                     Asset dimensions, manifests, and chart-output tests.
  test_validate.py                    Metadata, link, artifact, and prohibited-label tests.
hf_model_card_work/railroad-1959-release/
  dataset/
    README.md
    RIGHTS.md
    data/train.jsonl
    data/test.jsonl
    source/railroad_rules_complete.json
    assets/hero/hero-railroad-volume2gym.webp
    assets/archive/archive-light-table.webp
    assets/archive/archive-contact-sheet.webp
    assets/archive/source-manifest.csv
    assets/archive/source-pages/page_001.png ... page_117.png
    assets/diagrams/volume-to-gym-blueprint.webp
    assets/diagrams/dataset-anatomy.webp
    assets/diagrams/reward-environment-contract.webp
  model/
    README.md
    RIGHTS.md
    assets/hero/hero-railroad-volume2gym.webp
    assets/archive/archive-light-table.webp
    assets/diagrams/learning-loop.webp
    assets/charts/training-reward-trajectory.webp
    assets/charts/eval-gates.webp
    assets/charts/efficiency-and-format.webp
    assets/charts/run-facts.webp
    run/config.json
    run/metrics.jsonl
    run/checkpoints.jsonl
    analysis/run-summary.json
    analysis/scorer-reference.py
    analysis/railroad-reward-ledger-tinker.csv
```

Source inputs:

```text
C:/Users/chris/Daily/Projects/RailroadEngineer1959/data/processed_images/
C:/Users/chris/Daily/Projects/RailroadEngineer1959/data/railroad_extracted/
C:/Users/chris/Daily/Projects/RailroadEngineer1959/outputs/tinker_railroad_run/
C:/Users/chris/Daily/Projects/RailroadEngineer1959/wandb_analysis/
C:/Users/chris/Daily/Projects/RailroadEngineer1959/environments/railroad_1959/
```

## Task 1: Build the Run-Metrics Pipeline

**Files:**
- Create: `scripts/railroad_release/__init__.py`
- Create: `scripts/railroad_release/metrics.py`
- Create: `tests/railroad_release/test_metrics.py`

**Interfaces:**
- Consumes: Tinker `metrics.jsonl`, `config.json`, and `checkpoints.jsonl`.
- Produces: `summarize_metrics(rows: list[dict]) -> dict`, `scrub_config(config: dict) -> dict`, `write_run_bundle(source_dir: Path, ledger_path: Path, scorer_path: Path, model_dir: Path) -> dict`.
- Summary keys: `training`, `evaluation`, `configuration`, `scorer_audit`, and `source_sha256`.

- [ ] **Step 1: Write the failing metric-summary tests**

```python
from scripts.railroad_release.metrics import scrub_config, summarize_metrics


def test_summarize_metrics_keeps_training_peak_and_final_separate():
    rows = [
        {
            "step": 0,
            "env/all/reward/total": 0.20,
            "env/all/ac_tokens_per_turn": 200.0,
            "env/all/ledger/parse_success": 0.25,
            "env/all/total_episodes": 256,
            "test/env/railroad_1959/reward/total": 0.10,
            "test/env/railroad_1959/ac_tokens_per_turn": 220.0,
            "test/env/railroad_1959/ledger/parse_success": 0.0,
            "test/env/railroad_1959/total_episodes": 10,
        },
        {
            "step": 1,
            "env/all/reward/total": 0.40,
            "env/all/ac_tokens_per_turn": 100.0,
            "env/all/ledger/parse_success": 1.0,
            "env/all/total_episodes": 256,
        },
        {
            "step": 2,
            "env/all/reward/total": 0.30,
            "env/all/ac_tokens_per_turn": 80.0,
            "env/all/ledger/parse_success": 1.0,
            "env/all/total_episodes": 48,
            "test/env/railroad_1959/reward/total": 0.25,
            "test/env/railroad_1959/ac_tokens_per_turn": 90.0,
            "test/env/railroad_1959/ledger/parse_success": 1.0,
            "test/env/railroad_1959/total_episodes": 10,
        },
    ]
    summary = summarize_metrics(rows)
    assert summary["training"]["start"]["reward"] == 0.20
    assert summary["training"]["peak"]["reward"] == 0.40
    assert summary["training"]["peak"]["step"] == 1
    assert summary["training"]["final"]["reward"] == 0.30
    assert summary["evaluation"]["gates"] == [0, 2]
    assert summary["evaluation"]["episodes_per_gate"] == [10, 10]


def test_scrub_config_replaces_local_paths_but_preserves_hyperparameters():
    scrubbed = scrub_config({
        "learning_rate": 5e-5,
        "dataset_builder": {"dataset_path": "C:/Users/chris/private/tasks.json"},
        "log_path": "C:/Users/chris/private/run",
    })
    assert scrubbed["learning_rate"] == 5e-5
    assert scrubbed["dataset_builder"]["dataset_path"] == "data/train.jsonl"
    assert scrubbed["log_path"] == "run"
    assert "C:/Users" not in str(scrubbed)
```

- [ ] **Step 2: Run the focused tests and confirm the import fails**

Run:

```powershell
& .\.venv_win\Scripts\python.exe -m pytest tests\railroad_release\test_metrics.py -q
```

Expected: collection fails because `scripts.railroad_release.metrics` does not exist.

- [ ] **Step 3: Implement strict field extraction and summary generation**

Use these aggregate keys only:

```python
TRAIN = {
    "reward": "env/all/reward/total",
    "tokens": "env/all/ac_tokens_per_turn",
    "parse": "env/all/ledger/parse_success",
    "episodes": "env/all/total_episodes",
}
EVAL = {
    "reward": "test/env/railroad_1959/reward/total",
    "tokens": "test/env/railroad_1959/ac_tokens_per_turn",
    "parse": "test/env/railroad_1959/ledger/parse_success",
    "episodes": "test/env/railroad_1959/total_episodes",
}


def _series(rows: list[dict], keys: dict[str, str]) -> list[dict]:
    return [
        {
            "step": int(row["step"]),
            "reward": float(row[keys["reward"]]),
            "tokens_per_turn": float(row[keys["tokens"]]),
            "parse_success": float(row[keys["parse"]]),
            "episodes": int(row[keys["episodes"]]),
        }
        for row in rows
        if keys["reward"] in row
    ]
```

`summarize_metrics` must report:

- count, start, final, peak, and arithmetic mean for training;
- episode-weighted training mean and total train/evaluation rollout counts;
- gates, per-gate records, start, final, and peak for evaluation;
- signed absolute and percentage deltas for reward and tokens;
- percentage-point delta for parse success;
- `channels_independent: false` when safety, procedure, and terminology are equal across aggregate rows;
- SHA-256 for each copied raw artifact.

- [ ] **Step 4: Implement artifact scrubbing and writing**

`write_run_bundle` must:

1. copy `metrics.jsonl` unchanged;
2. transform `checkpoints.jsonl` into a public manifest retaining `name`, `batch`, and `locator_scheme: "tinker"` while replacing each state/sampler locator with its SHA-256; do not publish the internal Tinker run UUID;
3. replace local paths in `config.json` with repository-relative paths;
4. copy `railroad_reward_ledger_tinker.csv`;
5. write deterministic, sorted, indented `analysis/run-summary.json`;
6. write `analysis/scorer-reference.py`, a dependency-free public reference containing the exact lowercase/whitespace normalization, whitespace-token multiset F1, `safety = max(exact, f1)`, `procedure = f1`, `terminology = f1`, and `0.5/0.3/0.2` scalar calculation; record the SHA-256 of the inspected source environment in its header;
7. reject strings matching `hf_`, `Bearer `, `api_key`, `C:\Users\`, or `/home/`.

- [ ] **Step 5: Run tests and the real-data build**

Run:

```powershell
& .\.venv_win\Scripts\python.exe -m pytest tests\railroad_release\test_metrics.py -q
& .\.venv_win\Scripts\python.exe -m scripts.railroad_release.metrics `
  --source C:\Users\chris\Daily\Projects\RailroadEngineer1959\outputs\tinker_railroad_run `
  --ledger C:\Users\chris\Daily\Projects\RailroadEngineer1959\wandb_analysis\railroad_reward_ledger_tinker.csv `
  --scorer C:\Users\chris\Daily\Projects\RailroadEngineer1959\environments\railroad_1959\railroad_1959\environment.py `
  --output hf_model_card_work\railroad-1959-release\model
```

Expected: tests pass and the command prints `77 training steps; 4 evaluation gates; 270 episodes per gate`.

- [ ] **Step 6: Commit**

```powershell
git add scripts/railroad_release tests/railroad_release hf_model_card_work/railroad-1959-release/model/run hf_model_card_work/railroad-1959-release/model/analysis
git commit -m "feat: build railroad run evidence bundle"
```

## Task 2: Normalize the Dataset for Dataset Viewer

**Files:**
- Create: `scripts/railroad_release/dataset.py`
- Create: `tests/railroad_release/test_dataset.py`
- Create: generated `hf_model_card_work/railroad-1959-release/dataset/data/train.jsonl`
- Create: generated `hf_model_card_work/railroad-1959-release/dataset/data/test.jsonl`
- Create: generated `hf_model_card_work/railroad-1959-release/dataset/source/railroad_rules_complete.json`

**Interfaces:**
- Consumes: `safety_tasks_complete.json`, `railroad_rules_complete.json`.
- Produces: `normalize_record(row: dict) -> dict`, `split_records(rows: list[dict], seed: int = 42, eval_fraction: float = 0.1) -> tuple[list[dict], list[dict]]`, `write_dataset_bundle(...) -> dict`.

- [ ] **Step 1: Write deterministic conversion and split tests**

```python
from scripts.railroad_release.dataset import normalize_record, split_records


def test_normalize_record_preserves_source_fields_and_adds_prompt():
    record = normalize_record({
        "task_id": "101-001",
        "description": "A washout is reported ahead.",
        "applicable_rules": ["101"],
        "expected_outcome": "Reduce speed and be prepared to stop.",
    })
    assert record["task_id"] == "101-001"
    assert record["scenario"] == "A washout is reported ahead."
    assert record["applicable_rules"] == ["101"]
    assert record["reference_response"].startswith("Reduce speed")
    assert record["prompt"].startswith("Task ID: 101-001\nScenario:")


def test_split_matches_tinker_seed_and_fraction():
    rows = [{"task_id": str(i)} for i in range(20)]
    train_a, test_a = split_records(rows, seed=42, eval_fraction=0.1)
    train_b, test_b = split_records(rows, seed=42, eval_fraction=0.1)
    assert len(train_a) == 18
    assert len(test_a) == 2
    assert test_a == test_b
    assert train_a == train_b
    assert {r["task_id"] for r in train_a}.isdisjoint({r["task_id"] for r in test_a})
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```powershell
& .\.venv_win\Scripts\python.exe -m pytest tests\railroad_release\test_dataset.py -q
```

Expected: collection fails because `scripts.railroad_release.dataset` does not exist.

- [ ] **Step 3: Implement the exact Tinker split**

The split algorithm must be:

```python
shuffled = [dict(row) for row in rows]
random.Random(seed).shuffle(shuffled)
eval_size = max(1, int(len(shuffled) * eval_fraction))
test_rows = shuffled[:eval_size]
train_rows = shuffled[eval_size:]
```

Write UTF-8 JSONL with sorted keys and no ASCII escaping. For the real 2,708 records, assert 2,438 train rows, 270 test rows, unique task IDs, and no overlap.

- [ ] **Step 4: Copy the verified complete rules file and write checksums**

Copy `railroad_rules_complete.json` without semantic changes. Return row counts and SHA-256 values from `write_dataset_bundle`.

- [ ] **Step 5: Run tests and build the real dataset bundle**

Run:

```powershell
& .\.venv_win\Scripts\python.exe -m pytest tests\railroad_release\test_dataset.py -q
& .\.venv_win\Scripts\python.exe -m scripts.railroad_release.dataset `
  --tasks C:\Users\chris\Daily\Projects\RailroadEngineer1959\data\railroad_extracted\safety_tasks_complete.json `
  --rules C:\Users\chris\Daily\Projects\RailroadEngineer1959\data\railroad_extracted\railroad_rules_complete.json `
  --output hf_model_card_work\railroad-1959-release\dataset
```

Expected: `2438 train / 270 test / 2708 total`.

- [ ] **Step 6: Commit**

```powershell
git add scripts/railroad_release/dataset.py tests/railroad_release/test_dataset.py hf_model_card_work/railroad-1959-release/dataset/data hf_model_card_work/railroad-1959-release/dataset/source
git commit -m "feat: normalize railroad tasks for dataset viewer"
```

## Task 3: Build the Archival Asset Pipeline

**Files:**
- Create: `scripts/railroad_release/visuals.py`
- Create: `tests/railroad_release/test_visuals.py`
- Create: generated dataset `assets/archive/`, `assets/diagrams/`
- Create: generated model `assets/archive/`, `assets/diagrams/`, `assets/charts/`

**Interfaces:**
- Consumes: 117 PNG scans, model `analysis/run-summary.json`, and raw `metrics.jsonl`.
- Produces: `copy_source_archive`, `build_archive_collage`, `build_contact_sheet`, `build_blueprint`, `build_dataset_anatomy`, `build_environment_contract`, `build_run_charts`.

- [ ] **Step 1: Write image and manifest tests**

```python
from pathlib import Path
from PIL import Image
from scripts.railroad_release.visuals import build_contact_sheet, source_manifest


def test_source_manifest_sorts_pages_and_hashes_content(tmp_path: Path):
    for number in (2, 1):
        Image.new("RGB", (20, 30), "white").save(tmp_path / f"page_{number:03d}.png")
    rows = source_manifest(tmp_path)
    assert [row["page"] for row in rows] == [1, 2]
    assert all(len(row["sha256"]) == 64 for row in rows)
    assert all(row["width"] == 20 and row["height"] == 30 for row in rows)


def test_contact_sheet_is_card_width(tmp_path: Path):
    inputs = []
    for number in range(1, 5):
        path = tmp_path / f"page_{number:03d}.png"
        Image.new("RGB", (100, 140), (245, 238, 220)).save(path)
        inputs.append(path)
    output = tmp_path / "sheet.webp"
    build_contact_sheet(inputs, output, width=1600)
    with Image.open(output) as image:
        assert image.width == 1600
        assert image.height >= 900
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```powershell
& .\.venv_win\Scripts\python.exe -m pytest tests\railroad_release\test_visuals.py -q
```

Expected: collection fails because `scripts.railroad_release.visuals` does not exist.

- [ ] **Step 3: Implement the complete source archive**

`copy_source_archive` must copy `page_001.png` through `page_117.png`, fail on gaps, and write:

```csv
page,filename,width,height,bytes,sha256,evidence_class,source_title,source_date
```

Every row uses `confirmed` and `Consolidated Code of Operating Rules—Revised 1959`, with date `1959-12-01`.

- [ ] **Step 4: Implement the archival collages**

Use verified selections from the visual audit, with these mandatory anchors:

- page 001: cover;
- page 002: ownership/title/effective date;
- page 003: railroad roster and General Notice;
- page 012: hand, flag, and lantern signal figures;
- page 013: engine-whistle dot/dash notation;
- page 033: train-order semaphore/lamp chart;
- page 041: abbreviations and train-order forms;
- page 052: bracket/cantilever/bridge signal-placement diagrams;
- page 053: Milwaukee Road block/interlocking aspect grid;
- page 056: Milwaukee Road Rule 240-W reduce/resume speed-sign diagrams;
- page 057: Great Northern signal variants;
- page 066: Soo Line affiliate speed-zone signs;
- page 073: Northern Pacific speed signs;
- page 074: Union Pacific signal chart;
- page 076: Union Pacific indicators and track-occupancy pictograms;
- page 087: dense Railroad Radio Rules spread.

Generate:

- `archive-light-table.webp`: 1600×1050, warm paper, navy field, cyan rule lines, red editorial ticks, six readable page windows;
- `archive-contact-sheet.webp`: 1600 px wide, all 117 pages in numerical order with page numbers;
- individual full-resolution pages under `source-pages/`.

All source scans are expected to be 1140×967 RGB images. Treat `media/preview.jpg` as a derivative crop of page 056 and do not publish it as an independent source page.

- [ ] **Step 5: Implement deterministic diagrams**

Generate with Pillow primitives and real text:

- `volume-to-gym-blueprint.webp`: “117 PAGES → 536 RULES → 2,708 TASKS → RL ENVIRONMENT”;
- `dataset-anatomy.webp`: scan page, rule ID, scenario, reference response, reward record;
- `reward-environment-contract.webp`: observation → action → deterministic token-F1 verifier → scalar reward → update;
- `learning-loop.webp`: scenario → 8 sampled responses → grouped reward → LoRA update.

Use palette: navy `#071722`, blueprint `#0E3348`, cyan `#58D5E8`, paper `#E8D8B8`, amber `#D99A3D`, red `#C7533B`, graphite `#202426`.

- [ ] **Step 6: Implement charts from aggregate fields**

Generate at 1600×900:

- training reward with start, peak, and final separately labeled;
- evaluation reward at 0/20/40/60 with `n=270 per gate`;
- training/evaluation tokens per turn and parse success;
- run-facts plate containing base model, 77 steps, LR `5e-5`, batch `32`, group `8`, LoRA rank `32`, max tokens `256`, temperature `0.9`.

No chart may contain “Dakota”, blank panels, or three apparently independent safety/procedure/terminology curves.

- [ ] **Step 7: Run tests and generate every deterministic asset**

Run:

```powershell
& .\.venv_win\Scripts\python.exe -m pytest tests\railroad_release\test_visuals.py -q
& .\.venv_win\Scripts\python.exe -m scripts.railroad_release.visuals `
  --scans C:\Users\chris\Daily\Projects\RailroadEngineer1959\data\processed_images `
  --metrics hf_model_card_work\railroad-1959-release\model\run\metrics.jsonl `
  --summary hf_model_card_work\railroad-1959-release\model\analysis\run-summary.json `
  --dataset-output hf_model_card_work\railroad-1959-release\dataset `
  --model-output hf_model_card_work\railroad-1959-release\model
```

Expected: 117 manifest rows and 10 generated WebP graphics.

- [ ] **Step 8: Inspect every major visual**

Open the hero-independent archive collage, contact sheet, all diagrams, and all charts with the image inspection tool. Reject clipped text, unreadable scans, stale labels, low contrast, misleading axes, or blank panels.

- [ ] **Step 9: Commit**

```powershell
git add scripts/railroad_release/visuals.py tests/railroad_release/test_visuals.py hf_model_card_work/railroad-1959-release/dataset/assets hf_model_card_work/railroad-1959-release/model/assets
git commit -m "feat: create railroad archival visual system"
```

## Task 4: Generate and Composite the Hero

**Files:**
- Create: generated `hf_model_card_work/railroad-1959-release/model/assets/hero/railroad-scene-generated.png`
- Create: generated dataset/model `assets/hero/hero-railroad-volume2gym.webp`
- Modify: `scripts/railroad_release/visuals.py`
- Modify: `tests/railroad_release/test_visuals.py`

**Interfaces:**
- Consumes: one built-in image-generation output, pages 001/002/003, and the Rule 240-W page.
- Produces: `compose_hero(scene_path: Path, source_pages: list[Path], output_path: Path) -> Path`.

- [ ] **Step 1: Add the hero dimension and provenance test**

```python
def test_hero_is_hf_card_ratio_and_has_sidecar(tmp_path: Path):
    hero = tmp_path / "hero.webp"
    sidecar = tmp_path / "hero-generation.json"
    # The fixture calls compose_hero with a generated background and four scan pages.
    assert hero.exists()
    with Image.open(hero) as image:
        assert image.size == (1600, 900)
    assert sidecar.exists()
    assert '"tool": "built-in image_gen"' in sidecar.read_text(encoding="utf-8")
```

- [ ] **Step 2: Generate the atmospheric base with the built-in image tool**

Use this final prompt:

```text
Use case: historical-scene
Asset type: Hugging Face repository-card hero background
Primary request: A cinematic interpretive scene inspired by a 1959 North American railroad operating-rule volume: a streamlined diesel locomotive and freight consist passing a signal mast at blue hour, seen from a low three-quarter trackside angle, with subtle drafting-table geometry and translucent blueprint linework in the atmosphere.
Style/medium: painterly editorial realism with archival photographic texture; clearly interpretive art, not a claimed documentary photograph.
Composition/framing: wide landscape, strong locomotive silhouette on the right two-thirds, quieter dark negative space on the left for later deterministic typography and authentic scan overlays.
Lighting/mood: amber headlight and signal glow against deep navy dusk; serious, precise, monumental.
Color palette: blueprint navy, cyan, warm paper amber, restrained signal red.
Materials/textures: steel, weathered ballast, faint paper grain, technical ink lines.
Constraints: no text, no logos, no watermarks, no readable signs, no modern high-speed train, no steam locomotive, no people, no invented historical emblems.
Avoid: nostalgia poster kitsch, fantasy machinery, neon cyberpunk, fake newspaper headlines, illegible generated typography.
```

Save the selected built-in output inside the model bundle and record the exact prompt/tool in `assets/hero/hero-generation.json`.

- [ ] **Step 3: Implement the deterministic composite**

Compose to 1600×900:

- generated railroad scene as background;
- dark navy-to-transparent left overlay;
- four authentic scan fragments arranged as a light-table fan;
- cyan blueprint linework;
- title: `VOLUME₂GYM / RAILROAD 1959`;
- subtitle: `A rulebook compiled into a reinforcement-learning environment`;
- footer: `117 pages · 536 rules · 2,708 tasks · one auditable run`;
- a small label: `Interpretive hero + authentic 1959 source scans`.

Copy the same final hero to both bundles; retain the uncomposited generated scene only in the model bundle.

- [ ] **Step 4: Run tests and inspect the hero**

Run:

```powershell
& .\.venv_win\Scripts\python.exe -m pytest tests\railroad_release\test_visuals.py -q
```

Open the final hero at original detail. Verify period plausibility, title spelling, source-page legibility, left-to-right hierarchy, and absence of generated text or logos in the background.

- [ ] **Step 5: Commit**

```powershell
git add scripts/railroad_release/visuals.py tests/railroad_release/test_visuals.py hf_model_card_work/railroad-1959-release/dataset/assets/hero hf_model_card_work/railroad-1959-release/model/assets/hero
git commit -m "feat: add volume2gym railroad hero"
```

## Task 5: Author the Dataset Card

**Files:**
- Create: `hf_model_card_work/railroad-1959-release/dataset/README.md`
- Create: `hf_model_card_work/railroad-1959-release/dataset/RIGHTS.md`

**Interfaces:**
- Consumes: dataset counts/checksums, source manifest, archival assets, upstream links.
- Produces: Hugging Face dataset card with valid YAML and only relative asset paths.

- [ ] **Step 1: Write the exact YAML front matter**

```yaml
---
pretty_name: "Volume2Gym · Railroad 1959"
language:
- en
license: other
license_name: "Rights-holder authorized research release"
license_link: "RIGHTS.md"
thumbnail: "assets/hero/hero-railroad-volume2gym.webp"
task_categories:
- question-answering
- text-generation
tags:
- reinforcement-learning
- synthetic-data
- historical-documents
- structured-text
- volume2gym
- railroad
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: test
    path: data/test.jsonl
---
```

- [ ] **Step 2: Write the card in the locked narrative order**

Use these exact top-level headings:

```markdown
# The Rulebook Becomes a World
## One volume, one environment
## Open the archive
## From 117 pages to 2,708 decisions
## Dataset anatomy
## The environment contract
## Why this is reinforcement learning
## The indirect low-resource-language hypothesis
## Provenance and rights
## Limitations and intended use
## Reproduce and cite
```

The opening thesis must be:

> Volume2Gym asks a deliberately expansive question: what if any sufficiently structured text could become a small, inspectable world in which a model learns by acting, receiving feedback, and trying again?

The low-resource-language section must include:

> This railroad run does not demonstrate improvement in a low-resource language. It demonstrates an engineering pattern: bounded, community-governed materials can supply tasks, constraints, and evaluation surfaces when conventional pretraining corpora are small. Applying the pattern to a language would require community authority, appropriate licensing, fluent-speaker review, and culturally specific evaluation.

The safety warning must include:

> **Historical research artifact.** These 1959 rules may conflict with current rules, technology, law, terminology, and safe practice. Do not use this dataset to operate trains, qualify personnel, or make real-world safety decisions.

- [ ] **Step 3: Add source-aware captions and full-image links**

The card must embed the hero, light-table spread, contact sheet, three diagrams, and at least six individual full-resolution pages. Captions identify page number, evidence class `confirmed`, and what is visibly supported. Link to `assets/archive/source-manifest.csv` and the directory containing all 117 pages.

The provenance section must distinguish inspected local scans from catalog/resource leads and link:

- Pacific Northwest Chapter, National Railway Historical Society library list: `https://pnwc-nrhs.org/PNWC_Library_Book_list.html`
- Operations Special Interest Group resources: `https://www.opsig.org/Resources/Index`
- DCC Tips reference page for the 1959 CCOR: `https://www.dcctips.com/therulesccor1959.html`

It must also disclose that the project contains a photographed/scanned physical copy with visible gutter curvature, page edges, and wear, while the scanning institution and full custody chain are not documented in the repository. Page 002's serial number 4299 and blank employee loan form are visible historical artifact data; no borrower identity is present.

- [ ] **Step 4: Explain the structured records exactly**

Document:

- 2,708 total scenario tasks;
- deterministic seed-42 split: 2,438 train and 270 test;
- fields: `task_id`, `scenario`, `applicable_rules`, `reference_response`, `prompt`;
- 536 extracted rules, reported separately from the task rows;
- tasks are synthetic scenarios grounded in extracted rules, not verbatim historical incidents.

- [ ] **Step 5: Explain the scorer lineage without conflation**

State that the published Tinker run used normalized exact match plus token-F1, yielding a scalar similarity reward. State that an earlier checked-in environment explored an Anthropic LLM judge with separate safety/procedure/terminology scores, but those judge scores are not the results reported in the model card.

- [ ] **Step 6: Add the rights statement**

Write `RIGHTS.md` with this text:

```markdown
# Rights statement

The repository owner represents that they own or are authorized to publish the reproduced source scans and release graphics in this repository.

Unless a file says otherwise, the historical page images and release artwork are made available for viewing and research through this repository. No public-domain status is asserted, and no additional license to redistribute or commercially reuse the images is granted by this statement. The structured task records and model artifacts remain subject to the notices and upstream terms identified in their respective files.

For permissions beyond repository viewing and research use, contact the repository owner.
```

- [ ] **Step 7: Commit**

```powershell
git add hf_model_card_work/railroad-1959-release/dataset/README.md hf_model_card_work/railroad-1959-release/dataset/RIGHTS.md
git commit -m "docs: author volume2gym railroad dataset card"
```

## Task 6: Author the Model Card

**Files:**
- Create: `hf_model_card_work/railroad-1959-release/model/README.md`
- Create: `hf_model_card_work/railroad-1959-release/model/RIGHTS.md`

**Interfaces:**
- Consumes: `run-summary.json`, charts, hero, adapter metadata, linked dataset.
- Produces: Hugging Face model card with valid YAML and numerically auditable claims.

- [ ] **Step 1: Write the exact YAML front matter**

```yaml
---
language:
- en
license: other
license_name: "Rights-holder authorized research release"
license_link: "RIGHTS.md"
thumbnail: "assets/hero/hero-railroad-volume2gym.webp"
library_name: peft
base_model: Qwen/Qwen3-4B-Instruct-2507
base_model_relation: adapter
pipeline_tag: text-generation
datasets:
- HarleyCooper/volume2gym-railroad-1959
tags:
- reinforcement-learning
- grpo
- lora
- historical-documents
- volume2gym
- railroad
model-index:
- name: Qwen3-4B Railroad Engineer 1959
  results:
  - task:
      type: text-generation
      name: Railroad 1959 held-out scenarios
    dataset:
      type: HarleyCooper/volume2gym-railroad-1959
      name: Volume2Gym Railroad 1959
      split: test
    metrics:
    - type: custom-reward
      name: Deterministic similarity reward
      value: 0.3753234012
    - type: parse-compliance
      name: Parse compliance
      value: 1.0
---
```

- [ ] **Step 2: Write the card in the locked narrative order**

Use these exact top-level headings:

```markdown
# Learning the Rulebook
## Experiment at a glance
## The learning loop
## Actual run results
## What changed
## What the reward measures
## Run audit: two verifier generations
## Reproduce the run
## Use the adapter
## Limitations and safety
## Citation and links
```

The first result paragraph must distinguish:

- training: `0.2465` start, `0.3983` peak at step 57, `0.3338` final at step 76;
- evaluation: `0.2493` at step 0 to `0.3753` at step 60, approximately `+50.5%`;
- evaluation tokens/turn: `255.86` to `90.08`, approximately `−64.8%`;
- evaluation parse compliance: `0.37%` to `100%`;
- four evaluation gates, `n=270` at each gate and `1,080` held-out episodes total;
- `19,504` training rollouts across 77 batches;
- the final batch-77 checkpoint was not separately evaluated, so step 60 is the final evaluation gate rather than an evaluation of the final checkpoint.

The results table must show all evaluation rewards: step 0 `0.2493391492`, step 20 `0.3258478884`, step 40 `0.3531549504`, and step 60 `0.3753234012`. It must also identify the 77-batch arithmetic mean reward `0.3402347642` separately from the episode-weighted mean `0.3403035388`.

- [ ] **Step 3: State the scorer caveat in plain language**

Use this exact warning:

> The reward is lexical, not a safety oracle. It measures normalized exact match and token overlap with a reference response. In this run the logged safety, procedure, and terminology columns carry the same proxy value, so they are not three independent validations and are not plotted as though they were.

Also state that exact match remained zero throughout this run, the scalar reward therefore reduced to whitespace-token multiset F1, and parse compliance records whether the renderer parsed the response format—not whether the answer was correct.

Report the underlying parse counts: training `15/256 → 48/48` and evaluation `1/270 → 270/270`.

- [ ] **Step 4: Document configuration and adapter usage**

Include:

- Qwen3-4B-Instruct-2507;
- LoRA rank 32;
- learning rate `5e-5`;
- 77 batches;
- batch size 32 tasks;
- group size 8 responses per task;
- 10%/270-example evaluation holdout;
- max generation 256 tokens;
- temperature 0.9;
- importance-sampling loss;
- constant-reward groups removed;
- evaluation/save every 20 steps.

Adapter usage must show `transformers` + `peft` loading against the base model and must not claim a merged standalone checkpoint.

- [ ] **Step 5: Link every headline claim to evidence**

Link:

- `run/metrics.jsonl`;
- `run/config.json`;
- `run/checkpoints.jsonl`;
- `analysis/run-summary.json`;
- `analysis/scorer-reference.py`;
- `analysis/railroad-reward-ledger-tinker.csv`;
- dataset card;
- GitHub upstream.

- [ ] **Step 6: Add the rights statement**

Write `RIGHTS.md` with this text:

```markdown
# Rights statement

The repository owner represents that they own or are authorized to publish the reproduced source scans and release graphics in this repository.

Unless a file says otherwise, the historical page images and release artwork are made available for viewing and research through this repository. No public-domain status is asserted, and no additional license to redistribute or commercially reuse the images is granted by this statement. The structured task records and model artifacts remain subject to the notices and upstream terms identified in their respective files.

For permissions beyond repository viewing and research use, contact the repository owner.
```

- [ ] **Step 7: Commit**

```powershell
git add hf_model_card_work/railroad-1959-release/model/README.md hf_model_card_work/railroad-1959-release/model/RIGHTS.md
git commit -m "docs: publish audited railroad model narrative"
```

## Task 7: Validate Both Release Bundles

**Files:**
- Create: `scripts/railroad_release/validate.py`
- Create: `tests/railroad_release/test_validate.py`

**Interfaces:**
- Consumes: dataset/model bundle paths.
- Produces: `validate_bundle(path: Path, kind: Literal["dataset", "model"]) -> list[str]`; empty list means valid.

- [ ] **Step 1: Write failure-first validator tests**

```python
from scripts.railroad_release.validate import validate_bundle


def test_validator_rejects_missing_asset_and_stale_dakota_label(tmp_path):
    (tmp_path / "README.md").write_text(
        "---\nlanguage: [en]\n---\n![hero](assets/missing.webp)\nDakota RL Training",
        encoding="utf-8",
    )
    errors = validate_bundle(tmp_path, "model")
    assert any("missing asset" in error for error in errors)
    assert any("prohibited label" in error for error in errors)


def test_validator_rejects_unsupported_web_features(tmp_path):
    (tmp_path / "README.md").write_text(
        "---\nlanguage: [en]\n---\n<script>alert(1)</script>",
        encoding="utf-8",
    )
    errors = validate_bundle(tmp_path, "model")
    assert any("unsupported HTML" in error for error in errors)
```

- [ ] **Step 2: Run the tests and confirm failure**

Run:

```powershell
& .\.venv_win\Scripts\python.exe -m pytest tests\railroad_release\test_validate.py -q
```

Expected: collection fails because `scripts.railroad_release.validate` does not exist.

- [ ] **Step 3: Implement the complete validation gate**

Check:

- YAML parses and required fields exist;
- no `modality: video`, `preview_image`, custom CSS, script, iframe, or unsupported layout;
- no unresolved `TODO`, `TBD`, `FIXME`, placeholder, local absolute path, token-like secret, or `Dakota RL Training`;
- every relative Markdown/HTML image or link resolves;
- every image opens, is at least 800 px wide when embedded as a major visual, and is below 10 MB;
- dataset manifest has exactly 117 pages and all SHA-256 values match;
- JSONL has 2,438 train and 270 test records with unique/disjoint IDs;
- model summary recomputed from metrics matches card values within displayed precision;
- model aggregate channel equality is disclosed;
- required safety and low-resource-language disclaimers exist.

- [ ] **Step 4: Run focused and complete release tests**

Run:

```powershell
& .\.venv_win\Scripts\python.exe -m pytest tests\railroad_release -q
& .\.venv_win\Scripts\python.exe -m scripts.railroad_release.validate `
  --dataset hf_model_card_work\railroad-1959-release\dataset `
  --model hf_model_card_work\railroad-1959-release\model
```

Expected: all tests pass and `dataset: valid`, `model: valid`.

- [ ] **Step 5: Review staged artifacts for secrets and size**

Run:

```powershell
git diff --check
git status --short
Get-ChildItem -Recurse -File hf_model_card_work\railroad-1959-release |
  Sort-Object Length -Descending |
  Select-Object -First 20 FullName,Length
```

Confirm no access token, `.env`, local username, rollout HTML, adapter weight, or generated cache is staged.

- [ ] **Step 6: Commit**

```powershell
git add scripts/railroad_release/validate.py tests/railroad_release/test_validate.py
git commit -m "test: gate railroad hugging face release"
```

## Task 8: Publish and Verify the Dataset Repository

**Files:**
- Remote update: `datasets/HarleyCooper/volume2gym-railroad-1959`

**Interfaces:**
- Consumes: validated dataset upload bundle.
- Produces: one Hugging Face commit, rendered card, valid Dataset Viewer splits.

- [ ] **Step 1: Confirm authentication and remote head**

Run:

```powershell
hf auth whoami
hf datasets info HarleyCooper/volume2gym-railroad-1959 --format json
```

Expected authenticated user: `HarleyCooper`. Record the pre-upload SHA.

- [ ] **Step 2: Upload the dataset bundle in one commit**

Run:

```powershell
hf upload HarleyCooper/volume2gym-railroad-1959 `
  hf_model_card_work/railroad-1959-release/dataset `
  . `
  --type dataset `
  --commit-message "docs: publish archival Volume2Gym railroad dataset card"
```

Do not pass `--delete`.

- [ ] **Step 3: Verify remote files and Dataset Viewer**

Run:

```powershell
hf datasets info HarleyCooper/volume2gym-railroad-1959 --format json
```

Call:

```text
https://datasets-server.huggingface.co/is-valid?dataset=HarleyCooper/volume2gym-railroad-1959
https://datasets-server.huggingface.co/splits?dataset=HarleyCooper/volume2gym-railroad-1959
https://datasets-server.huggingface.co/first-rows?dataset=HarleyCooper/volume2gym-railroad-1959&config=default&split=train
```

Expected: default train/test splits, 2,438/270 rows after Hub processing. If processing is queued, poll without changing data and report the queued state accurately.

- [ ] **Step 4: Open the rendered card**

Verify the hero, all embedded images, YAML metadata, links, warning, and source-page directory on:

```text
https://huggingface.co/datasets/HarleyCooper/volume2gym-railroad-1959
```

## Task 9: Publish and Verify the Model Repository

**Files:**
- Remote update: `HarleyCooper/Qwen3-4B-RailRoadEngineer1959`

**Interfaces:**
- Consumes: validated model upload bundle and verified live dataset URL.
- Produces: one Hugging Face commit and rendered model card without touching adapter weights.

- [ ] **Step 1: Confirm remote head and existing weights**

Run:

```powershell
hf models info HarleyCooper/Qwen3-4B-RailRoadEngineer1959 --format json
```

Record the pre-upload SHA and confirm `adapter_model.safetensors` and `adapter_config.json` exist.

- [ ] **Step 2: Upload the model bundle in one commit**

Run:

```powershell
hf upload HarleyCooper/Qwen3-4B-RailRoadEngineer1959 `
  hf_model_card_work/railroad-1959-release/model `
  . `
  --type model `
  --commit-message "docs: publish audited Railroad 1959 run card"
```

Do not pass `--delete` and do not include adapter files in the local upload bundle.

- [ ] **Step 3: Verify remote files and render**

Confirm:

- remote SHA changed;
- adapter files retain their previous object identifiers/sizes;
- README metadata resolves the base model and dataset;
- hero, charts, and raw-artifact links render;
- displayed values match `analysis/run-summary.json`;
- no stale Dakota graphic remains embedded in the README.

Open:

```text
https://huggingface.co/HarleyCooper/Qwen3-4B-RailRoadEngineer1959
```

## Task 10: Final Evidence Audit

**Files:**
- Modify: `.superpowers/sdd/progress.md` during execution only.
- No additional release files unless a validation finding requires a fix.

**Interfaces:**
- Consumes: final local bundles, both live Hub repositories, design spec, and this plan.
- Produces: fresh verification evidence and final handoff with both Hub SHAs.

- [ ] **Step 1: Run the full local release verification**

```powershell
& .\.venv_win\Scripts\python.exe -m pytest tests\railroad_release -q
& .\.venv_win\Scripts\python.exe -m scripts.railroad_release.validate `
  --dataset hf_model_card_work\railroad-1959-release\dataset `
  --model hf_model_card_work\railroad-1959-release\model
git diff --check
git status --short
```

- [ ] **Step 2: Re-read every acceptance criterion**

Check the design specification section “Acceptance Criteria” line by line against local files and rendered Hub pages. Record any exception rather than silently weakening a criterion.

- [ ] **Step 3: Request a whole-branch review**

Provide the reviewer:

- the design specification;
- this implementation plan;
- the complete branch diff package;
- local validation output;
- both Hugging Face repository URLs and post-upload SHAs.

Fix every Critical or Important finding, rerun the covering tests, and request re-review.

- [ ] **Step 4: Record final publication evidence**

The final handoff must include:

- dataset URL and commit SHA;
- model URL and commit SHA;
- Dataset Viewer status and row counts;
- local pytest and validator totals;
- generated hero path and exact built-in prompt;
- source manifest path and page count;
- any known limitation that remains.
