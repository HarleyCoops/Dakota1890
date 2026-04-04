# Grammar-to-RL: Training Language Models on Historical Texts via Compositional Rewards

<div align="right">



</div>

![CI](https://github.com/HarleyCoops/Dakota1890/actions/workflows/ci.yml/badge.svg)
![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)
![Smoke Tests](https://img.shields.io/badge/smoke%20tests-offline-brightgreen)
![Python Package](https://img.shields.io/badge/pypi-dakota--grammar--env-blue)


## Novel Methodology: Zero-to-One Language Model

**This project introduces a novel approach to low-resource language learning by transforming a single historical textbook into a complete, self-contained training ecosystem.**

### Why This Is Now Working

The fundamental insight driving this approach: **grammar rules can be reward functions, and rewards should decompose into linguistic primitives.** When you do this, syntax emerges naturally without an external verifier as judge. This is particularly powerful for RL on non-coding tasks where compositional structure matters.

#### Grammar AS Reward Function

Everyone else treats grammar as either:
- **Preprocessing constraints** (rule-based systems)
- **Post-hoc evaluation** (check grammar after generation)

This project makes grammar rules directly differentiable through compositional rewards. Each rule becomes a gradient signal, not just a binary check.

<div align="center" style="margin: 3rem 0;">

<img src="Public/grammar.jpg" alt="Dakota Grammar - Historical Text Detail" style="width: 100%; max-width: 1400px; height: auto; display: block; margin: 0 auto; border-radius: 4px; box-shadow: 0 8px 24px rgba(0,0,0,0.2);">

</div>

The key advantage: **interpretability**. You can actually see where in the latent space each linguistic level is being encoded. This makes debugging possible: "Oh, the model is failing on ć preservation because the character embedding gradient is being overwhelmed by the semantic gradient."

### Mathematical Formulation

The efficiency of this method comes from treating the historical text not just as data, but as a **computable specification**.

#### 1. The Transformation Function (Book to Environment)

Let $\mathcal{T}$ be the historical textbook source. We define an extraction function $\mathcal{E}$ (the VLM) that maps the raw text into a structured grammar space $\mathcal{G}$:

$$ \mathcal{G} = \{g_1, g_2, \dots, g_N\} = \mathcal{E}(\mathcal{T}) $$

Where each grammar rule $g_k$ acts as a constraint function on the generated token sequence $y$:

$$ g_k: \Sigma^* \rightarrow \{0, 1\} $$

#### 2. The Compositional Reward Function

Standard RLHF or GRPO typically uses a singular reward model $R(y)$. The Grammar-as-signal approach decomposes $R$ into a weighted sum of linguistic primitives.

Let $y_i$ be the $i$-th generation in a group of size $G$. The reward $r_i$ for generation $y_i$ given prompt $x$ is:

$$ r(y_i, x) = \lambda_{diff}(x) \cdot \left[ \alpha \cdot R_{char}(y_i, x) + \beta \cdot R_{morph}(y_i, \mathcal{G}) + \gamma \cdot R_{sem}(y_i, y^*) \right] $$

Where:

*   **$R_{char}$ (Orthography)**: The Intersection-over-Union (or Recall) of required special unicode characters $\mathcal{C}_{spec}$ (e.g., `ŋ`, `š`, `ć`).

    $R_{char} = \frac{|chars(y_i) \cap chars(x)|}{|chars(x) \cap \mathcal{C}_{spec}|}$

*   **$R_{morph}$ (Syntax)**: A binary or scalar check against specific grammar rules $g_k \in \mathcal{G}$ (e.g., affix presence regex).

    $R_{morph} = \frac{1}{|A|}\sum_{a \in A} \mathbb{I}(a \subset y_i) \quad \text{where } A \text{ are required affixes}$

*   **$R_{sem}$ (Semantics)**: Semantic similarity to ground truth (or Dictionary lookup).

*   **Weights**: $(\alpha, \beta, \gamma) = (0.4, 0.4, 0.2)$ per the config.

*   **$\lambda_{diff}$**: The curriculum difficulty multiplier ($1.0 \dots 2.0$).

#### 3. The Modified GRPO Objective

In standard GRPO, we compute the advantage $A_i$ by normalizing rewards within the group. By injecting the new compositional reward, the gradient ascent objective becomes:

$$ \mathcal{L}_{G-GRPO}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \frac{1}{G} \sum_{i=1}^G \underbrace{\left( \frac{r(y_i, x) - \bar{r}}{\sigma_r} \right)}_{\text{Grammar-Verified Advantage}} \cdot \underbrace{\min \left( \frac{\pi_\theta(y_i|x)}{\pi_{old}(y_i|x)}, 1+\epsilon \right)}_{\text{Clipped Policy Ratio}} - \beta_{KL} \mathbb{D}_{KL}(\pi_\theta || \pi_{ref}) \right] $$

Where $\bar{r}$ is the mean reward of the group of $G$ rollouts, and $\sigma_r$ is the standard deviation.

#### 4. The "Wow! Signal": Gradient Signal Density

The mathematical reason for the **160-step convergence** and **97.9% morphology accuracy** is the density of the gradient signal.

In standard Language Modeling (Next Token Prediction), the loss is:

$$ \mathcal{L}_{LM} = -\log P(y_{target} | x) $$

This gives feedback only on exact token matches.

In the **Grammar-GRPO**, the feedback signal $\nabla J$ allows the model to perform gradient ascent on the *structure* of the language directly:

$$ \nabla J(\theta) \propto \sum_{components} w_c \nabla R_c(y) $$

Because $R_{char}$ and $R_{morph}$ are deterministic and verifiable (unlike subjective human preference), the variance of the reward $\sigma^2_r$ is significantly lower than standard RLHF.

**Mathematically, you reduced the noise in the reward signal, allowing the policy to traverse the optimization landscape directly toward the "Grammar Valley" (the manifold where valid Dakota syntax exists).**

## Training Results: RL Performance Visualizations

This section presents comprehensive visualizations from our successful Reinforcement Learning training runs, demonstrating the effectiveness of the grammar-to-RL methodology on the Dakota language. There have been two successful runs: one at 1000 steps (final) and one at 400 steps (initial).

### Run 1: 1000-Step Training (Final)

#### Training Run Details

- **Project**: `dakota-rl-grammar`
- **Entity**: `christian-cooper-us`
- **Trainer Run**: [`7nikv4vp`](https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/7nikv4vp) - `dakota-0.6b-rl-trainer`
- **Orchestrator Run**: [`29hn8w98`](https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/29hn8w98) - `dakota-0.6b-rl-orchestrator`
- **Model**: Qwen3-0.6B-Dakota-Grammar-RL
- **Training Steps**: 1,000 steps (998 completed)
- **Total Samples**: 256,000 samples processed
- **Training Duration**: 1.54 hours (5,537 seconds)

#### Key Achievements

- **190% improvement** in overall reward (0.120 → 0.349)
- **97.9% morphological accuracy** - exceptional performance in affix application
- **53.5% character preservation** - significant improvement for complex orthography
- **90% of improvement achieved in first 160 steps** (16% of training) - demonstrating rapid learning
- **Stable training** with controlled KL divergence throughout

#### Comprehensive Dashboard

The comprehensive dashboard provides an at-a-glance view of all training metrics, combining reward progression, component performance, loss dynamics, entropy, KL divergence, and throughput metrics into a single visualization.

![Comprehensive Dashboard](wandb_visualizations/comprehensive_dashboard.png)

**What this shows**: This multi-panel dashboard synthesizes all key training signals. The top panel shows reward progression with milestone markers indicating when 25%, 50%, 75%, and 90% of total improvement was achieved. The component comparison bar chart (middle-left) reveals the differential performance: morphological accuracy reached 97.9% while character preservation achieved 53.5%, reflecting the challenge of preserving Dakota's complex orthography (ć, š, ŋ, ḣ, ṡ, á, é, í, ó, ú) with a 0.6B parameter model. The loss and entropy panels demonstrate stable optimization, while the KL divergence metrics show controlled policy adaptation without catastrophic forgetting.

**View full run**: [Trainer Run](https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/7nikv4vp) | [Orchestrator Run](https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/29hn8w98)

#### Reward Progression

The reward progression visualization demonstrates the learning trajectory over 1,000 training steps, showing both overall composite reward and individual component breakdown.

![Reward Progression](wandb_visualizations/reward_progression.png)

**What this shows**: The top panel tracks overall reward progression from 0.120 (step 0) to 0.349 (step 999), representing a 190.1% improvement. Milestone markers highlight key learning efficiency points: 25% improvement at step 49 (4.9% of training), 50% at step 71 (7.1%), 75% at step 109 (10.9%), and 90% at step 160 (16%). The rapid initial learning validates the methodology's efficiency - grammar-based tasks provide dense learning signals compared to general language modeling. The bottom panel shows the component breakdown: Morphological Accuracy (green) achieved near-perfect performance (0.979), Character Preservation (orange) showed substantial improvement from 0.038 to 0.535 (14x increase), while the Overall Composite (blue) reflects the weighted combination including semantic components.

**Interpretation**: The divergence between component performances demonstrates that the model learned morphological patterns more effectively than orthographic preservation. This suggests potential areas for future improvement through specialized character-focused training or larger model capacity. The semantic component (20% weight) likely contributes to the composite score being lower than individual components, indicating multi-objective optimization challenges.

**View full run**: [Orchestrator Run](https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/29hn8w98)

#### Training Metrics

This visualization tracks the core training dynamics: policy loss, model entropy (confidence), KL divergence (policy adaptation), and inference probabilities.

![Training Metrics](wandb_visualizations/training_metrics.png)

**What this shows**: 
- **Policy Loss (top-left)**: Values ranged from approximately 1e-5 to 1e-3, typical of GRPO training with conservative learning rates. The log-scale visualization shows consistent small magnitudes indicating stable gradient-based optimization. The shaded region represents ±1 standard deviation, showing controlled variance throughout training.
- **Model Entropy (top-right)**: Decreased from 0.93 to 0.21, indicating the model became significantly more confident in its predictions. Low final entropy (0.21) suggests the model is highly confident, which aligns with the high morphological accuracy achieved.
- **KL Divergence (bottom-left)**: Three metrics track policy adaptation:
  - **Masked KL**: Increased from near-zero to 9.32, indicating substantial policy adaptation for Dakota-specific masked tokens
  - **Overall KL**: Moderate increase from 0.001 to 3.83, suggesting controlled policy adaptation
  - **Unmasked KL**: Remained extremely low (mean: 0.070, final: 0.042), confirming the model preserved general language capabilities while learning Dakota-specific patterns
- **Inference Probabilities (bottom-right)**: Increased from 0.63 to 0.86, showing the model became more certain in its predictions over time.

**Interpretation**: The increasing KL divergence trends indicate active learning and policy adaptation, while the relatively moderate values (especially for unmasked tokens) suggest training remained stable. The model successfully specialized for Dakota grammar while preserving general language understanding, validating the training approach.

**View full run**: [Trainer Run](https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/7nikv4vp)

#### Performance Metrics

Performance metrics track computational efficiency: training throughput (tokens per second) and GPU utilization (Model FLOPS Utilization).

![Performance Metrics](wandb_visualizations/performance_metrics.png)

**What this shows**: 
- **Training Throughput (left)**: Average throughput of 8,178 tokens/sec with consistent performance throughout training. The red dashed line indicates the average, showing stable training execution without significant throughput variations.
- **GPU Efficiency - MFU (right)**: Average Model FLOPS Utilization of 2.68%, indicating GPU efficiency. While this may seem low, it's typical for small models (0.6B parameters) where memory bandwidth rather than compute is often the bottleneck. The consistent MFU suggests stable training without memory pressure or compute bottlenecks.

**Interpretation**: The consistent performance metrics validate stable training execution. The throughput remained stable throughout 1,000 steps, processing 256,000 total samples with an average of 256 samples per step. Peak memory usage was 11.5 GiB, well within reasonable bounds for the model size.

**View full run**: [Trainer Run](https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/7nikv4vp)

### Run 2: 400-Step Training (Initial Breakthrough)

#### Training Run Details

- **Project**: `dakota-rl-grammar`
- **Entity**: `christian-cooper-us`
- **Trainer Run**: [`yut26kcm`](https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/yut26kcm) - `dakota-0.6b-ledger-test-400-trainer`
- **Orchestrator Run**: [`1y33h9zr`](https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/1y33h9zr) - `dakota-0.6b-ledger-test-400-orchestrator`
- **Model**: Qwen3-0.6B-Dakota-Grammar-RL-400
- **Training Steps**: 400 steps (all completed)
- **Total Samples**: 102,400 samples processed
- **Base Model**: Qwen/Qwen3-0.6B (small instruct model optimized for RL)

#### Key Achievements

- **150.3% improvement** in overall reward (0.128 → 0.321, peak: 0.345)
- **Rapid learning**: 90% of improvement achieved in first 85 steps (21.25% of training)
- **Sample efficiency**: 0.000483 improvement per step - demonstrating dense learning signals
- **Stable training**: Controlled KL divergence with unmasked KL remaining low (mean: 0.094, final: 0.092)
- **Policy confidence**: Entropy decreased from 0.93 to 0.28, showing increased model certainty

#### Reward Progression (400 Steps)

**Interpretation**: The rapid learning trajectory suggests that compositional reward functions can support efficient learning on qualitative linguistic tasks. The milestone markers show consistent acceleration, with each 25% improvement requiring progressively fewer steps, indicating the model is learning to learn more effectively. These early signals hint that GRPO could extend beyond coding/math domains when rewards are decomposed into linguistic primitives, but this still needs more confirmation.

**View full run**: [Orchestrator Run](https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/1y33h9zr)

### Run 3: 30B Tinker (Thinking Machines Final)

- **Project / Run ID**: `dakota-rl-grammar` / [`i55d4x26`](https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/i55d4x26)
- **Base Model**: Qwen/Qwen3-30B-A3B-Instruct-2507 (LoRA rank 32)
- **Steps**: 199 (0–198) on Tinker; checkpoint at `tinker://da1ef918-d67a-5080-b500-dd1256db9ca7:train:0/sampler_weights/final`
- **Composite reward**: 0.105 → **0.442 peak (step 116)** → 0.317 final
- **Character preservation**: 0.265 → **0.699 peak (step 185)** → 0.619 final
- **Affix accuracy**: 0.957 → **1.000** (stayed perfect in later stages)
- **Exact match**: 0.001 → **0.337 peak (step 116)** → 0.100 final; length penalty rose to 1.0 by step 52

![Comprehensive Dashboard (30B Tinker)](wandb_visualizations/qwen30b/comprehensive_dashboard.png)

### GRPO for Qualitative Tasks: Early Signals

**This work suggests that GRPO (Group Relative Policy Optimization) can achieve strong learning on qualitative linguistic tasks when rewards are properly decomposed into interpretable components, contingent on continued results.** This is significant because:

#### Why This Matters

GRPO has been successfully applied to **quantitative domains** (code generation, mathematical reasoning) where correctness is verifiable and rewards are clear. However, **qualitative tasks** like language learning, translation, and grammar have traditionally been considered unsuitable for RL because:

1. **Subjective evaluation**: "Is this translation good?" lacks clear criteria
2. **Multi-dimensional quality**: A translation can be semantically correct but orthographically wrong
3. **Nuanced feedback**: Binary correct/incorrect fails to capture partial correctness

#### Our Solution: Compositional Rewards

By decomposing rewards into **linguistic primitives** (character preservation, morphological accuracy, semantic correctness), we transform qualitative tasks into **quantitatively optimizable objectives**:

- **Character preservation (40%)**: Verifiable Unicode-level correctness
- **Morphological accuracy (40%)**: Pattern-matching against grammar rules
- **Semantic correctness (20%)**: Meaning preservation metrics

This decomposition enables GRPO to work effectively because:
- **Each component is independently verifiable** (no human judgment needed)
- **Gradients flow through each component** (model learns what to prioritize)
- **Multi-dimensional feedback** (model knows exactly what it got wrong)

#### Key Results Demonstrating Significance

1. **150.3% improvement in 400 steps** - Comparable to GRPO performance on coding tasks
2. **90% improvement in 21% of training** - Demonstrates dense learning signals from compositional rewards
3. **Low unmasked KL (0.092)** - Model specializes without catastrophic forgetting
4. **Stable training dynamics** - No reward hacking or instability issues

#### Implications

This work hints that **GRPO may not be limited to quantitative domains**. When qualitative tasks are decomposed into verifiable components, they may become more learnable, opening possibilities for:

- **Low-resource language learning** (this work)
- **Style transfer** (decompose into syntax, semantics, register)
- **Dialogue systems** (decompose into coherence, relevance, appropriateness)
- **Creative tasks** (decompose into structure, originality, coherence)

### Methodology Validation

These results provide early evidence for the core methodological idea: transforming grammar rules from a 130-year-old historical textbook into verifiable RL environments. The strong morphological accuracy (97.9%) suggests that:

1. **Rule Extraction Quality**: The VLM-based extraction successfully captured testable grammar patterns from historical text, preserving morphological rules in a format suitable for RL training.

2. **Task Generation Effectiveness**: The conversion of 1,497 grammar rules into 10,576 RL tasks created sufficient training signal for the model to learn morphological patterns. The high accuracy suggests the task generation process successfully encoded the grammar rules as verifiable constraints.

3. **Compositional Reward Structure**: The decomposition into character, morphology, and semantic components enabled fine-grained learning. The model learned morphological patterns more effectively than orthographic preservation, suggesting potential areas for future improvement.

4. **Sample Efficiency**: The rapid initial learning (90% improvement in 16% of training) demonstrates the methodology's efficiency for low-resource language scenarios. With only 256,000 samples processed over 1,000 steps, the model achieved substantial improvement.

### Visualization Generation

These visualizations were generated using `scripts/create_rl_visualizations.py`, which loads data from Weights & Biases and creates publication-quality plots. To regenerate with updated data:

```bash
# For the 1000 step run (default):
python scripts/create_rl_visualizations.py \
    --trainer-id 7nikv4vp \
    --orchestrator-id 29hn8w98 \
    --project dakota-rl-grammar \
    --entity christian-cooper-us
```

All visualization code and data are available in the repository for reproducibility and further analysis.

### Reward Ledger: Transparent Reward Decomposition

We log detailed reward component breakdowns to provide full transparency into how rewards are computed. The reward ledger exposes all internal components, weights, penalties, and multipliers used in reward calculation.

![Reward Ledger Reconciliation](wandb_analysis/reward_ledger.png)

**What this shows**: The reward ledger visualization demonstrates how individual components combine to form the final reward. Each line represents a different component or intermediate computation:

- **Weighted components**: Individual reward components (exact match, character overlap, pattern matching, affix accuracy) multiplied by their respective weights
- **Composite pre**: Weighted sum of components before applying penalties and multipliers
- **Length penalty multiplier**: Applied to penalize overly long responses (1.0 = no penalty, <1.0 = penalty)
- **Difficulty multiplier**: Applied based on task difficulty (1.0 for easy, up to 2.0 for expert)
- **Composite predicted**: Final computed reward from components
- **Reward scalar**: Actual reward used in training (should match composite_predicted)

The visualization validates that the logged components correctly reconstruct the final reward, with `composite_diff` tracking any numerical discrepancies.

**Ledger Fields**: The ledger logs raw and normalized component scores, weights, penalties, difficulty multipliers, and intermediate composites. All data is logged to both Weights & Biases (under `ledger/*` namespace) and CSV (`wandb_analysis/reward_ledger.csv`) for offline analysis.

<details><summary>Head/tail of reward_ledger.csv</summary>

```text
<!-- Auto-generated by scripts/analysis/make_ledger_snippet.py -->
<!-- Run: python scripts/analysis/make_ledger_snippet.py -->
```

</details>

**Integration**: The ledger is automatically computed by `DakotaGrammarRubric.score()` and can be accessed via `environment.get_reward_ledger()`. See `dakota_rl_training/utils/LEDGER_INTEGRATION.md` for integration instructions.

---

<div align="center">


</div>

<div align="center" style="margin: 2rem 0;">

<img src="Public/Prepositions.jpg" alt="Dakota Prepositions - High Detail Scan" style="width: 100%; max-width: 1400px; height: auto; display: block; margin: 0 auto; border-radius: 4px; box-shadow: 0 8px 24px rgba(0,0,0,0.2);">

</div>

<div align="center" style="margin-bottom: 2rem;">

*Exceptional level of detail preserved from the 1890 source material — every character, accent, and linguistic nuance captured with precision*

</div>

<div align="center" style="margin-bottom: 2rem;">

**All images, extractions, and datasets are freely available in this repository:** [data/processed_images](https://github.com/HarleyCoops/Dakota1890/tree/main/data/processed_images)

</div>



<div align="center">

![Dakota Dictionary Sample](Public/Dictionary.jpeg)

</div>



<div align="center">

![Overview](Public/overview.jpg)

</div>

`/data` now preserves every artifact of the Dakota Grammar preliminary RL build, from VLM extraction outputs and audit traces to the structured rule corpora and PrimeIntellect-ready task sets, making the full pipeline from 1890 scans to verifiable environments reproducible and inspectable. This snapshot captures the linguistic provenance, reasoning logs, and RL curriculum that power the repository's closed-loop grammar gym.

<div align="center">

# Ask DeepWiki How This Works

**[https://deepwiki.com/HarleyCoops/Dakota1890](https://deepwiki.com/HarleyCoops/Dakota1890)**

</div>

### The Process So Far

We take a 1890 grammar textbook and split it into two complementary components that feed into each other:

![Dakota Grammar Gym Methodology](docs/diagram.png)

### Why This Is Novel

**1. Single-Source Completeness**
   - One textbook provides BOTH grammar rules AND vocabulary
   - No need for separate corpora or parallel datasets
   - Self-contained: grammar validates what dictionary teaches

**2. Grammar as Verifiable RL Environment**
   - Grammar rules → RL reward functions
   - Each rule becomes a testable constraint
   - PrimeIntellect TOPLOC verifies special character preservation
   - Compositional rewards: characters + affixes + semantics

**3. Bi-Directional Synthesis**
   - Forward: {dakota:english} from dictionary
   - Reverse: {english:dakota} synthetic generation
   - Stoney Nakoda Q&A methodology adapted for full sentences
   - Grammar gym validates synthetic outputs

**4. Closed-Loop Training**
   - Dictionary words → Synthetic sentences
   - Synthetic sentences → Grammar gym (RL verification)
   - RL feedback → Improved generation
   - Grammar rules from same source ensure consistency

**5. Historical Text → Modern RL**
   - 1890s textbook → 2025 distributed RL training
   - VLM extraction bridges 135-year gap
   - No OCR training required
   - Preserves original orthography perfectly

### Key Differentiators from Existing Literature

**Traditional Approaches:**
- Require parallel corpora (expensive, rare for low-resource languages)
- Separate grammar documentation and training data
- Rule-based systems OR data-driven (not both)
- No verifiable reward functions for morphology

**Our Approach:**
- Single textbook = complete ecosystem
- Grammar rules ARE the reward function
- Synthetic data validated by grammar from same source
- Verifiable character-level preservation (TOPLOC)
- Fully reproducible (just needs scanned textbook + VLM API)

### Research Contributions

1. **VLM-Based Historical Text Extraction**
   - 92-95% accuracy on 130-year-old text
   - No OCR training required
   - Preserves complex orthography (ć, š, ŋ, ḣ)

2. **Grammar-to-RL Pipeline**
   - Linguistic rules → Compositional reward functions
   - Multi-turn environments for morphology
   - Curriculum learning (easy → medium → hard)

3. **Closed-Loop Synthesis**
   - Dictionary pairs → Grammar-validated sentences
   - Bi-directional generation (dakota ↔ english)
   - Self-consistency through shared source

4. **Distributed Verification**
   - TOPLOC for Unicode preservation
   - Verifiable special character handling
   - Prevents corruption in untrusted workers

**Potential Impact**: This methodology could be applied to ANY low-resource language with historical grammar documentation, democratizing language preservation through modern AI.

---

## Overview

This project uses modern Vision-Language Models (VLMs) to extract and preserve the Dakota language from historical 1890s grammar texts and dictionaries. Our goal is to create high-quality structured datasets that enable **Reinforcement Learning-based fine-tuning** of open-source language models on Dakota, contributing to Indigenous language revitalization efforts.

**Key Innovation**: We've proven that VLMs can extract complex orthographic features (special characters like ć, š, ŋ) from 130-year-old texts **without requiring traditional OCR training**, achieving 92-95% accuracy through prompt engineering alone. We then transform these extractions into **verifiable RL training tasks** with compositional reward functions that preserve linguistic structure.

**Universal Framework**: While demonstrated on Dakota, this pipeline is **language-agnostic** and can be applied to any low-resource language with historical documentation. The only requirements are: (1) scanned images of source texts, (2) a VLM API key, and (3) definition of the language's special characters. The extraction and RL training infrastructure adapts automatically.

---

## The Language: Dakota

Dakota is a Siouan language historically spoken by the Dakota people across the Great Plains. The language uses a rich orthographic system with special characters to represent sounds unique to Dakota phonology:

- **Special consonants**: ć (c-acute), š (s-caron), ŋ (eng), ḣ (h-dot)
- **Pitch accents**: á, é, í, ó, ú
- **Long vowels**: Represented by doubled letters (aa, ii, oo)
- **Syllable structure**: Marked with hyphens (e.g., é-iŋ-hiŋ-tku)

**Example Dakota text**:
```
Wićašta wańŋ éińhińtku nonpa : unkań hakakata kiń he atkuku kiń heéiya
Man     a    son-his    two   : and   youngest  the  that father-his the said-to-him
"A man had two sons: and the youngest said to his father"
```

---

## Complete Pipeline: From 1890 Textbook to RL Training

### Step 1: Source Material Acquisition
**Input**: Stephen Return Riggs' Dakota Grammar and Dictionary (1890)
- 665 pages from Internet Archive
- JP2 format, 2000x3000px scans
- **Pages 31-92**: Grammar rules (62 pages)
- **Pages 93-440**: Dictionary entries (~10,000 words)

### Step 2: Image Conversion
**Script**: `scripts/extraction/convert_all_images.py`
```bash
python scripts/extraction/convert_all_images.py
```
**Output**: 440 JPEG images in `data/processed_images/`

### Step 3A: Grammar Extraction (RL Gym Foundation)
**Script**: `scripts/extraction/extract_grammar_pages.py`
```bash
python scripts/extraction/extract_grammar_pages.py --pages 31-92 --yes
```
**Process**:
- Claude Sonnet 4.5 extracts linguistic rules
- Specialized prompt for Dakota orthography
- Interlinear text parsing
- Confidence scoring

**Output**: `data/grammar_extracted/`
- Page-level JSON files for the grammar section
- 1,497 grammar rules extracted
- 6 categories: morphology, syntax, phonology, conjugation, particles, translation

**Key Innovation**: Each grammar rule becomes a verifiable constraint

### Step 3B: Dictionary Extraction (Vocabulary Source)
**Script**: `scripts/extraction/extract_dakota_dictionary_v2.py`
```bash
python scripts/extraction/extract_dakota_dictionary_v2.py --pages 93-440
```
**Output**: `data/extracted/`
- ~10,000 {dakota:english} pairs
- Etymology and usage notes
- Part of speech tags

<div align="center" style="margin: 3rem 0;">

<img src="Public/dictionary2.jpg" alt="Dakota Dictionary - Historical Text Detail" style="width: 100%; max-width: 1400px; height: auto; display: block; margin: 0 auto; border-radius: 4px; box-shadow: 0 8px 24px rgba(0,0,0,0.2);">

</div>

### Step 4: Grammar → RL Environment Conversion
**Script**: `scripts/rl/organize_grammar_for_rl.py`
```bash
python scripts/rl/organize_grammar_for_rl.py --input data/grammar_extracted/
```
**Process**:
- Rules → RL task format
- Positive/negative example generation
- Difficulty estimation (easy/medium/hard)
- Affix and special character tagging

**Output**: `data/rl_training_rules/`
- 1,497 rules organized by category
- Verification patterns defined
- Source page tracking

### Step 5: RL Task Generation
**Script**: `scripts/conversion/convert_rules_to_primeintellect.py`
```bash
python scripts/conversion/convert_rules_to_primeintellect.py
```
**Process**:
- 1 rule → 5.5 tasks average
- Multiple task types per rule:
  - Morphology application
  - Translation (dakota → english)
  - Reverse translation (english → dakota)
  - Syntax analysis
  - Pattern identification

**Output**: `dakota_rl_training/datasets/`
- **10,576 total tasks** from 1,497 rules
- Curriculum-ready:
  - Easy: 1,998 tasks
  - Medium: 2,155 tasks
  - Hard: 398 tasks
  - Advanced: 1,106 tasks

### Step 6: Synthetic Dataset Generation
**Script**: `scripts/conversion/generate_synthetic_dakota.py` (secondary SFT baseline)
```bash
python scripts/conversion/generate_synthetic_dakota.py \
    --extracted-dir data/extracted \
    --pairs-per-language 8 \
    --output-file data/bilingual_training_set.jsonl

python scripts/conversion/convert_extracted_to_chat.py \
    --input-file data/bilingual_training_set.jsonl \
    --output-dir OpenAIFineTune
```
**Process**:
1. Load extracted dictionary entries from `data/extracted/`
2. Generate Gemini-backed Q&A variations:
    - "How do you say X in Dakota?" → dakota_word
    - "Translate X to English" → english_word
    - "Use X in a sentence" → full_sentence
3. Convert synthetic Q&A into OpenAI chat JSONL
4. Preserve Dakota orthography and source provenance fields for review

**Output**:
- `data/bilingual_training_set.jsonl`
- `OpenAIFineTune/dakota_train.jsonl` (980 examples)
- `OpenAIFineTune/dakota_valid.jsonl` (245 examples)

### Step 7: RL Environment Setup
**Script**: `scripts/rl/create_grammar_rl_environment.py`
```bash
python scripts/rl/create_grammar_rl_environment.py --rules-dir data/rl_training_rules/
```
**Creates**:
- Packaged Dakota grammar environment in `environments/dakota_grammar_translation/`
- `DakotaGrammarRubric`: Compositional rewards with reward-ledger support
- Consumer-hardware local checks and documented remote/Tinker path
- TOPLOC-compatible interfaces for distributed runs

**Output**: importable `load_environment()` package and JSONL task datasets

### Step 8: Training on PrimeIntellect
**Script**: `dakota_rl_training/train.py`
```bash
# Install PrimeIntellect
pip install git+https://github.com/PrimeIntellect-ai/verifiers.git
pip install git+https://github.com/PrimeIntellect-ai/prime-rl.git

# Launch distributed training
cd dakota_rl_training
prime-rl train \
    --config configs/training_config.yaml \
    --num-workers 4 \
    --use-toploc \
    --wandb-project dakota-rl-grammar
```

**Training Loop**:
1. Model generates Dakota text
2. `DakotaGrammarEnv` verifies against grammar rules
3. `DakotaGrammarRubric` calculates reward:
   - Character preservation: 0-1.0
   - Affix accuracy: 0-1.0
   - Semantic accuracy: 0-1.0
   - Composite with difficulty multiplier
4. GRPO updates policy
5. Curriculum advances (easy → medium → hard)

**Synthetic Data Integration**:
- Synthetic sentences used as training examples
- Grammar gym validates each generation
- Closed loop: dictionary → sentences → grammar validation

---

## Results & Statistics

### Extraction Quality
- **Grammar extraction**: 97% average confidence
- **Dictionary extraction**: 92-95% accuracy
- **Special character preservation**: 100% (ć, š, ŋ, ḣ preserved exactly)

### Dataset Size
- **Grammar rules**: 1,497 rules across 6 categories
- **RL training tasks**: 10,576 tasks
- **Dictionary entries**: ~10,000 word pairs
- **OpenAI chat baseline**: 980 train / 245 validation examples

### Training Metrics (Expected)
- **Character accuracy**: >90% for all special chars
- **Affix accuracy**: >85% for common morphology
- **Translation accuracy**: 85-90% (easy), 65-70% (hard)
- **Training time**: 8-12 hours on distributed workers

### Cost Analysis
- **Grammar extraction**: $15.50
- **Dictionary extraction**: ~$87.50
- **RL training**: Distributed (free via PrimeIntellect)
- **Total**: ~$103 for complete system

---

## Key Files & Documentation

### Core Scripts

**Extraction Scripts** (`scripts/extraction/`):
1. `convert_all_images.py` - JP2 → JPEG conversion
2. `extract_grammar_pages.py` - Grammar rule extraction
3. `extract_dakota_dictionary_v2.py` - Dictionary extraction

**Conversion Scripts** (`scripts/conversion/`):
4. `convert_rules_to_primeintellect.py` - Generate RL tasks
5. `generate_synthetic_dakota.py` - Synthetic data generation
6. `convert_extracted_to_chat.py` - Convert to chat format

**RL Scripts** (`scripts/rl/`):
7. `organize_grammar_for_rl.py` - Rules → RL format
8. `create_grammar_rl_environment.py` - RL environment setup
9. `publish_dakota_environment.py` - Publish to PrimeIntellect
10. `run_complete_grammar_pipeline.py` - Run full pipeline

**Training**:
11. `dakota_rl_training/train.py` - Launch training

### Documentation
- `PIPELINE.md` - Canonical Dakota DAG and active entrypoints
- `SETUP.md` - Sandbox-ready environment and smoke-test instructions
- `REPO_MAP.md` - Step 0 repository inventory and classifications
- `PIPELINE_AUDIT.md` - Answers to the architecture and grant-critical audit questions
- `CLEANUP_PLAN.md` - Conservative pruning/archive decisions
- `VALIDATION_REPORT.md` - Pre/post cleanup validation results
- `GRANT_TECHNICAL_SUMMARY.md` - Grant-ready technical overview

### Output Directories
- `data/processed_images/` - Converted JPEG images (440 files)
- `data/grammar_extracted/` - Raw grammar rule extractions
- `data/rl_training_rules/` - Organized RL rules (1,497 rules)
- `data/extracted/` - Dictionary page extractions (~10,000 entries)
- `data/bilingual_training_set.jsonl` - Synthetic QA baseline corpus
- `OpenAIFineTune/` - OpenAI chat-format baseline splits (980 / 245)
- `dakota_rl_training/datasets/` - RL training tasks (10,576)
- `dakota_rl_training/checkpoints/` - Model checkpoints

---

## Technical Architecture

### VLM Extraction Layer
- **Primary**: Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`)
- **Prompt Engineering**: Specialized Dakota orthography preservation
- **Output**: Structured JSON with confidence scores

### RL Training System
- **Local check model**: Qwen/Qwen3-0.6B
- **Published adapter base**: Qwen/Qwen3-30B-A3B-Instruct-2507
- **Method**: LoRA fine-tuning (rank 32 on the published adapter)
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Frameworks**: PrimeIntellect prime-rl and Thinking Machines Tinker
- **Verification**: TOPLOC for distributed Unicode validation

### Reward Function
Multi-component compositional rewards:
```python
reward = (
    0.4 * character_preservation +  # Dakota special chars
    0.4 * affix_accuracy +          # Morphological correctness
    0.2 * semantic_accuracy         # Translation quality
) * difficulty_multiplier           # 1.0x - 2.0x
```

### Curriculum Learning
Progressive difficulty over 3 stages:
1. Easy tasks (1,973) → baseline orthography and simple morphology
2. Medium tasks (5,294) → the main training mass
3. Hard/advanced tasks (1,172 + 2,137) → compositional morphology and reverse translation

---

## Research Significance

### Novel Contributions

1. **VLM-Based Historical Text Processing**
   - First demonstration of VLM special character extraction from 19th-century texts
   - No OCR training required
   - 92-95% accuracy on complex orthography

2. **Grammar-to-RL Pipeline**
   - Linguistic rules → Verifiable reward functions
   - Compositional rewards for morphology
   - Multi-turn environments for complex tasks

3. **Closed-Loop Synthesis Method**
   - Single textbook → Complete training ecosystem
   - Dictionary + Grammar = Self-validating system
   - Bi-directional generation with verification

4. **Distributed Verification (TOPLOC)**
   - First application to Unicode preservation
   - Prevents character corruption in untrusted workers
   - Critical for low-resource language training

### Potential Applications

**Immediate**:
- Dakota language revitalization
- Educational tools for language learners
- Grammar-aware translation systems

**Broader Impact**:
- Template for other Siouan languages (Lakota, Nakota, Stoney)
- Methodology for any historical grammar textbook
- Framework for low-resource language preservation
- Verifiable distributed training for linguistic tasks

### Generalized RL Use Cases

The same closed-loop, rule-to-reward pattern extends beyond low-resource language learning:

- Legal & Compliance: Turn the US Tax Code (text) into a reward function to train models that avoid non-compliant financial advice.
- Biology: Turn protein folding heuristics papers into reward constraints for generative chemistry.
- Code Migration: Turn a 1970 COBOL manual into a verification environment to train a Python-to-COBOL translation agent.
- Industrial Safety (1950s Rail): Build an RL environment that mirrors a postwar railroad engineer's operating rules using the procedures documented in `Public/1959RailRoadCodeRL.pdf`, enforcing safety, dispatch, and signaling constraints in generation.

## Project Expansion: Railroad Engineer 1959

Building on the success of the Dakota grammar pipeline, we are applying the same **Grammar-to-RL** methodology to industrial safety.

- **Objective**: Train RL agents to operate trains strictly according to the **1959 Consolidated Code of Operating Rules**.
- **Methodology**:
    1.  **Extraction**: VLM extracts rules from `Public/1959RailRoadCodeRL.pdf`.
    2.  **Environment**: `RailroadEnv` verifies adherence to safety protocols, signaling rules, and terminology.
    3.  **Reward**: Compositional reward function prioritizing **Safety** (critical), **Procedure** (sequence), and **Terminology** (exact phrasing).
- **Status**: Externalized from this repo during step-0 cleanup and now maintained in the `Daily` monorepo under `Projects/RailroadEngineer1959/`.

### Future Research Directions

1. **Multi-Language Extension**
   - Apply to related languages (Lakota, Stoney Nakoda)
   - Cross-lingual transfer learning
   - Comparative Siouan linguistics

2. **Enhanced Synthesis**
   - Context-aware sentence generation
   - Dialogue system integration
   - Cultural knowledge embedding

3. **Evaluation Framework**
   - Linguistic acceptability metrics
   - Native speaker validation
   - Grammaticality benchmarks

4. **Morphological Productivity**
   - Test generalization to unseen words
   - Measure affix composition ability
   - Analyze systematic errors

---

## Requirements

### Software Dependencies
```bash
# Core extraction
pip install anthropic pillow python-dotenv

# Synthetic QA baseline
pip install google-generativeai openai

# RL training
pip install git+https://github.com/PrimeIntellect-ai/verifiers.git
pip install git+https://github.com/PrimeIntellect-ai/prime-rl.git
```

### API Keys
- `ANTHROPIC_API_KEY` - Claude Sonnet 4.5 (required)
- `GOOGLE_API_KEY` - Gemini synthetic QA generation (optional)
- `OPENAI_API_KEY` - OpenAI SFT baseline submission or readiness checks (optional)
- `HF_TOKEN` - Hugging Face inference or publishing flows (optional)

### System Requirements
- **Extraction**: Python 3.8+, 8GB RAM
- **Training**: Distributed via PrimeIntellect (or local GPU)
- **Storage**: ~5GB for images + extractions

---

## Quick Start

### 1. Setup
```bash
git clone https://github.com/HarleyCoops/Dakota1890.git
cd Dakota1890
python -m pip install -r requirements.txt

# Add API keys to .env
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

### 2. Convert Images
```bash
python scripts/extraction/convert_all_images.py
# Output: 440 images in data/processed_images/
```

### 3. Extract Grammar
```bash
python scripts/extraction/extract_grammar_pages.py --pages 31-92 --yes
# Output: 1,497 rules in data/grammar_extracted/
```

### 4. Generate RL Tasks
```bash
python scripts/rl/organize_grammar_for_rl.py --input data/grammar_extracted/
python scripts/conversion/convert_rules_to_primeintellect.py
# Output: 10,576 tasks in dakota_rl_training/datasets/
```

### 5. Check the SFT Baseline Assets
```bash
python scripts/rl/dakota_openai_finetune.py --check-only
# Output: readiness report for OpenAIFineTune/dakota_train.jsonl and dakota_valid.jsonl
```

### 5b. Launch the OpenAI SFT Baseline
```bash
# This submits a paid remote fine-tuning job.
python scripts/rl/dakota_openai_finetune.py
```

### 6. Launch Training
```bash
cd dakota_rl_training
python train.py --config configs/training_config.yaml
# Follow instructions to install PrimeIntellect and launch
```

### Curriculum Section

The repo ships with two trainer presets:

- `dakota_rl_training/configs/train_30b.toml` – the lightweight config we use in the quick start above. It has no curriculum logic; every RL step samples uniformly from whatever dataset the environment serves.
- `dakota_rl_training/configs/train.toml` (and the matching `training_config.yaml` used by `python train.py`) – these contain an explicit `[curriculum]` block with staged datasets (easy → medium → hard) plus target step ranges.

To activate the curriculum when launching through the Prime RL CLI, point each component at the staged configs:

```bash
uv run rl \
  --trainer @ ~/dakota-rl-training/configs/train.toml \
  --orchestrator @ ~/dakota-rl-training/configs/orch.toml \
  --inference @ ~/dakota-rl-training/configs/infer.toml \
  ... # other flags (gpu ids, wandb, etc.)
```

The orchestrator/inference presets that accompany `train.toml` live in the same folder (`orch.toml`, `infer.toml`) and expect the staged datasets described in `training_config.yaml`. If you stick with the `*_30b.toml` files, no curriculum will trigger. In short: choose the config trio you pass to `uv run rl` based on whether you want uniform sampling (`*_30b.toml`) or curriculum-driven sampling (`train.toml` + `orch.toml` + `infer.toml`).

---

## Citation

If you use this work, please cite:

```bibtex
@software{dakota_rl_grammar_2025,
  title = {Dakota Language Preservation Through VLM-Based Grammar Extraction and RL Training},
  author = {Cooper, Christian H.},
  year = {2025},
  url = {https://github.com/[repo]},
  note = {Novel closed-loop methodology for low-resource language training}
}
```

---

## Acknowledgments

- **Stephen Return Riggs**: Original Dakota grammar documentation (1890)
- **Internet Archive**: Historical text preservation
- **PrimeIntellect**: Distributed RL training infrastructure
- **Anthropic**: Claude Sonnet 4.5 VLM capabilities
- **Dakota Language Community**: Ongoing language revitalization efforts

---

## License

Code is licensed under **Apache-2.0** (see `LICENSE`). The included historical Dakota text (Riggs, 1890) is in the **public domain**; see `DATA_LICENSE.md` for details and scope.

---

## Contact & Contributions

This methodology is designed to be reproducible and extensible. Contributions welcome for:
- Additional language applications
- Improved synthesis methods
- Enhanced verification functions
- Evaluation frameworks

See `docs/CONTRIBUTING.md` for guidelines.
