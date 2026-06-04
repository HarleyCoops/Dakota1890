# Adaption Dakota QA Environment

Prime Intellect / Verifiers single-turn RL environment for `HarleyCooper/adaption-dakota-english-qa`.

The dataset contains 2,445 English-language question-answer tasks derived from the 1890 Dakota-English Dictionary and remastered by Adaption Labs. This environment adapts the new schema:

- `question`: concise user question
- `answer`: authoritative target answer used for reward
- `enhanced_prompt`: richer instruction/context prompt, used by default
- `enhanced_completion`: retained as metadata only, not treated as ground truth
- source metadata: `pair_id`, `source_files`, `source_pages`, `source_language`

The reward is deterministic and continuous so hosted RL has useful variance: normalized exact-match, Dakota term coverage, character F1 against the target answer, lexical keyword coverage, and a brevity/readability component.

## Local smoke

```bash
uv pip install -e .
uv run vf-eval adaption_dakota_qa -n 5 -r 1
```

## Hosted training starter

```bash
prime train configs/rl/laguna-smoke.toml --plain -y
```

`poolside/Laguna-XS.2` is the intended free model from the Prime example. If it is at capacity, `sprints/Llama-3.2-1B-Instruct` is also currently listed by Prime as free.
