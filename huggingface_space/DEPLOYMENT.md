# Deploy HarleyCooper/Dakota1890-Grant-Clean

New CPU Gradio Space for the grant-clean 30B Tinker run (`cebp9acs`).
Do **not** overwrite [HarleyCooper/Dakota-.6B](https://huggingface.co/spaces/HarleyCooper/Dakota-.6B).
That Space stays the old 0.6B Prime Intellect GPU demo.

## What this Space is

- Hardware: **CPU Basic** (no GPU, no `spaces.GPU`)
- App: curated holdout examples + gold from frozen `grammar_tasks_heldout.jsonl`
- Deps: Gradio 5 only (`gradio>=5.20.0,<6`). Do **not** add `tinker` to `requirements.txt` — optional import already handles a missing package; installing Tinker on a CPU Space is the next crash.
- Live infer: optional remote Tinker sampler when secrets are set
- Not hosted here: `Qwen/Qwen3-30B-A3B-Instruct-2507` weights, `owf98569`, or the 0.6B model

## 1. Create the Space

1. Open https://huggingface.co/spaces
2. **Create new Space**
3. Name: `HarleyCooper/Dakota1890-Grant-Clean`
4. SDK: **Gradio**
5. Hardware: **CPU Basic**
6. Visibility: Public
7. Create Space

## 2. Optional secrets (live infer only)

In the Space **Settings → Secrets**:

| Secret | Required | Value |
| --- | --- | --- |
| `TINKER_API_KEY` | No | Thinking Machines API key |
| `TINKER_SAMPLER_PATH` | No | Grant-clean sampler URI |

Recommended `TINKER_SAMPLER_PATH` (session from `cebp9acs`; `sampler_weights/final` was not confirmed in-repo logs, so set this explicitly if the live path differs):

```
tinker://dc44ca83-ce9e-5c91-a38d-0e866549f397:train:0/sampler_weights/final
```

Do not set this to the `owf98569` / `1f23df9c-...` 35B sampler.

Without `TINKER_API_KEY` the Space must still load: examples and gold show, live button explains that sampling is off.

## 3. Push this folder

From the repo, upload only `huggingface_space/`:

```powershell
cd C:\Users\chris\Dakota1890
python -c @"
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    repo_id='HarleyCooper/Dakota1890-Grant-Clean',
    repo_type='space',
    folder_path='huggingface_space',
    path_in_repo='.',
    ignore_patterns=['**/__pycache__/**', '*.pyc'],
    commit_message='Deploy grant-clean Dakota1890 CPU Space',
)
"@
```

Or copy `app.py`, `demo.py`, `examples.jsonl`, `requirements.txt`, and `README.md` into a clone of the Space repo and `git push`.

## 4. Check the build

1. Open https://huggingface.co/spaces/HarleyCooper/Dakota1890-Grant-Clean
2. Confirm it starts on CPU with no Tinker key (example dropdown + gold)
3. After secrets are set, rebuild and try **Run live Tinker sampler** on an EN→Dakota example
4. Confirm the status text names the grant-clean sampler, not `owf98569` or the 0.6B model id
