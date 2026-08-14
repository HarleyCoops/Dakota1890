# Push to HarleyCooper/Dakota1890-Grant-Clean

Leave [HarleyCooper/Dakota-.6B](https://huggingface.co/spaces/HarleyCooper/Dakota-.6B) alone.

## After the Space exists

```powershell
cd C:\Users\chris\Dakota1890
git clone https://huggingface.co/spaces/HarleyCooper/Dakota1890-Grant-Clean hf-space-grant-clean
Copy-Item huggingface_space\* hf-space-grant-clean\ -Force
cd hf-space-grant-clean
git add app.py demo.py examples.jsonl requirements.txt README.md
git commit -m "Deploy grant-clean Dakota1890 CPU Space"
git push
```

## Or upload from the Hub UI

Files and versions → Upload:

- `app.py`
- `demo.py`
- `examples.jsonl`
- `requirements.txt`
- `README.md`

## Optional secrets

`TINKER_API_KEY` and `TINKER_SAMPLER_PATH` in Space Settings. The example bank works without them.
