"""End-to-end check of riggs1890-lexgym against a scripted OpenAI-compatible server.

Run: python tests/e2e_scripted_model.py   (needs the package installed; no API key)
"""
import asyncio
import gzip
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import verifiers as vf

from riggs1890_lexgym import load_environment
from riggs1890_lexgym.environment import DATA

PORT = 8765
MODE = {"policy": "oracle"}
GOLD = {}
for name in ("eval_heldout_rule", "eval_recall"):
    with gzip.open(DATA / f"{name}.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            GOLD[row["prompt"]] = row


def reply(messages: list[dict]) -> str:
    question = messages[1]["content"]
    task = GOLD[question]
    word = question.rstrip("?").split()[-1]
    if MODE["policy"] == "always_lookup":
        return f"LOOKUP: {word}"
    if MODE["policy"] == "wrong":
        return "ANSWER: zzz"
    if MODE["policy"] == "gold_in_thinking":
        return f"<think>{task['answer']}</think>\nANSWER: zzz"
    if task["task"].startswith("lookup"):
        lookups = sum(1 for m in messages if m["role"] == "assistant")
        if lookups == 0:
            return "LOOKUP: " + question.split("Dakota word ")[1].split()[0].strip("?")
        if task["task"] == "lookup_root_gloss" and lookups == 1:
            match = re.search(r"\[derived from: ([^\],]+)", messages[-1]["content"])
            return "LOOKUP: " + (match.group(1) if match else "x")
    return "ANSWER: " + task["answer"]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        message = {"role": "assistant", "content": reply(body["messages"])}
        out = {
            "id": "x", "object": "chat.completion", "created": 0, "model": body.get("model", "m"),
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


async def main() -> bool:
    env = load_environment(max_train_per_task=2, max_eval_per_task=6)
    os.environ["E2E_KEY"] = "x"
    client = vf.ClientConfig(api_key_var="E2E_KEY", api_base_url=f"http://127.0.0.1:{PORT}/v1", max_retries=0)
    ok = True
    for policy in ("oracle", "wrong", "gold_in_thinking", "always_lookup"):
        MODE["policy"] = policy
        result = await env.evaluate(client=client, model="scripted", num_examples=-1, max_concurrent=8)
        outputs = result["outputs"] if isinstance(result, dict) else result.outputs
        by_task: dict[str, dict] = {}
        for out in outputs:
            get = out.get if isinstance(out, dict) else lambda key, o=out: getattr(o, key)
            metrics = get("metrics") or {}
            stats = by_task.setdefault(get("info")["task_type"], {"n": 0, "reward": 0.0, "lookups": 0.0})
            stats["n"] += 1
            stats["reward"] += get("reward")
            stats["lookups"] += metrics.get("lookups_used", 0)
        mean = {t: s["reward"] / s["n"] for t, s in by_task.items()}
        lookups = {t: s["lookups"] / s["n"] for t, s in by_task.items()}
        print(policy, {t: (round(mean[t], 2), lookups[t]) for t in sorted(mean)})
        if policy == "oracle":
            ok &= all(v > 0.999 for v in mean.values())
            ok &= lookups["lookup_gloss"] == 1 and lookups["lookup_root_gloss"] == 2
        elif policy in ("wrong", "gold_in_thinking"):
            ok &= all(v < 0.3 for v in mean.values())
        else:
            ok &= mean["lookup_gloss"] == 0 and lookups["lookup_gloss"] == 3
    return ok


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    passed = asyncio.run(main())
    print("E2E", "PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)
