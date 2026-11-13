"""Know It or Look It Up: live demo of Inkling 2 Small fine-tuned on Riggs' Dakota-English Dictionary (1890).

FastAPI app for a Docker Space. It serves one page and three JSON routes. Live answers come from the
Tinker OpenAI-compatible endpoint. The key is read from the TINKER_API_KEY secret and is never sent to
the page. Request limits protect the account; recorded evaluation traces are the fallback.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from openai import AsyncOpenAI

import engine

HERE = Path(__file__).parent
TUNED = os.environ.get("LEXGYM_TUNED", "tinker://551aa04a-f385-5fdc-825c-7f47cf3a769d:train:0/sampler_weights/final")
BASE = os.environ.get("LEXGYM_BASE", "thinkingmachines/Inkling-2-Small")
# One question makes two calls (tuned model, base model), so the limits count calls.
PER_VISITOR_PER_HOUR = int(os.environ.get("LEXGYM_VISITOR_LIMIT", "30"))
PER_DAY = int(os.environ.get("LEXGYM_DAY_LIMIT", "1200"))
BASE_MAX_TOKENS = int(os.environ.get("LEXGYM_BASE_MAX_TOKENS", "1500"))
WORD_OK = re.compile(r"^[^\W\d_][\w'’\- .]{0,39}$", re.UNICODE)
POS_OK = re.compile(r"^[a-z. ]{0,12}$")

index = engine.Index(Path(hf_hub_download("HarleyCooper/riggs1890-lexicon-graph", "tasks_jsonl/lookup_index.json",
                                          repo_type="dataset")))
examples = json.loads((HERE / "examples.json").read_text(encoding="utf-8"))
client = AsyncOpenAI(base_url=engine.BASE_URL, api_key=os.environ["TINKER_API_KEY"], timeout=120.0,
                     max_retries=1) if os.environ.get("TINKER_API_KEY") else None
visitor_calls: dict[str, deque] = defaultdict(deque)
day = {"date": "", "count": 0}
cache: dict[tuple, dict] = {}

app = FastAPI(title="Know It or Look It Up", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


def visitor(request: Request) -> str:
    return (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()


def allow(who: str) -> str | None:
    """Return None when the request can run, or the reason it cannot."""
    now, today = time.time(), time.strftime("%Y-%m-%d", time.gmtime())
    if day["date"] != today:
        day.update(date=today, count=0)
    if day["count"] >= PER_DAY:
        return "The demo reached its limit of live questions for today. The recorded examples still work."
    calls = visitor_calls[who]
    while calls and now - calls[0] > 3600:
        calls.popleft()
    if len(calls) >= PER_VISITOR_PER_HOUR:
        return f"You reached the limit of {PER_VISITOR_PER_HOUR // 2} live questions per hour. The recorded examples still work."
    calls.append(now)
    day["count"] += 1
    return None


@app.get("/")
async def home():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/status")
async def status():
    return {"live": client is not None, "tuned": TUNED.split("/")[-1], "base": BASE, "entries": len(index.entries),
            "visitor_limit": PER_VISITOR_PER_HOUR // 2, "left_today": max(0, PER_DAY - day["count"]) // 2}


@app.get("/api/examples")
async def get_examples():
    return examples


@app.post("/api/ask")
async def ask(request: Request):
    try:
        body = await request.json()
        task, word, pos = str(body.get("task", "")), str(body.get("word", "")).strip(), str(body.get("pos", "")).strip()
    except Exception:
        return JSONResponse({"error": "The request was not valid JSON."}, status_code=400)
    if task not in engine.PROMPTS:
        return JSONResponse({"error": "Unknown task."}, status_code=400)
    if not WORD_OK.match(word):
        return JSONResponse({"error": "Type one word of up to 40 letters. Use the keys below the box for Dakota letters."}, status_code=400)
    if task in engine.NEEDS_POS and not POS_OK.match(pos):
        return JSONResponse({"error": "Use a part-of-speech abbreviation such as n. or v. a."}, status_code=400)
    which = str(body.get("model", "tuned"))
    if which not in ("tuned", "base"):
        return JSONResponse({"error": "Unknown model."}, status_code=400)
    key = (which, task, engine.canon(word), pos if task in engine.NEEDS_POS else "")
    _, prompt = engine.build_prompt(task, word, pos)
    if key in cache:
        return {**cache[key], "cached": True}
    if client is None:
        return JSONResponse({"error": "Live answers are off because the Space has no Tinker key. The recorded examples still work."}, status_code=503)
    reason = allow(visitor(request))
    if reason:
        return JSONResponse({"error": reason}, status_code=429)
    try:
        if which == "tuned":
            answer = await engine.run_model(client, TUNED, task, word, pos, index, max_tokens=96)
        else:
            answer = await engine.run_model(client, BASE, task, word, pos, index, max_tokens=BASE_MAX_TOKENS)
            last = answer["steps"][-1]
            if not last.get("text"):
                last["text"] = f"No answer. The base model used all {BASE_MAX_TOKENS:,} tokens of its limit on reasoning."
                last["gave_up"] = True
    except Exception as e:
        return JSONResponse({"error": f"The Tinker endpoint did not answer ({type(e).__name__}). The checkpoint may have "
                                      "expired or the service may be busy. The recorded examples still work."}, status_code=502)
    result = {"task": task, "word": word, "pos": pos, "prompt": prompt, "model": which,
              "entry": engine.book_entries(index, task, word), "answer": answer, "live": True}
    if len(cache) < 4000:
        cache[key] = result
    return result
