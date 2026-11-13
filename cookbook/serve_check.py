"""Sample from a Tinker sampler checkpoint through the OpenAI-compatible endpoint.

    python serve_check.py "Give the English definition of the Dakota word: iwiśtaŋ"

Reads TINKER_API_KEY from the environment or a local .env file. Set LEXGYM_MODEL to use a different checkpoint.
"""
from __future__ import annotations

import os
import sys

from openai import OpenAI

BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
MODEL = os.environ.get("LEXGYM_MODEL", "tinker://551aa04a-f385-5fdc-825c-7f47cf3a769d:train:0/sampler_weights/final")
SYSTEM = ("You answer from Stephen R. Riggs, A Dakota-English Dictionary (1890). Use the spelling of that book. "
          "Give only the answer on one line, starting with 'ANSWER:'.")


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    sys.stdout.reconfigure(encoding="utf-8")
    client = OpenAI(base_url=BASE_URL, api_key=os.environ["TINKER_API_KEY"])
    question = sys.argv[1] if len(sys.argv) > 1 else "Give the English definition of the Dakota word: iwiśtaŋ"
    out = client.chat.completions.create(
        model=MODEL, max_tokens=96, temperature=0.0, reasoning_effort="none",
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}])
    print(out.choices[0].message.content)
    print(out.usage)


if __name__ == "__main__":
    main()
