"""CPU Gradio demo for the grant-clean Dakota1890 Tinker run.

This Space does not download or host Qwen3-30B. When TINKER_API_KEY is set it
calls the remote Tinker sampler. Without the key, the frozen holdout example
bank and gold answers still work.
"""

from __future__ import annotations

import os

import gradio as gr

try:
    from demo import (
        BASE_MODEL_NAME,
        DEFAULT_MAX_TOKENS,
        DEFAULT_TEMPERATURE,
        REPO_URL,
        WANDB_RUN_URL,
        example_label,
        format_example_card,
        live_status_message,
        load_examples,
        resolve_sampler_path,
        run_live_or_explain,
        tinker_key_configured,
    )
except ImportError:
    from huggingface_space.demo import (
        BASE_MODEL_NAME,
        DEFAULT_MAX_TOKENS,
        DEFAULT_TEMPERATURE,
        REPO_URL,
        WANDB_RUN_URL,
        example_label,
        format_example_card,
        live_status_message,
        load_examples,
        resolve_sampler_path,
        run_live_or_explain,
        tinker_key_configured,
    )

EXAMPLES = load_examples()
EXAMPLE_CHOICES = [example_label(row) for row in EXAMPLES]
EXAMPLES_BY_LABEL = {example_label(row): row for row in EXAMPLES}

INTRO = f"""
# Dakota1890 grant-clean demo

This is a public look at the **grant-clean** Dakota1890 scaffold — W&B run
[`cebp9acs`]({WANDB_RUN_URL}) on `{BASE_MODEL_NAME}` — not a fluent Dakota model.

The 1890 Riggs *Dakota-English Dictionary* grammar is a historical scaffold.
Modern Dakota speakers should correct these outputs. This Space is not a
replacement for speakers.

**What the run actually shows:** eval exact match rose on English-glossary
tasks. English→Dakota is still weak. Do not read a correct glossary item as
fluent generation.

This Space is **CPU-only**. It does not load 30B weights. Live generation is
optional and, when enabled, calls the existing remote Tinker sampler.

- W&B: [{WANDB_RUN_URL}]({WANDB_RUN_URL})
- Code: [{REPO_URL}]({REPO_URL})
- Prior 0.6B Prime Intellect demo (unchanged): [HarleyCooper/Dakota-.6B](https://huggingface.co/spaces/HarleyCooper/Dakota-.6B)
"""


def _row_from_label(label: str) -> dict:
    return EXAMPLES_BY_LABEL[label]


def show_example(label: str) -> tuple[str, str, str, str]:
    row = _row_from_label(label)
    return (
        str(row.get("prompt") or ""),
        str(row.get("answer") or ""),
        format_example_card(row),
        live_status_message(),
    )


def infer(label: str, prompt: str, max_tokens: int, temperature: float) -> tuple[str, str, str]:
    row = EXAMPLES_BY_LABEL.get(label) or {}
    gold = str(row.get("answer") or "")
    # If the user edited away from the selected example, do not pretend gold applies.
    if prompt.strip() != str(row.get("prompt") or "").strip():
        gold = ""
    return run_live_or_explain(
        prompt,
        gold,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
    )


def build_demo() -> gr.Blocks:
    first = EXAMPLE_CHOICES[0]
    first_row = _row_from_label(first)
    with gr.Blocks(title="Dakota1890 Grant-Clean") as demo:
        gr.Markdown(INTRO)
        status = gr.Markdown(live_status_message())
        with gr.Accordion("Sampler and secrets", open=False):
            gr.Markdown(
                f"""
Default sampler URI (grant-clean session, `sampler_weights/final` convention):

`{resolve_sampler_path()}`

TINKER_API_KEY configured: **{"yes" if tinker_key_configured() else "no"}**

Override the URI with the Space secret `TINKER_SAMPLER_PATH` if the live
checkpoint path differs. This is not the later 35B `owf98569` adapter and not
the old 0.6B local-GPU demo.
"""
            )

        label = gr.Dropdown(
            choices=EXAMPLE_CHOICES,
            value=first,
            label="Holdout example (cebp9acs eval set)",
        )
        meta = gr.Markdown(format_example_card(first_row))
        prompt = gr.Textbox(
            value=str(first_row.get("prompt") or ""),
            lines=10,
            label="Prompt",
        )
        gold = gr.Textbox(
            value=str(first_row.get("answer") or ""),
            lines=2,
            label="Gold (frozen holdout v1)",
            interactive=False,
        )
        with gr.Row():
            max_tokens = gr.Slider(
                16,
                128,
                value=DEFAULT_MAX_TOKENS,
                step=8,
                label="Max tokens (live only)",
            )
            temperature = gr.Slider(
                0.0,
                0.7,
                value=DEFAULT_TEMPERATURE,
                step=0.05,
                label="Temperature (live only; default 0)",
            )
        run = gr.Button("Run live Tinker sampler", variant="primary")
        extracted = gr.Textbox(lines=2, label="Extracted answer (boxed / last line)")
        raw = gr.Textbox(lines=6, label="Raw sampler output / status")
        comparison = gr.Textbox(lines=4, label="Gold comparison")

        label.change(
            fn=show_example,
            inputs=[label],
            outputs=[prompt, gold, meta, status],
        )
        run.click(
            fn=infer,
            inputs=[label, prompt, max_tokens, temperature],
            outputs=[extracted, raw, comparison],
        )
    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
