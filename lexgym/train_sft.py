"""Stage 1: supervised warm start on Tinker. Teaches the lexicon before RL.

    python -m lexgym.train_sft --model-name <tinker model> --max-steps 200
"""
from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from .tinker_env import pick_renderer_name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--sft-file", default="data/lexgym_tasks/sft_train.jsonl")
    ap.add_argument("--log-path", default="lexgym_runs/sft")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--num-epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--wandb-project", default="riggs1890-lexgym")
    ap.add_argument("--wandb-name", default=None)
    a = ap.parse_args()
    load_dotenv()
    renderer_name = pick_renderer_name(a.model_name)
    common = ChatDatasetBuilderCommonConfig(model_name_for_tokenizer=a.model_name, renderer_name=renderer_name,
                                            max_length=1024, batch_size=a.batch_size,
                                            train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)
    cfg = train.Config(
        log_path=a.log_path, model_name=a.model_name, recipe_name="lexgym_riggs1890_sft", renderer_name=renderer_name,
        dataset_builder=FromConversationFileBuilder(common_config=common, file_path=a.sft_file, test_size=256),
        learning_rate=a.learning_rate, lora_rank=a.lora_rank, num_epochs=a.num_epochs, max_steps=a.max_steps,
        save_every=a.save_every, eval_every=25, wandb_project=a.wandb_project or None, wandb_name=a.wandb_name,
    )
    asyncio.run(train.main(cfg))


if __name__ == "__main__":
    main()
