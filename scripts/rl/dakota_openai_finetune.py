"""
Dakota OpenAI Fine-Tuning Script
Adapted from Stoney Nakoda openai_finetune.py pattern.

Handles fine-tuning Dakota language models via OpenAI API with optional
HuggingFace dataset publishing and Weights & Biases tracking.
"""

import io
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI
from dotenv import load_dotenv

try:
    from scripts.rl.dakota_openai_config import DakotaOpenAIConfig, load_dakota_openai_config
except ModuleNotFoundError:  # Allow direct script execution from the repo root.
    from dakota_openai_config import DakotaOpenAIConfig, load_dakota_openai_config

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    import wandb
except ImportError:
    wandb = None

# Set up logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

STATUS_TO_INDEX: Dict[str, int] = {
    "validating_files": -1,
    "queued": 0,
    "running": 1,
    "succeeded": 2,
    "failed": -2,
    "cancelled": -3,
    "expired": -4,
}

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_openai_object(obj: Any) -> Any:
    """Convert OpenAI SDK objects to plain JSON-serializable structures."""

    if obj is None:
        return None
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_serialize_openai_object(item) for item in obj]
    if isinstance(obj, tuple):
        return [_serialize_openai_object(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): _serialize_openai_object(value) for key, value in obj.items()}
    return str(obj)


def _sanitize_error_text(error: Any) -> str:
    """Remove API-key shaped strings from persisted error messages."""

    text = str(error)
    return re.sub(r"sk-[A-Za-z0-9_*.-]+", "sk-REDACTED", text)


def _has_plausible_openai_key_shape(api_key: Optional[str]) -> bool:
    """Return whether a value looks like an OpenAI API key without validating it remotely."""

    if not api_key:
        return False
    return api_key.startswith("sk-") and len(api_key) >= 40


@dataclass
class BaselineRunLedger:
    """Durable ledger for a Dakota OpenAI SFT baseline launch."""

    experiment: str
    base_model: str
    epochs: int
    train_file: str
    valid_file: str
    train_examples: int
    valid_examples: int
    train_token_estimate: int
    valid_token_estimate: int
    estimated_training_tokens: int
    training_file_id: Optional[str] = None
    validation_file_id: Optional[str] = None
    job_id: Optional[str] = None
    status: Optional[str] = None
    fine_tuned_model: Optional[str] = None
    result_files: list[Any] = field(default_factory=list)
    error: Optional[Any] = None
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    final_job: Optional[Any] = None

    def write(self, path: Path) -> None:
        """Write the ledger as JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _utc_now()
        path.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

class DakotaOpenAIFineTuner:
    def __init__(self, require_api_key: bool = True):
        """Initialize file paths immediately and the OpenAI client only when needed."""
        # Load environment variables
        load_dotenv(override=True)
        self.api_key = os.getenv('OPENAI_API_KEY')
        if require_api_key and not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        if require_api_key and not _has_plausible_openai_key_shape(self.api_key):
            raise ValueError(
                "OPENAI_API_KEY is present but does not look like a complete OpenAI API key. "
                "Check C:\\Users\\chris\\Dakota1890\\.env."
            )

        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        if self.client is not None:
            logger.info("OpenAI client initialized successfully")
        else:
            logger.info("OPENAI_API_KEY not set; running in file-readiness mode only")

        # Define file paths - look in project root, not script directory
        # Script is in scripts/rl/, but files are in root OpenAIFineTune/
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent  # Go up from scripts/rl/ to project root
        self.train_file = project_root / "OpenAIFineTune" / "dakota_train.jsonl"
        self.valid_file = project_root / "OpenAIFineTune" / "dakota_valid.jsonl"

        # Ensure files exist
        if not self.train_file.exists():
            raise FileNotFoundError(f"Training file not found: {self.train_file}")
        if not self.valid_file.exists():
            raise FileNotFoundError(f"Validation file not found: {self.valid_file}")

        logger.info(f"Found training file: {self.train_file}")
        logger.info(f"Found validation file: {self.valid_file}")
        
        self.config: DakotaOpenAIConfig = load_dakota_openai_config()
        logger.info("\n%s", self.config.diagnostics())

        self.fine_tune_model = self.config.openai_finetune_model
        logger.info("Using fine-tune base model: %s", self.fine_tune_model)

        # Hugging Face dataset publishing (optional)
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.hf_repo_id = os.getenv("HUGGINGFACE_DATASET_REPO")
        private_flag = os.getenv("HUGGINGFACE_DATASET_PRIVATE", "false").strip().lower()
        self.hf_private = private_flag in {"1", "true", "yes", "y", "on"}
        self.hf_api: Optional[HfApi] = None
        if self.hf_token and self.hf_repo_id:
            if HfApi is None:
                raise ImportError(
                    "huggingface_hub is required for dataset publishing but is not installed."
                )
            self.hf_api = HfApi(token=self.hf_token)
            logger.info("Hugging Face dataset publishing enabled for repo %s", self.hf_repo_id)
        else:
            logger.info(
                "Hugging Face dataset publishing disabled (set HUGGINGFACE_TOKEN and "
                "HUGGINGFACE_DATASET_REPO to enable)."
            )

        # Weights & Biases experiment tracking (optional)
        self.wandb_api_key = os.getenv("WANDB_API_KEY")
        self.wandb_project = os.getenv("WANDB_PROJECT")
        self.wandb_entity = os.getenv("WANDB_ENTITY") or None
        self.wandb_run_name = os.getenv("WANDB_RUN_NAME")
        self.wandb_enabled = bool(self.wandb_api_key and self.wandb_project)
        self.wandb_run = None

    @property
    def default_ledger_path(self) -> Path:
        """Return the default run-ledger path for this launch."""

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return self.train_file.parent / "runs" / f"dakota-openai-sft-{timestamp}.json"

    def readiness_report(self) -> Dict[str, Any]:
        """Return a non-mutating readiness report for the SFT baseline."""
        train_examples = self._count_lines(self.train_file)
        valid_examples = self._count_lines(self.valid_file)
        train_token_estimate = self._estimate_chat_tokens(self.train_file)
        valid_token_estimate = self._estimate_chat_tokens(self.valid_file)
        epochs = self.config.openai_finetune_epochs
        return {
            "train_file": str(self.train_file),
            "valid_file": str(self.valid_file),
            "train_exists": self.train_file.exists(),
            "valid_exists": self.valid_file.exists(),
            "train_examples": train_examples,
            "valid_examples": valid_examples,
            "base_model": self.fine_tune_model,
            "epochs": epochs,
            "train_token_estimate": train_token_estimate,
            "valid_token_estimate": valid_token_estimate,
            "combined_token_estimate": train_token_estimate + valid_token_estimate,
            "estimated_training_tokens": train_token_estimate * epochs,
            "token_estimator": "tiktoken" if tiktoken is not None else "unavailable",
            "openai_api_key_present": bool(self.api_key),
            "openai_api_key_shape_valid": _has_plausible_openai_key_shape(self.api_key),
            "hf_publish_enabled": bool(self.hf_api),
            "wandb_enabled": self.wandb_enabled,
        }

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging if configured."""
        if not self.wandb_enabled:
            return
        if wandb is None:
            raise ImportError("wandb is required for experiment tracking but is not installed.")

        try:
            wandb.login(key=self.wandb_api_key, relogin=True)
            run_name = self.wandb_run_name or f"dakota-finetune-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=run_name,
                job_type="openai-fine-tune",
                config={
                    "base_model": self.fine_tune_model,
                    "train_file": str(self.train_file),
                    "valid_file": str(self.valid_file),
                },
            )
            if self.wandb_run:
                logger.info("Weights & Biases run started: %s", self.wandb_run.url)
        except Exception as exc:
            logger.warning("Unable to initialize Weights & Biases logging: %s", exc)
            self.wandb_run = None
            self.wandb_enabled = False

    def _wandb_log(self, metrics: Dict[str, Any]) -> None:
        """Safely log metrics to Weights & Biases."""
        if not self.wandb_run:
            return
        try:
            wandb.log(metrics)
        except Exception as exc:
            logger.warning("Failed to log metrics to Weights & Biases: %s", exc)

    def _finish_wandb(self) -> None:
        """Close the active Weights & Biases run."""
        if not self.wandb_run:
            return
        try:
            self.wandb_run.finish()
        except Exception as exc:
            logger.warning("Failed to finalize Weights & Biases run: %s", exc)
        finally:
            self.wandb_run = None

    @staticmethod
    def _count_lines(file_path: Path) -> int:
        """Count the number of lines in a JSONL file."""
        with file_path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    def _estimate_chat_tokens(self, file_path: Path) -> int:
        """Estimate training tokens for chat-format JSONL examples."""
        if tiktoken is None:
            return 0

        try:
            encoding = tiktoken.encoding_for_model(self.fine_tune_model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        total_tokens = 0
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                example = json.loads(line)
                total_tokens += 3
                for message in example.get("messages", []):
                    total_tokens += 4
                    total_tokens += len(encoding.encode(message.get("content", "")))
                    if "name" in message:
                        total_tokens += 1
                total_tokens += 3
        return total_tokens

    def _build_dataset_card(self, train_count: int, valid_count: int, timestamp: str) -> str:
        """Draft a simple dataset card for the Hugging Face repo."""
        return (
            "# Dakota Fine-Tuning Dataset\n\n"
            f"- Updated: {timestamp} UTC\n"
            f"- Training examples: {train_count}\n"
            f"- Validation examples: {valid_count}\n\n"
            "This dataset is produced by the automated Dakota dictionary-to-fine-tuning "
            "pipeline. It packages conversational training examples used to fine-tune OpenAI "
            "models on Dakota language tasks, with data extracted from the 1890 Dakota-English Dictionary.\n"
        )

    def _publish_dataset_to_hf(self) -> Optional[Dict[str, int]]:
        """Upload the latest training artifacts to Hugging Face Datasets."""
        if not self.hf_api or not self.hf_repo_id:
            return None

        train_count = self._count_lines(self.train_file)
        valid_count = self._count_lines(self.valid_file)
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        commit_message = f"Update dataset via automated Dakota pipeline ({timestamp} UTC)"

        try:
            self.hf_api.create_repo(
                repo_id=self.hf_repo_id,
                repo_type="dataset",
                private=self.hf_private,
                exist_ok=True,
            )

            uploads = [
                (self.train_file, f"data/{self.train_file.name}"),
                (self.valid_file, f"data/{self.valid_file.name}"),
            ]
            for local_path, remote_path in uploads:
                logger.info(
                    "Uploading %s to Hugging Face dataset repo %s",
                    remote_path,
                    self.hf_repo_id,
                )
                self.hf_api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=remote_path,
                    repo_id=self.hf_repo_id,
                    repo_type="dataset",
                    token=self.hf_token,
                    commit_message=commit_message,
                )

            dataset_card = self._build_dataset_card(train_count, valid_count, timestamp)
            self.hf_api.upload_file(
                path_or_fileobj=io.BytesIO(dataset_card.encode("utf-8")),
                path_in_repo="README.md",
                repo_id=self.hf_repo_id,
                repo_type="dataset",
                token=self.hf_token,
                commit_message=commit_message,
            )

            logger.info("Dataset published to Hugging Face repo %s", self.hf_repo_id)
            return {"train_examples": train_count, "valid_examples": valid_count}
        except Exception as exc:
            logger.error("Failed to publish dataset to Hugging Face: %s", exc)
            raise

    def upload_file(self, file_path: Path, purpose: str) -> str:
        """Upload a file to OpenAI and return its file ID."""
        if self.client is None:
            raise RuntimeError("OpenAI client is unavailable. Set OPENAI_API_KEY before uploading files.")
        file_path = Path(file_path)
        logger.info("Uploading %s file: %s", purpose, file_path)

        with file_path.open("rb") as file:
            response = self.client.files.create(
                file=file,
                purpose=purpose
            )

        logger.info("Successfully uploaded %s file. File ID: %s", purpose, response.id)
        return response.id

    def create_fine_tuning_job(self, training_file_id: str, validation_file_id: str) -> str:
        """Create a fine-tuning job and return its ID."""
        if self.client is None:
            raise RuntimeError("OpenAI client is unavailable. Set OPENAI_API_KEY before creating jobs.")
        logger.info("Creating fine-tuning job...")
        
        response = self.client.fine_tuning.jobs.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model=self.fine_tune_model,
            hyperparameters={
                "n_epochs": self.config.openai_finetune_epochs
            }
        )
        
        logger.info(f"Fine-tuning job created successfully. Job ID: {response.id}")
        return response.id

    def monitor_job_progress(self, job_id: str, check_interval: int = 60):
        """Monitor the progress of a fine-tuning job."""
        if self.client is None:
            raise RuntimeError("OpenAI client is unavailable. Set OPENAI_API_KEY before monitoring jobs.")
        logger.info("Starting to monitor fine-tuning job: %s", job_id)
        start_time = time.time()

        while True:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            status = job.status

            # Log detailed status information
            logger.info("Status: %s", status)
            if hasattr(job, 'trained_tokens'):
                logger.info("Trained tokens: %s", job.trained_tokens)
            if hasattr(job, 'training_accuracy'):
                logger.info("Training accuracy: %s", job.training_accuracy)
            if hasattr(job, 'validation_loss'):
                logger.info("Validation loss: %s", job.validation_loss)

            metrics: Dict[str, Any] = {
                "status_index": STATUS_TO_INDEX.get(status, -5),
                "elapsed_seconds": time.time() - start_time,
            }
            if getattr(job, "trained_tokens", None) is not None:
                metrics["trained_tokens"] = job.trained_tokens
            if getattr(job, "training_accuracy", None) is not None:
                metrics["training_accuracy"] = job.training_accuracy
            if getattr(job, "validation_loss", None) is not None:
                metrics["validation_loss"] = job.validation_loss
            self._wandb_log(metrics)

            if status == "succeeded":
                logger.info("Fine-tuning completed successfully!")
                logger.info("Fine-tuned model ID: %s", job.fine_tuned_model)
                if self.wandb_run:
                    self.wandb_run.summary["final_status"] = status
                    self.wandb_run.summary["fine_tuned_model"] = job.fine_tuned_model
                return job
            elif status == "failed":
                logger.error("Fine-tuning failed: %s", job.error)
                if self.wandb_run:
                    self.wandb_run.summary["final_status"] = status
                    self.wandb_run.summary["failure_reason"] = str(job.error)
                return job
            elif status in ["cancelled", "expired"]:
                logger.warning("Fine-tuning job %s", status)
                if self.wandb_run:
                    self.wandb_run.summary["final_status"] = status
                return job

            logger.info("Waiting %d seconds before next check...", check_interval)
            time.sleep(check_interval)

    def _build_ledger(self, experiment: str) -> BaselineRunLedger:
        """Build the initial run ledger from local readiness data."""

        report = self.readiness_report()
        return BaselineRunLedger(
            experiment=experiment,
            base_model=report["base_model"],
            epochs=report["epochs"],
            train_file=report["train_file"],
            valid_file=report["valid_file"],
            train_examples=report["train_examples"],
            valid_examples=report["valid_examples"],
            train_token_estimate=report["train_token_estimate"],
            valid_token_estimate=report["valid_token_estimate"],
            estimated_training_tokens=report["estimated_training_tokens"],
        )

    def run_fine_tuning(
        self,
        *,
        experiment: str = "dakota-openai-sft-baseline",
        ledger_path: Path | None = None,
        wait: bool = True,
        check_interval: int = 60,
        training_file_id: str | None = None,
        validation_file_id: str | None = None,
    ):
        """Run the complete fine-tuning process."""
        dataset_stats: Optional[Dict[str, int]] = None
        ledger_output = ledger_path or self.default_ledger_path
        ledger = self._build_ledger(experiment)
        ledger.write(ledger_output)
        logger.info("Run ledger: %s", ledger_output)
        try:
            if self.wandb_enabled and not self.wandb_run:
                self._init_wandb()
                self._wandb_log({"stage_index": -1, "stage": "initialization"})

            if self.hf_api:
                logger.info("Publishing dataset to Hugging Face before fine-tuning")
                dataset_stats = self._publish_dataset_to_hf()
                if dataset_stats:
                    if self.wandb_run:
                        self.wandb_run.summary["hf_dataset_repo"] = self.hf_repo_id
                        self.wandb_run.summary["train_examples"] = dataset_stats["train_examples"]
                        self.wandb_run.summary["valid_examples"] = dataset_stats["valid_examples"]
                    self._wandb_log(
                        {
                            "stage_index": 0,
                            "stage": "huggingface_dataset_publish",
                            "train_examples": dataset_stats["train_examples"],
                            "valid_examples": dataset_stats["valid_examples"],
                        }
                    )
            else:
                logger.info("Skipping Hugging Face dataset publishing step (disabled).")

            # Step 1: Upload files, or reuse file IDs from a previous failed launch.
            if training_file_id and validation_file_id:
                logger.info("Step 1/3: Reusing existing OpenAI fine-tune file IDs")
                self._wandb_log({"stage_index": 1, "stage": "reuse_files"})
                train_file_id = training_file_id
                valid_file_id = validation_file_id
            else:
                logger.info("Step 1/3: Uploading files to OpenAI")
                self._wandb_log({"stage_index": 1, "stage": "upload_files"})
                train_file_id = self.upload_file(self.train_file, "fine-tune")
                valid_file_id = self.upload_file(self.valid_file, "fine-tune")
            ledger.training_file_id = train_file_id
            ledger.validation_file_id = valid_file_id
            ledger.status = "files_uploaded"
            ledger.write(ledger_output)
            self._wandb_log(
                {
                    "uploaded_train_file": 0 if training_file_id else 1,
                    "uploaded_valid_file": 0 if validation_file_id else 1,
                }
            )
            
            # Step 2: Create fine-tuning job
            logger.info("Step 2/3: Creating fine-tuning job")
            job_id = self.create_fine_tuning_job(train_file_id, valid_file_id)
            ledger.job_id = job_id
            ledger.status = "submitted"
            ledger.write(ledger_output)
            if self.wandb_run:
                self.wandb_run.summary["openai_job_id"] = job_id
            self._wandb_log({"stage_index": 2, "stage": "create_job"})

            if not wait:
                logger.info("Launch-only requested; job submitted and ledger written.")
                return ledger
            
            # Step 3: Monitor progress
            logger.info("Step 3/3: Monitoring fine-tuning progress")
            self._wandb_log({"stage_index": 3, "stage": "monitor_progress"})
            final_job = self.monitor_job_progress(job_id, check_interval=check_interval)
            ledger.status = getattr(final_job, "status", None)
            ledger.fine_tuned_model = getattr(final_job, "fine_tuned_model", None)
            ledger.result_files = _serialize_openai_object(getattr(final_job, "result_files", [])) or []
            ledger.error = _serialize_openai_object(getattr(final_job, "error", None))
            ledger.final_job = _serialize_openai_object(final_job)
            ledger.write(ledger_output)
            if final_job and self.wandb_run:
                result_files = getattr(final_job, "result_files", None)
                if result_files:
                    self.wandb_run.summary["result_files"] = [
                        getattr(file_info, "id", file_info) for file_info in result_files
                    ]

            return final_job

        except Exception as e:
            sanitized_error = _sanitize_error_text(e)
            logger.error("Error during fine-tuning process: %s", sanitized_error)
            ledger.status = "exception"
            ledger.error = sanitized_error
            ledger.write(ledger_output)
            if self.wandb_run:
                self.wandb_run.summary["final_status"] = "exception"
                self.wandb_run.summary["failure_reason"] = sanitized_error
            self._wandb_log({"exception": 1})
            raise
        finally:
            self._finish_wandb()

def main() -> None:
    """Main function to run the fine-tuning process or readiness-only validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run or validate Dakota OpenAI fine-tuning assets.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate files and configuration without creating a fine-tuning job.",
    )
    parser.add_argument(
        "--launch-only",
        action="store_true",
        help="Upload files and create the fine-tuning job, then exit without waiting for completion.",
    )
    parser.add_argument(
        "--ledger",
        default=None,
        help="Path to write the run ledger JSON. Defaults to OpenAIFineTune/runs/<timestamp>.json.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
        help="Seconds between status checks when waiting for completion.",
    )
    parser.add_argument(
        "--training-file-id",
        default=None,
        help="Reuse an existing OpenAI training file ID instead of uploading dakota_train.jsonl.",
    )
    parser.add_argument(
        "--validation-file-id",
        default=None,
        help="Reuse an existing OpenAI validation file ID instead of uploading dakota_valid.jsonl.",
    )
    args = parser.parse_args()

    logger.info("=== Starting Dakota Language Model Fine-Tuning ===")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        tuner = DakotaOpenAIFineTuner(require_api_key=not args.check_only)
        if args.check_only:
            logger.info("Readiness report: %s", json.dumps(tuner.readiness_report(), indent=2))
            return

        final_job = tuner.run_fine_tuning(
            ledger_path=Path(args.ledger) if args.ledger else None,
            wait=not args.launch_only,
            check_interval=args.poll_seconds,
            training_file_id=args.training_file_id,
            validation_file_id=args.validation_file_id,
        )

        if isinstance(final_job, BaselineRunLedger):
            logger.info("=== Fine-Tuning Job Submitted ===")
            logger.info("OpenAI job ID: %s", final_job.job_id)
        elif final_job and hasattr(final_job, 'fine_tuned_model'):
            logger.info("=== Fine-Tuning Process Completed Successfully ===")
            logger.info(f"Fine-tuned model ID: {final_job.fine_tuned_model}")
        else:
            logger.warning("=== Fine-Tuning Process Completed with Issues ===")

        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        logger.error("Fatal error: %s", _sanitize_error_text(e))
        raise

if __name__ == "__main__":
    main()
