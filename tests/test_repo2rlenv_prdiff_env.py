from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = ROOT / "repo2rlenv" / "dakota1890-prdiff"
EXPECTED_TASKS = {
    "HarleyCoops__Dakota1890-15",
    "HarleyCoops__Dakota1890-16",
    "HarleyCoops__Dakota1890-18",
}


def test_repo2rlenv_prdiff_tasks_are_present_and_harbor_shaped():
    task_dirs = {path.name for path in ENV_ROOT.iterdir() if path.is_dir()}
    assert task_dirs == EXPECTED_TASKS

    for task_id in EXPECTED_TASKS:
        task_dir = ENV_ROOT / task_id
        for relative in (
            "task.toml",
            "instruction.md",
            "solution/patch.diff",
            "solution/solve.sh",
            "environment/Dockerfile",
            "tests/test.sh",
        ):
            assert (task_dir / relative).is_file(), f"missing {task_id}/{relative}"

        with (task_dir / "task.toml").open("rb") as handle:
            metadata = tomllib.load(handle)["metadata"]["repo2env"]

        assert metadata["pipeline"] == "pr_diff"
        assert metadata["repo"] == "HarleyCoops/Dakota1890"
        assert metadata["reward_kinds"] == ["diff_similarity"]
        assert metadata["reference"].startswith("https://github.com/HarleyCoops/Dakota1890/pull/")
        assert (task_dir / "instruction.md").read_text(encoding="utf-8").strip()
        assert (task_dir / "solution" / "patch.diff").read_text(encoding="utf-8").startswith("diff --git")
