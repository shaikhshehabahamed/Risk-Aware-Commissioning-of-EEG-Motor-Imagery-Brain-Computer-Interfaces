from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "local.json"


@dataclass(frozen=True)
class ProjectConfig:
    dataset_root: Path
    run_root: Path
    task_runs: tuple[int, ...]
    random_seed: int = 42


def _find_config_path(path: str | Path | None) -> Path:
    if path is None:
        return DEFAULT_CONFIG_PATH

    config_path = Path(path).expanduser()
    if config_path.is_absolute():
        return config_path.resolve()

    candidates = [
        Path.cwd() / config_path,
        PROJECT_ROOT / config_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _config_relative_base(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


def _resolve_config_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def load_config(path: str | Path | None = None) -> ProjectConfig:
    config_path = _find_config_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    base = _config_relative_base(config_path)
    dataset_root = _resolve_config_path(data["dataset_root"], base)
    run_root = _resolve_config_path(data["run_root"], base)
    task_runs = tuple(int(run) for run in data.get("task_runs", [4, 8, 12]))
    random_seed = int(data.get("random_seed", 42))

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    run_root.mkdir(parents=True, exist_ok=True)
    return ProjectConfig(
        dataset_root=dataset_root,
        run_root=run_root,
        task_runs=task_runs,
        random_seed=random_seed,
    )
