from __future__ import annotations

import numpy as np

from rabp_bmi.features import MOTOR_CHANNELS


def cohen_abs_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {x.shape}")
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"Expected binary labels, got {classes}")

    x0 = x[y == classes[0]]
    x1 = x[y == classes[1]]
    mean_diff = np.abs(x1.mean(axis=0) - x0.mean(axis=0))
    pooled = np.sqrt(0.5 * (x0.var(axis=0, ddof=1) + x1.var(axis=0, ddof=1)))
    return mean_diff / (pooled + 1e-12)


def motor_candidate_indices(channel_names: tuple[str, ...]) -> list[int]:
    return [idx for idx, name in enumerate(channel_names) if name in MOTOR_CHANNELS]


def central_parietal_candidate_indices(channel_names: tuple[str, ...]) -> list[int]:
    prefixes = ("FC", "C", "CP", "P")
    return [
        idx
        for idx, name in enumerate(channel_names)
        if name.startswith(prefixes) and not name.startswith("PO")
    ]


def select_channel_indices(
    x_train: np.ndarray,
    y_train: np.ndarray,
    channel_names: tuple[str, ...],
    policy: str,
    budget: int,
) -> list[int]:
    if policy == "all":
        return list(range(len(channel_names)))

    if policy == "fixed_motor":
        indices = motor_candidate_indices(channel_names)
        if not indices:
            raise ValueError("No motor channels found")
        return indices[:budget]

    if policy == "topk_effect":
        candidates = list(range(len(channel_names)))
    elif policy == "motor_topk_effect":
        candidates = motor_candidate_indices(channel_names)
    elif policy == "central_parietal_topk_effect":
        candidates = central_parietal_candidate_indices(channel_names)
    else:
        raise ValueError(f"Unknown channel policy: {policy}")

    if not candidates:
        raise ValueError(f"No candidates for channel policy: {policy}")

    scores = cohen_abs_scores(x_train[:, candidates], y_train)
    order = np.argsort(scores)[::-1]
    selected = [candidates[int(idx)] for idx in order[: min(budget, len(candidates))]]
    return selected
