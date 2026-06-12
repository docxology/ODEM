from __future__ import annotations

from collections.abc import Mapping
import os

import numpy as np


def resolve_combo_slice(
    total: int,
    *,
    start_index: int | None = None,
    end_index: int | None = None,
    max_combos: int | None = None,
    use_slurm_env: bool = True,
    environ: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    if total <= 0:
        raise ValueError("total combinations must be positive")
    if max_combos is not None and int(max_combos) < 1:
        raise ValueError("max_combos must be at least 1")

    env = os.environ if environ is None else environ
    if start_index is None and end_index is None and use_slurm_env and "SLURM_ARRAY_TASK_ID" in env:
        start, end = _slurm_slice(total, env)
    else:
        start = 0 if start_index is None else int(start_index)
        end = total if end_index is None else int(end_index)

    if start < 0 or end < 0 or start > total or end > total:
        raise ValueError(f"combo slice bounds must be within [0, {total}]")
    if start >= end:
        raise ValueError("combo slice is empty")
    if max_combos is not None:
        end = min(end, start + int(max_combos))
    if start >= end:
        raise ValueError("combo slice is empty")
    return start, end


def _slurm_slice(total: int, env: Mapping[str, str]) -> tuple[int, int]:
    task_id = int(env["SLURM_ARRAY_TASK_ID"])
    task_count = int(env.get("SLURM_ARRAY_TASK_COUNT", "1"))
    task_min = int(env.get("SLURM_ARRAY_TASK_MIN", "0"))
    task_rank = task_id - task_min
    if task_count < 1:
        raise ValueError("SLURM_ARRAY_TASK_COUNT must be positive")

    splits = np.array_split(np.arange(total), task_count)
    if task_rank < 0 or task_rank >= len(splits) or len(splits[task_rank]) == 0:
        raise ValueError("SLURM array selection produced an empty combo slice")
    return int(splits[task_rank][0]), int(splits[task_rank][-1]) + 1
