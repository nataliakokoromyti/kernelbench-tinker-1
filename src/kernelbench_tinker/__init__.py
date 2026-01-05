
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from kernelbench_tinker.envs.kernelbench_env import KernelBenchDatasetBuilder
from kernelbench_tinker.training.loop import TrainingConfig

try:
    __version__ = version("kernelbench-tinker")
except PackageNotFoundError:  # pragma: no cover - fallback for editable/dev use
    __version__ = "0.1.0"

__all__ = [
    "KernelBenchDatasetBuilder",
    "TrainingConfig",
    "__version__",
]
