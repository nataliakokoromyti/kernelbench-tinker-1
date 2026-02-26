"""Reward shaping for KernelBench RL training."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# Import static checker with fallback for development
try:
    import kernelbench.kernel_static_checker as static_checker
except ImportError:
    # Fallback: add KernelBench/src to path if package not installed
    project_root = Path(__file__).parent.parent.parent.parent.parent
    kernelbench_src = project_root / "KernelBench" / "src"
    if kernelbench_src.exists() and str(kernelbench_src) not in sys.path:
        sys.path.insert(0, str(kernelbench_src))
    import kernelbench.kernel_static_checker as static_checker

if TYPE_CHECKING:
    from kernelbench_tinker.envs.kernelbench_client import KernelEvalResult

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for reward computation.

    Default: S = 0.3 * correct + speedup (if correct).
    """

    # Reward weights
    format_weight: float = 0.0
    compile_weight: float = 0.0
    correctness_weight: float = 0.3
    speed_weight: float = 1.0
    length_weight: float = 0.0

    format_penalty: float = 0.0

    # Speed reward
    speed_baseline: float = 0.0
    speed_scale: float = 1.0
    speed_max_reward: float = 10.0

    # Length reward (active when length_weight > 0)
    length_max: int = 8000
    length_min: int = 500

    # Static checker for reward hacking detection
    enable_static_checker: bool = True
    static_checker_backend: str = "triton"
    static_checker_precision: str = "fp32"
    static_checker_strict: list[str] | None = None
    static_checker_warnings: list[str] | None = None

    # Reward clipping
    reward_clip_min: float | None = None
    reward_clip_max: float | None = None


def format_reward(eval_result: "KernelEvalResult", config: RewardConfig) -> float:
    """1.0 if format is valid, config.format_penalty otherwise."""
    if eval_result["format_ok"]:
        return 1.0
    return config.format_penalty


def compile_reward(eval_result: "KernelEvalResult", config: RewardConfig) -> float:
    """1.0 if compiled successfully, 0.0 otherwise."""
    return 1.0 if eval_result["compiled"] else 0.0


def correctness_reward(eval_result: "KernelEvalResult") -> float:
    """Return 1.0 if all tests pass, 0.0 otherwise."""
    if not eval_result["compiled"]:
        return 0.0
    if eval_result["tests_total"] == 0:
        return 0.0
    return 1.0 if eval_result["tests_passed"] == eval_result["tests_total"] else 0.0


def speed_reward(eval_result: "KernelEvalResult", config: RewardConfig) -> float:
    """Compute linear speedup reward: T_baseline / T_kernel.

    Only gives reward for fully correct kernels with available speedup data.
    """
    if not eval_result["correctness"]:
        return 0.0

    speedup = eval_result.get("speedup")
    if speedup is None or speedup <= 0:
        return 0.0

    if speedup <= config.speed_baseline:
        return 0.0

    reward = config.speed_scale * (speedup - config.speed_baseline)
    return min(reward, config.speed_max_reward)


def length_reward(eval_result: "KernelEvalResult", config: RewardConfig) -> float:
    """Linear interpolation: 1.0 at length_min, 0.0 at length_max."""
    code_length = eval_result.get("code_length", 0)

    if code_length <= config.length_min:
        return 1.0
    elif code_length >= config.length_max:
        return 0.0
    else:
        range_size = config.length_max - config.length_min
        position = code_length - config.length_min
        return 1.0 - (position / range_size)


def check_reward_hacking(
    kernel_code: str,
    config: RewardConfig,
    backend: str | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Check for reward hacking patterns using static checker.

    Returns (has_errors, errors, warnings).
    """
    if not config.enable_static_checker:
        return (False, [], [])

    backend = backend or config.static_checker_backend

    strict_checks = config.static_checker_strict
    if strict_checks is None:
        strict_checks = list(static_checker.STRICT_CHECKS)
        if backend in static_checker.BACKEND_IMPL_CHECK:
            backend_check = static_checker.BACKEND_IMPL_CHECK[backend]
            if backend_check not in strict_checks:
                strict_checks.append(backend_check)
        for check in ("pytorch_wrap", "torch_computation_ops"):
            if check not in strict_checks:
                strict_checks.append(check)

    warning_checks = config.static_checker_warnings
    if warning_checks is None:
        warning_checks = [
            w for w in static_checker.WARNING_CHECKS
            if w not in strict_checks
        ]

    valid, errors, warnings = static_checker.validate_kernel_static(
        code=kernel_code,
        backend=backend,
        precision=config.static_checker_precision,
        forbidden=strict_checks,
        warnings=warning_checks,
    )

    return (not valid, errors, warnings)


def compute_reward(
    eval_result: "KernelEvalResult",
    config: RewardConfig | None = None,
    kernel_code: str | None = None,
    backend: str | None = None,
) -> float:
    """Compute the total reward for a kernel evaluation.

    Formula: S = correctness_weight * correct + speed_weight * speedup.
    Zero reward for bad format, reward hacking, or incorrect kernels.
    """
    if config is None:
        config = RewardConfig()

    if not eval_result["format_ok"]:
        return 0.0

    # Static checker
    if config.enable_static_checker and kernel_code:
        has_errors, errors, warnings = check_reward_hacking(
            kernel_code=kernel_code,
            config=config,
            backend=backend,
        )
        for warning in warnings:
            logger.warning(f"Static checker warning: {warning}")
        if has_errors:
            for error in errors:
                logger.error(f"Reward hacking detected (reward set to 0): {error}")
            return 0.0

    # Correctness (binary)
    corr_reward = correctness_reward(eval_result)
    if corr_reward == 0.0:
        return 0.0

    total = config.correctness_weight * corr_reward

    # Speedup reward (only for correct kernels)
    if config.speed_weight > 0:
        total += config.speed_weight * speed_reward(eval_result, config)

    # Optional: format/compile/length rewards (disabled by default)
    if config.format_weight > 0:
        total += config.format_weight * format_reward(eval_result, config)
    if config.compile_weight > 0:
        total += config.compile_weight * compile_reward(eval_result, config)
    if config.length_weight > 0:
        total += config.length_weight * length_reward(eval_result, config)

    # Clip
    if config.reward_clip_min is not None:
        total = max(total, config.reward_clip_min)
    if config.reward_clip_max is not None:
        total = min(total, config.reward_clip_max)

    return total


def compute_reward_breakdown(
    eval_result: "KernelEvalResult",
    config: RewardConfig | None = None,
    kernel_code: str | None = None,
    backend: str | None = None,
) -> dict[str, float | list[str]]:
    """Compute individual reward components for logging."""
    if config is None:
        config = RewardConfig()

    static_checker_errors: list[str] = []
    static_checker_warnings: list[str] = []
    if config.enable_static_checker and kernel_code:
        _, static_checker_errors, static_checker_warnings = check_reward_hacking(
            kernel_code=kernel_code,
            config=config,
            backend=backend,
        )

    return {
        "reward_format": format_reward(eval_result, config),
        "reward_compile": compile_reward(eval_result, config),
        "reward_correctness": correctness_reward(eval_result),
        "reward_speed": speed_reward(eval_result, config),
        "reward_length": length_reward(eval_result, config),
        "reward_total": compute_reward(eval_result, config, kernel_code=kernel_code, backend=backend),
        "static_checker_errors": static_checker_errors,
        "static_checker_warnings": static_checker_warnings,
    }


def compute_discounted_returns(
    step_scores: list[float],
    gamma: float = 0.4,
    aggregation: str = "sum",
) -> list[float]:
    """Compute discounted returns for multi-turn RL.

    sum: R_t = S_t + gamma * R_{t+1}  (backward recursion)
    max: R_t = max{ gamma^(i-t) * S_i }
    """
    if aggregation not in ("sum", "max"):
        raise ValueError(f"Unknown aggregation mode: {aggregation!r}. Must be 'sum' or 'max'.")

    if not step_scores:
        return []

    T = len(step_scores)

    if aggregation == "sum":
        returns = [0.0] * T
        returns[-1] = step_scores[-1]
        for t in range(T - 2, -1, -1):
            returns[t] = step_scores[t] + gamma * returns[t + 1]
        return returns

    # aggregation == "max"
    return [
        max(gamma ** (i - t) * step_scores[i] for i in range(t, T))
        for t in range(T)
    ]
