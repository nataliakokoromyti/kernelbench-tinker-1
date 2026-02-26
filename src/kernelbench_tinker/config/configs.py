"""
Configuration dataclasses for KernelBench-Tinker integration.

This module centralizes all configuration to avoid duplication across classes.
Configs are designed to be loaded from YAML and passed through the system.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalConfig:
    """
    Configuration for kernel evaluation.

    This config is passed to Modal for kernel evaluation and controls
    how correctness and performance are measured.
    """

    # Correctness testing
    num_correct_trials: int = 5

    # Performance measurement
    measure_performance: bool = False
    num_perf_trials: int = 100
    timing_method: str = "cuda_event"

    # Precision
    precision: str = "fp32"

    # Speedup validation
    check_for_excessive_speedup: bool = True
    excessive_speedup_threshold: float = 10.0

    # Modal configuration
    modal_gpu_type: str = "A100"
    modal_timeout: float = 120.0


@dataclass
class MultiTurnConfig:
    """
    Configuration for multi-turn RL training.

    Controls the iterative refinement loop where the model receives
    evaluation feedback and can fix errors across multiple turns.
    """

    # Enable multi-turn mode (False = single-turn)
    enabled: bool = False

    # Refinement turns per trajectory
    n: int = 4

    # Discount factor for multi-turn returns: R_t = S_t + gamma * R_{t+1}
    gamma: float = 0.4

    # Return aggregation mode: "sum" or "max"
    #   sum: R_t = Σ γ^(i-t) × S_i  (reward turns leading to many good kernels)
    #   max: R_t = max{ γ^(i-t) × S_i } (reward turns leading to one great kernel)
    aggregation: str = "sum"

    # Stop the episode early when the kernel is correct.
    # Default False for training: model needs post-correctness turns to
    # learn speedup optimization.  Set True at eval time if desired.
    early_stop_on_correct: bool = False

    # Optional: require this speedup before early stopping
    speedup_threshold: float | None = None
