"""
Tinker RL Environment for KernelBench.

This module implements the Env, EnvGroupBuilder, and RLDataset interfaces
from tinker_cookbook.rl.types for training models to write GPU kernels.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Sequence

import chz
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
)
from tinker_cookbook.utils import logtree

from kernelbench_tinker.envs.kernelbench_client import (
    KernelBenchProblem,
    KernelEvalResult,
    ParsedResponse,
    evaluate_kernel_async,
    get_problem_ids,
    parse_structured_response,
)
from kernelbench_tinker.config.configs import EvalConfig
from kernelbench_tinker.training.reward import (
    compute_reward,
    compute_reward_breakdown,
    RewardConfig,
)
from kernelbench_tinker.training.trace_logger import get_trace_logger

logger = logging.getLogger(__name__)


_BASE_SYSTEM_PROMPT = """\
You are an expert GPU kernel developer. Your task is to optimize PyTorch \
operations by writing efficient custom {backend} kernels.

When given a PyTorch model, write an optimized kernel implementation.

Your solution must:
- Be a drop-in replacement as a class named `ModelNew`
- Use custom {backend} kernels, not just PyTorch operations
- Be correct and produce the same results as the reference

You MUST respond in exactly this format:

<KERNEL>
```python
# Your complete optimized implementation here
class ModelNew(nn.Module):
    ...
```
</KERNEL>"""

def build_system_prompt(backend: str) -> str:
    """Build the system prompt with backend-specific instructions."""
    return _BASE_SYSTEM_PROMPT.format(backend=backend.upper())


class KernelBenchEnv(Env):
    """
    A single-turn RL environment for a KernelBench problem.

    Each episode consists of:
    1. Initial observation: The problem prompt (PyTorch reference code + instructions)
    2. Action: The model generates a kernel implementation
    3. Step: Evaluate the kernel and compute reward

    The episode ends after one step.
    """

    def __init__(
        self,
        problem: KernelBenchProblem,
        renderer: renderers.Renderer,
        eval_config: EvalConfig | None = None,
        reward_config: RewardConfig | None = None,
    ):
        self.problem = problem
        self.renderer = renderer
        self.eval_config = eval_config or EvalConfig()
        self.reward_config = reward_config or RewardConfig()

        self._current_prompt_messages: list[renderers.Message] | None = None

    @property
    def stop_condition(self) -> StopCondition:
        """Get stop sequences for generation."""
        return self.renderer.get_stop_sequences()

    def _build_initial_messages(self) -> list[renderers.Message]:
        """Build the initial conversation for the problem."""
        messages: list[renderers.Message] = []

        messages.append({"role": "system", "content": build_system_prompt(self.problem.backend)})

        # Add the problem prompt as user message
        messages.append({"role": "user", "content": self.problem.prompt})

        return messages

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        messages = self._build_initial_messages()
        observation = self.renderer.build_generation_prompt(messages)
        self._current_prompt_messages = messages
        return observation, self.stop_condition

    async def step(self, action: Action) -> StepResult:
        step_start = time.perf_counter()
        # Parse the response to get text
        message, _ = self.renderer.parse_response(action)
        response_text = message.get("content", "")

        # Parse structured response (extracts <KERNEL> block)
        parsed = parse_structured_response(response_text)
        kernel_code = parsed.kernel
        # Check format validity
        format_ok = parsed.format_ok

        # Evaluate the kernel (Modal for isolated GPU execution)
        eval_start = time.perf_counter()
        cfg = self.eval_config
        eval_result = await evaluate_kernel_async(
            level=self.problem.level,
            problem_id=self.problem.problem_id,
            backend=self.problem.backend,
            kernel_code=kernel_code,
            dataset_src=self.problem.dataset_src,
            num_correct_trials=cfg.num_correct_trials,
            measure_performance=cfg.measure_performance,
            num_perf_trials=cfg.num_perf_trials,
            timing_method=cfg.timing_method,
            precision=cfg.precision,
            check_for_excessive_speedup=cfg.check_for_excessive_speedup,
            excessive_speedup_threshold=cfg.excessive_speedup_threshold,
            timeout=cfg.modal_timeout,
        )
        eval_time = time.perf_counter() - eval_start

        # Compute reward (pass kernel_code for static checking)
        reward = compute_reward(
            eval_result,
            self.reward_config,
            kernel_code=kernel_code,
            backend=self.problem.backend,
        )

        # Log the attempt
        logtree.log_text(f"Problem: Level {self.problem.level}, ID {self.problem.problem_id}")
        logtree.log_text(f"Format OK: {'Yes' if format_ok else 'No'}")
        logtree.log_text(f"Compiled: {'Yes' if eval_result['compiled'] else 'No'}")
        logtree.log_text(
            f"Correctness: {eval_result['tests_passed']}/{eval_result['tests_total']}"
        )
        if eval_result.get("speedup"):
            logtree.log_text(f"Speedup: {eval_result['speedup']:.2f}x")
        logtree.log_text(f"Reward: {reward:.3f}")
        error_message = eval_result.get("error_message")
        if error_message:
            logtree.log_text(f"Error: {error_message[:200]}")

        # Build metrics
        metrics: Metrics = {
            "level": self.problem.level,
            "problem_id": self.problem.problem_id,
            "format_ok": float(format_ok),
            "compiled": float(eval_result["compiled"]),
            "correctness": float(eval_result["correctness"]),
            "tests_passed": eval_result["tests_passed"],
            "tests_total": eval_result["tests_total"],
        }
        if eval_result.get("speedup"):
            metrics["speedup"] = eval_result["speedup"]
        if eval_result.get("runtime_ms"):
            metrics["runtime_ms"] = eval_result["runtime_ms"]
        metrics["time/eval"] = eval_time
        timing_metadata = (eval_result.get("metadata") or {}).get("timings", {})
        if "reference_load_s" in timing_metadata:
            metrics["time/ref_load"] = timing_metadata["reference_load_s"]
        if "modal_eval_s" in timing_metadata:
            metrics["time/modal_eval"] = timing_metadata["modal_eval_s"]
        metrics["time/step_total"] = time.perf_counter() - step_start

        # Trace logging (prompt + response + eval)
        await self._log_trace(
            parsed=parsed,
            eval_result=eval_result,
            format_ok=format_ok,
            reward=reward,
            metrics=metrics,
        )

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )

    async def _log_trace(
        self,
        parsed: ParsedResponse,
        eval_result: KernelEvalResult,
        format_ok: bool,
        reward: float,
        metrics: Metrics,
    ) -> None:
        """Append a JSONL trace for this step if trace logging is enabled."""
        trace_logger = get_trace_logger()
        if trace_logger is None:
            return

        trace_record = {
            "level": self.problem.level,
            "problem_id": self.problem.problem_id,
            "backend": self.problem.backend,
            "dataset_src": self.problem.dataset_src,
            "prompt_option": self.problem.prompt_option,
            "prompt_messages": self._current_prompt_messages,
            "renderer": getattr(self.renderer, "name", type(self.renderer).__name__),
            "response": {
                "raw": parsed.raw,
                "kernel": parsed.kernel,
                "format_ok": format_ok,
            },
            "eval_result": eval_result,
            "reward": reward,
            "reward_breakdown": compute_reward_breakdown(
                eval_result, self.reward_config
            ),
            "metrics": metrics,
            "timestamp": time.time(),
            "stop_condition": str(self.stop_condition),
        }

        await trace_logger.log(trace_record)


@dataclass(frozen=True)
class KernelBenchEnvGroupBuilder(EnvGroupBuilder):
    """
    Builder for creating groups of KernelBench environments.

    Each group corresponds to a single problem with multiple rollouts.
    This enables GRPO-style training where rewards are normalized within groups.
    """

    problem: KernelBenchProblem
    renderer: renderers.Renderer
    group_size: int  # Number of rollouts per problem
    eval_config: EvalConfig = field(default_factory=EvalConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)

    async def make_envs(self) -> Sequence[Env]:
        """Create a group of environments for this problem."""
        return [
            KernelBenchEnv(
                problem=self.problem,
                renderer=self.renderer,
                eval_config=self.eval_config,
                reward_config=self.reward_config,
            )
            for _ in range(self.group_size)
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """
        Compute final group rewards.

        By default, we use per-step rewards only (returned by step()).
        This method can be overridden for group-level reward modifications.
        """
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        """Return tags for logging and metric aggregation."""
        return [
            f"level_{self.problem.level}",
            f"problem_{self.problem.problem_id}",
            "kernelbench",
        ]


class KernelBenchRLDataset(RLDataset):
    """
    RL Dataset for KernelBench problems.

    Provides batches of EnvGroupBuilders for training.
    Each batch contains multiple problems, each with multiple rollouts.
    """

    def __init__(
        self,
        problems: list[KernelBenchProblem],
        renderer: renderers.Renderer,
        batch_size: int,
        group_size: int,
        eval_config: EvalConfig | None = None,
        reward_config: RewardConfig | None = None,
        shuffle: bool = True,
        num_epochs: int = 1,
        # Multi-turn fields (active when max_turns > 1)
        max_turns: int = 1,
        system_prompt: str | None = None,
        early_stop_on_correct: bool = False,
        speedup_threshold: float | None = None,
        tokenizer: object | None = None,
        prompt_max_tokens: int | None = None,
        inject_think_token: bool = False,
    ):
        self.problems = problems
        self.renderer = renderer
        self.batch_size = batch_size
        self.group_size = group_size
        self.eval_config = eval_config or EvalConfig()
        self.reward_config = reward_config or RewardConfig()
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        self.early_stop_on_correct = early_stop_on_correct
        self.speedup_threshold = speedup_threshold
        self.tokenizer = tokenizer
        self.prompt_max_tokens = prompt_max_tokens
        self.inject_think_token = inject_think_token

        # Create shuffled indices for each epoch
        self._problem_indices: list[int] = []
        for _ in range(num_epochs):
            epoch_indices = list(range(len(problems)))
            if shuffle:
                random.shuffle(epoch_indices)
            self._problem_indices.extend(epoch_indices)

    def __len__(self) -> int:
        """Number of batches in the dataset."""
        total_problems = len(self.problems) * self.num_epochs
        return (total_problems + self.batch_size - 1) // self.batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self._problem_indices))

        builders = []
        for i in range(start_idx, end_idx):
            problem_idx = self._problem_indices[i]
            problem = self.problems[problem_idx]

            if self.max_turns > 1:
                from kernelbench_tinker.envs.multiturn_kernelbench_env import (
                    MultiTurnKernelBenchEnvGroupBuilder,
                )
                builder = MultiTurnKernelBenchEnvGroupBuilder(
                    problem=problem,
                    renderer=self.renderer,
                    group_size=self.group_size,
                    max_turns=self.max_turns,
                    eval_config=self.eval_config,
                    reward_config=self.reward_config,
                    system_prompt=self.system_prompt,
                    early_stop_on_correct=self.early_stop_on_correct,
                    speedup_threshold=self.speedup_threshold,
                    tokenizer=self.tokenizer,
                    prompt_max_tokens=self.prompt_max_tokens,
                    inject_think_token=self.inject_think_token,
                )
            else:
                builder = KernelBenchEnvGroupBuilder(
                    problem=problem,
                    renderer=self.renderer,
                    group_size=self.group_size,
                    eval_config=self.eval_config,
                    reward_config=self.reward_config,
                )
            builders.append(builder)

        return builders


@chz.chz
class KernelBenchDatasetBuilder(RLDatasetBuilder):
    """
    Builder for creating KernelBench RL datasets.

    This is a chz-compatible configuration class that can be used
    in Tinker's training configuration.
    """

    # Problem selection
    level: int = 1
    levels: list[int] | None = None  # Train on multiple levels (overrides level when set)
    start_problem: int | None = None
    end_problem: int | None = None
    backend: str = "triton"
    dataset_src: str = "huggingface"

    # Training configuration
    batch_size: int = 4
    group_size: int = 4
    num_epochs: int = 1
    shuffle: bool = True

    # Multi-turn configuration (active when max_turns > 1)
    max_turns: int = 1
    early_stop_on_correct: bool = False
    speedup_threshold: float | None = None
    inject_think_token: bool = False  # Append <think>\n to generation prompts

    # Evaluation configuration
    num_correct_trials: int = 5
    measure_performance: bool = False
    num_perf_trials: int = 100
    timing_method: str = "cuda_event"
    precision: str = "fp32"
    check_for_excessive_speedup: bool = True
    excessive_speedup_threshold: float = 10.0

    # Reward configuration
    reward_format_weight: float = 0.0
    reward_compile_weight: float = 0.0
    reward_correctness_weight: float = 0.3
    reward_speed_weight: float = 1.0
    reward_length_weight: float = 0.0

    # Reward clipping and speed cap
    reward_clip_min: float | None = None
    reward_clip_max: float | None = None
    reward_speed_max_reward: float = 10.0  # Cap on speed reward component

    # Reward hacking detection (static checker)
    reward_enable_static_checker: bool = True
    reward_static_checker_backend: str = "triton"
    reward_static_checker_precision: str = "fp32"
    reward_static_checker_strict: list[str] | None = None
    reward_static_checker_warnings: list[str] | None = None

    # Renderer
    renderer_name: str = "qwen3"

    # Test split
    test_fraction: float = 0.1
    # Explicit holdout indices per level (overrides test_fraction when set)
    # Format: {level: [problem_ids]} e.g. {1: [3,10,25], 2: [10,20,30]}
    holdout_indices: dict[int, list[int]] | None = None

    # Prompt configuration
    prompt_option: str = "one_shot"  # "zero_shot", "one_shot", "few_shot"
    prompt_precision: str | None = None
    prompt_include_hardware: bool = False
    prompt_gpu_name: str | None = None
    prompt_max_tokens: int | None = None  # Token budget for multi-turn history truncation

    # Modal configuration (isolated GPU evaluation)
    modal_gpu_type: str = "A100"  # GPU type to use on Modal
    modal_timeout: float = 120.0  # Timeout in seconds per kernel

    async def __call__(self, tokenizer=None) -> tuple[RLDataset, RLDataset | None]:
        """Build train and optional test datasets.

        Args:
            tokenizer: The tokenizer to use for the renderer. Required for most renderers.
        """
        # Determine which levels to use
        active_levels = self.levels if self.levels else [self.level]

        # Collect problems across all levels
        all_problems: list[KernelBenchProblem] = []
        for lvl in active_levels:
            problem_ids = get_problem_ids(
                lvl,
                start=self.start_problem,
                end=self.end_problem,
                dataset_src=self.dataset_src,
            )
            all_problems.extend(
                KernelBenchProblem(
                    level=lvl,
                    problem_id=pid,
                    backend=self.backend,
                    dataset_src=self.dataset_src,
                    prompt_option=self.prompt_option,
                    prompt_precision=self.prompt_precision or self.precision,
                    prompt_include_hardware=self.prompt_include_hardware,
                    prompt_gpu_name=self.prompt_gpu_name or (
                        self.modal_gpu_type if self.prompt_include_hardware else None
                    ),
                )
                for pid in problem_ids
            )

        # Split into train/test
        if self.holdout_indices:
            # Explicit holdout: separate by (level, problem_id) membership
            holdout_set = {
                (lvl, pid)
                for lvl, pids in self.holdout_indices.items()
                for pid in pids
            }
            train_problems = [
                p for p in all_problems
                if (p.level, p.problem_id) not in holdout_set
            ]
            test_problems = [
                p for p in all_problems
                if (p.level, p.problem_id) in holdout_set
            ] or None
        elif self.test_fraction > 0 and len(all_problems) > 1:
            n_test = max(1, int(len(all_problems) * self.test_fraction))
            # Use last N problems as test set for reproducibility
            train_problems = all_problems[:-n_test]
            test_problems = all_problems[-n_test:]
        else:
            train_problems = all_problems
            test_problems = None

        # Create renderer
        renderer = renderers.get_renderer(self.renderer_name, tokenizer)

        # Create EvalConfig from flat YAML fields (single source of truth)
        eval_config = EvalConfig(
            num_correct_trials=self.num_correct_trials,
            measure_performance=self.measure_performance,
            num_perf_trials=self.num_perf_trials,
            timing_method=self.timing_method,
            precision=self.precision,
            check_for_excessive_speedup=self.check_for_excessive_speedup,
            excessive_speedup_threshold=self.excessive_speedup_threshold,
            modal_gpu_type=self.modal_gpu_type,
            modal_timeout=self.modal_timeout,
        )

        # Create reward config
        reward_config = RewardConfig(
            format_weight=self.reward_format_weight,
            compile_weight=self.reward_compile_weight,
            correctness_weight=self.reward_correctness_weight,
            speed_weight=self.reward_speed_weight,
            length_weight=self.reward_length_weight,
            speed_max_reward=self.reward_speed_max_reward,
            reward_clip_min=self.reward_clip_min,
            reward_clip_max=self.reward_clip_max,
            enable_static_checker=self.reward_enable_static_checker,
            static_checker_backend=self.reward_static_checker_backend or self.backend,
            static_checker_precision=self.reward_static_checker_precision or self.precision,
            static_checker_strict=self.reward_static_checker_strict,
            static_checker_warnings=self.reward_static_checker_warnings,
        )

        # Configure Modal evaluator with the same config
        from kernelbench_tinker.modal.evaluator import ModalEvaluatorConfig, set_modal_evaluator, ModalKernelEvaluator
        modal_config = ModalEvaluatorConfig(
            enabled=True,
            gpu_type=eval_config.modal_gpu_type,
            timeout=int(eval_config.modal_timeout),
        )
        set_modal_evaluator(ModalKernelEvaluator(modal_config))
        logger.info(f"Modal evaluator configured: GPU={eval_config.modal_gpu_type}, timeout={eval_config.modal_timeout}s")

        # Create train dataset
        train_dataset = KernelBenchRLDataset(
            problems=train_problems,
            renderer=renderer,
            batch_size=self.batch_size,
            group_size=self.group_size,
            eval_config=eval_config,
            reward_config=reward_config,
            shuffle=self.shuffle,
            num_epochs=self.num_epochs,
            max_turns=self.max_turns,
            early_stop_on_correct=self.early_stop_on_correct,
            speedup_threshold=self.speedup_threshold,
            tokenizer=tokenizer,
            prompt_max_tokens=self.prompt_max_tokens,
            inject_think_token=self.inject_think_token,
        )

        # Create test dataset if we have test problems
        test_dataset = None
        if test_problems:
            test_dataset = KernelBenchRLDataset(
                problems=test_problems,
                renderer=renderer,
                batch_size=self.batch_size,
                group_size=self.group_size,
                eval_config=eval_config,
                reward_config=reward_config,
                shuffle=False,
                num_epochs=1,
                max_turns=self.max_turns,
                early_stop_on_correct=self.early_stop_on_correct,
                speedup_threshold=self.speedup_threshold,
                tokenizer=tokenizer,
                prompt_max_tokens=self.prompt_max_tokens,
                inject_think_token=self.inject_think_token,
            )

        return train_dataset, test_dataset

