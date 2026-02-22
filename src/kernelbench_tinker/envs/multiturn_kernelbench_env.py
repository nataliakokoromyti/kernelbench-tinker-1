"""
Multi-turn KernelBench RL environment.

Extends the single-turn KernelBenchEnv to support iterative kernel refinement.
Each episode consists of up to T turns where the model receives evaluation
feedback and can fix errors or improve performance.
"""

from __future__ import annotations

import logging
import re
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

from kernelbench_tinker.config.configs import EvalConfig
from kernelbench_tinker.envs.kernelbench_client import (
    KernelBenchProblem,
    KernelEvalResult,
    ParsedResponse,
    evaluate_kernel_async,
    get_problem_ids,
    parse_structured_response,
)
from kernelbench_tinker.training.reward import (
    RewardConfig,
    compute_reward,
    compute_reward_breakdown,
)
from kernelbench_tinker.training.trace_logger import get_trace_logger

logger = logging.getLogger(__name__)

# Limits for feedback content included in refinement prompts
MAX_KERNEL_HISTORY_LEN = 2000
MAX_ERROR_LEN = 800
MAX_ERROR_LINES = 10


# ---------------------------------------------------------------------------
# Error extraction and categorization helpers
# ---------------------------------------------------------------------------


def _extract_key_error(error_message: str | None) -> str:
    """Extract a cleaned summary of the main error with relevant context."""
    if not error_message:
        return ""

    patterns = [
        r"(\w+Error: .+?)(?:\n\n|\n(?=[A-Z])|$)",
        r"(\w+Exception: .+?)(?:\n\n|\n(?=[A-Z])|$)",
        r"(triton\.compiler\.errors\.\w+: .+?)(?:\n|$)",
        r"(triton\.runtime\.errors\.\w+: .+?)(?:\n|$)",
        r"(nvcc.+?error.+?)(?:\n|$)",
        r"(CUDA error: .+?)(?:\n|$)",
        r"(RuntimeError: .+?)(?:\n|$)",
        r"(TypeError: .+?)(?:\n|$)",
        r"(ValueError: .+?)(?:\n|$)",
        r"(Correctness check failed.+?)(?:\n|$)",
        r"(Output mismatch.+?)(?:\n|$)",
        r"(CompilationError: .+?)(?:\n|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, error_message, re.IGNORECASE | re.DOTALL)
        if match:
            error_text = match.group(1).strip()
            error_text = re.sub(r"\n\s*\n", "\n", error_text)
            return error_text[:MAX_ERROR_LEN]

    # Fallback: collect meaningful lines
    meaningful_lines = []
    for line in error_message.split("\n"):
        line = line.strip()
        if not line or line.startswith("Traceback") or line.startswith('File "'):
            continue
        meaningful_lines.append(line)
        if len(meaningful_lines) >= MAX_ERROR_LINES:
            break

    if meaningful_lines:
        return "\n".join(meaningful_lines)[:MAX_ERROR_LEN]

    return error_message[:MAX_ERROR_LEN]


def _categorize_error(eval_result: KernelEvalResult) -> str:
    """Categorize the evaluation result for feedback."""
    if not eval_result["format_ok"]:
        return "format_error"
    if not eval_result["compiled"]:
        return "compilation_error"
    if not eval_result["correctness"]:
        error_msg = eval_result.get("error_message", "") or ""
        if "Error" in error_msg or "Exception" in error_msg:
            return "runtime_error"
        return "correctness_error"
    if eval_result.get("speedup") is not None and eval_result["speedup"] < 1.0:
        return "performance_warning"
    return "success"


_ERROR_GUIDANCE = {
    "format_error": (
        "Ensure your output is a valid Python class named `ModelNew` "
        "wrapped in a code block."
    ),
    "compilation_error": (
        "Fix the syntax/API errors. Check that all kernel functions "
        "are correctly decorated and all imports are valid."
    ),
    "runtime_error": (
        "Fix the runtime error. Common issues: shape mismatches, "
        "incorrect tensor indexing, or invalid kernel launch configurations."
    ),
    "correctness_error": (
        "The kernel runs but produces incorrect output. Check your algorithm "
        "implementation, boundary conditions, and reduction operations."
    ),
    "performance_warning": (
        "The kernel is correct but slower than PyTorch. Consider optimizing "
        "memory access patterns, tiling, or parallelization strategy."
    ),
}


# ---------------------------------------------------------------------------
# Multi-turn state
# ---------------------------------------------------------------------------


@dataclass
class MultiTurnState:
    """Mutable state for a multi-turn kernel refinement episode."""

    level: int
    problem_id: int
    backend: str
    turn_idx: int
    max_turns: int
    last_kernel: str | None
    last_eval: KernelEvalResult | None
    step_scores: list[float]
    done: bool
    success: bool


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

MULTITURN_SYSTEM_PROMPT = (
    "You are an expert GPU kernel developer. Your task is to optimize PyTorch "
    "operations by writing efficient custom {backend} kernels.\n\n"
    "When given a PyTorch model, write an optimized kernel implementation. "
    "If a previous attempt failed, fix the errors based on the feedback provided.\n\n"
    "Your solution must:\n"
    "- Be a drop-in replacement as a class named `ModelNew`\n"
    "- Use custom {backend} kernels, not just PyTorch operations\n"
    "- Be correct and produce the same results as the reference\n\n"
    "You MUST respond in exactly this format:\n\n"
    "<KERNEL>\n"
    "```python\n"
    "# Your complete optimized implementation here\n"
    "class ModelNew(nn.Module):\n"
    "    ...\n"
    "```\n"
    "</KERNEL>"
)


REFINEMENT_TEMPLATE = (
    "\n## Previous Attempt (Turn {turn})\n\n"
    "```python\n{previous_kernel}\n```\n\n"
    "## Evaluation Feedback\n"
    "- **Status**: {error_category}\n"
    "- **Compiled**: {compiled}\n"
    "- **Tests Passed**: {tests_passed}/{tests_total}\n"
    "{speedup_line}\n\n"
    "{error_section}\n\n"
    "## Instructions\n\n"
    "{guidance}\n\n"
    "Keep what works. Do not change the function signature unless necessary. "
    "Do not use PyTorch APIs for the core computation.\n"
)


ERROR_SECTION_TEMPLATE = "### Error Details\n```\n{error_text}\n```\n"


# ---------------------------------------------------------------------------
# Multi-turn environment
# ---------------------------------------------------------------------------


class MultiTurnKernelBenchEnv(Env):
    """
    Multi-turn RL environment for KernelBench.

    Each episode consists of up to T refinement steps:
    1. Turn 0: problem prompt (same as single-turn)
    2. Turn 1+: problem prompt + previous attempt feedback

    The episode ends when the kernel is correct (early stopping) or
    max_turns is reached.
    """

    def __init__(
        self,
        problem: KernelBenchProblem,
        renderer: renderers.Renderer,
        max_turns: int = 4,
        eval_config: EvalConfig | None = None,
        reward_config: RewardConfig | None = None,
        system_prompt: str | None = None,
        early_stop_on_correct: bool = True,
        speedup_threshold: float | None = None,
    ):
        self.problem = problem
        self.renderer = renderer
        self.max_turns = max_turns
        self.eval_config = eval_config or EvalConfig()
        self.reward_config = reward_config or RewardConfig()
        self.early_stop_on_correct = early_stop_on_correct
        self.speedup_threshold = speedup_threshold

        self._system_prompt = system_prompt or MULTITURN_SYSTEM_PROMPT.format(
            backend=problem.backend.upper()
        )

        self._current_prompt_messages: list[renderers.Message] | None = None
        self._current_observation: tinker.ModelInput | None = None
        self._state: MultiTurnState | None = None

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    @property
    def state(self) -> MultiTurnState:
        if self._state is None:
            raise RuntimeError(
                "Environment not initialized. Call initial_observation first."
            )
        return self._state

    def _build_initial_messages(self) -> list[renderers.Message]:
        messages: list[renderers.Message] = []
        messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": self.problem.prompt})
        return messages

    def _build_refinement_messages(self) -> list[renderers.Message]:
        """Build a fresh prompt with the original problem + feedback from last attempt."""
        messages: list[renderers.Message] = []
        messages.append({"role": "system", "content": self._system_prompt})

        user_parts = [self.problem.prompt]

        if self.state.last_eval is not None and self.state.last_kernel is not None:
            eval_result = self.state.last_eval
            error_category = _categorize_error(eval_result)
            display_names = {
                "format_error": "FORMAT ERROR - Invalid code structure",
                "compilation_error": "COMPILATION ERROR - Build failed",
                "runtime_error": "RUNTIME ERROR - Crashed during execution",
                "correctness_error": "CORRECTNESS ERROR - Wrong output",
                "performance_warning": "CORRECT (but slower than baseline)",
                "success": "SUCCESS",
            }

            speedup_line = ""
            if eval_result.get("speedup") is not None:
                speedup_line = f"- **Speedup**: {eval_result['speedup']:.2f}x"

            error_section = ""
            if eval_result.get("error_message") and error_category != "success":
                error_text = _extract_key_error(eval_result["error_message"])
                if error_text:
                    error_section = ERROR_SECTION_TEMPLATE.format(
                        error_text=error_text
                    )

            guidance = _ERROR_GUIDANCE.get(
                error_category, "Fix the issues and try again."
            )

            kernel_code = self.state.last_kernel
            if len(kernel_code) > MAX_KERNEL_HISTORY_LEN:
                kernel_code = (
                    kernel_code[:MAX_KERNEL_HISTORY_LEN] + "\n# ... (truncated)"
                )

            refinement = REFINEMENT_TEMPLATE.format(
                turn=self.state.turn_idx,
                previous_kernel=kernel_code,
                error_category=display_names.get(
                    error_category, error_category.upper()
                ),
                compiled="Yes" if eval_result["compiled"] else "No",
                tests_passed=eval_result["tests_passed"],
                tests_total=eval_result["tests_total"],
                speedup_line=speedup_line,
                error_section=error_section,
                guidance=guidance,
            )
            refinement += "\nRemember: respond using <KERNEL>...</KERNEL>."
            user_parts.append(refinement)

        messages.append({"role": "user", "content": "\n".join(user_parts)})
        return messages

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        self._state = MultiTurnState(
            level=self.problem.level,
            problem_id=self.problem.problem_id,
            backend=self.problem.backend,
            turn_idx=0,
            max_turns=self.max_turns,
            last_kernel=None,
            last_eval=None,
            step_scores=[],
            done=False,
            success=False,
        )
        messages = self._build_initial_messages()
        observation = self.renderer.build_generation_prompt(messages)
        self._current_prompt_messages = messages
        self._current_observation = observation
        return observation, self.stop_condition

    async def step(self, action: Action) -> StepResult:
        step_start = time.perf_counter()
        state = self.state

        message, _ = self.renderer.parse_response(action)
        response_text = message.get("content", "")

        parsed = parse_structured_response(response_text)
        kernel_code = parsed.kernel
        state.last_kernel = kernel_code
        format_ok = parsed.format_ok

        # Evaluate on Modal
        eval_start = time.perf_counter()
        cfg = self.eval_config
        eval_result = await evaluate_kernel_async(
            level=state.level,
            problem_id=state.problem_id,
            backend=state.backend,
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
        state.last_eval = eval_result

        # Per-step reward
        step_score = compute_reward(
            eval_result,
            self.reward_config,
            kernel_code=kernel_code,
            backend=state.backend,
        )
        state.step_scores.append(step_score)

        # Log
        logtree.log_text(
            f"Multi-turn: Level {state.level}, ID {state.problem_id}, "
            f"Turn {state.turn_idx}"
        )
        logtree.log_text(f"Format OK: {'Yes' if format_ok else 'No'}")
        logtree.log_text(
            f"Compiled: {'Yes' if eval_result['compiled'] else 'No'}"
        )
        logtree.log_text(
            f"Correctness: {eval_result['tests_passed']}/{eval_result['tests_total']}"
        )
        if eval_result.get("speedup"):
            logtree.log_text(f"Speedup: {eval_result['speedup']:.2f}x")
        logtree.log_text(f"Step score: {step_score:.3f}")

        # Early stopping
        is_correct = eval_result["correctness"]
        meets_speedup = (
            self.speedup_threshold is None
            or (eval_result.get("speedup") or 0.0) >= self.speedup_threshold
        )
        if self.early_stop_on_correct and is_correct and meets_speedup:
            state.done = True
            state.success = True

        state.turn_idx += 1
        if state.turn_idx >= state.max_turns:
            state.done = True

        # Metrics
        metrics: Metrics = {
            "level": state.level,
            "problem_id": state.problem_id,
            "turn": state.turn_idx - 1,
            "format_ok": float(format_ok),
            "compiled": float(eval_result["compiled"]),
            "correctness": float(eval_result["correctness"]),
            "tests_passed": eval_result["tests_passed"],
            "tests_total": eval_result["tests_total"],
            "step_score": step_score,
            "episode_done": float(state.done),
            "episode_success": float(state.success),
        }
        if eval_result.get("speedup") is not None:
            metrics["speedup"] = eval_result["speedup"]
        if eval_result.get("runtime_ms") is not None:
            metrics["runtime_ms"] = eval_result["runtime_ms"]
        metrics["time/eval"] = eval_time
        timing_metadata = (eval_result.get("metadata") or {}).get("timings", {})
        if "reference_load_s" in timing_metadata:
            metrics["time/ref_load"] = timing_metadata["reference_load_s"]
        if "modal_eval_s" in timing_metadata:
            metrics["time/modal_eval"] = timing_metadata["modal_eval_s"]
        metrics["time/step_total"] = time.perf_counter() - step_start

        # Trace logging
        await self._log_trace(
            parsed=parsed,
            eval_result=eval_result,
            format_ok=format_ok,
            reward=step_score,
            metrics=metrics,
        )

        # Next observation or done
        if state.done:
            next_observation = tinker.ModelInput.empty()
        else:
            messages = self._build_refinement_messages()
            next_observation = self.renderer.build_generation_prompt(messages)
            self._current_prompt_messages = messages
            self._current_observation = next_observation

        return StepResult(
            reward=step_score,
            episode_done=state.done,
            next_observation=next_observation,
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
        trace_logger = get_trace_logger()
        if trace_logger is None:
            return

        trace_record = {
            "mode": "multi_turn",
            "level": self.problem.level,
            "problem_id": self.problem.problem_id,
            "backend": self.problem.backend,
            "dataset_src": self.problem.dataset_src,
            "prompt_option": self.problem.prompt_option,
            "turn": self.state.turn_idx - 1,
            "max_turns": self.state.max_turns,
            "prompt_messages": self._current_prompt_messages,
            "renderer": getattr(
                self.renderer, "name", type(self.renderer).__name__
            ),
            "response": {
                "raw": parsed.raw,
                "kernel": parsed.kernel,
                "format_ok": format_ok,
            },
            "eval_result": eval_result,
            "reward": reward,
            "reward_breakdown": compute_reward_breakdown(
                eval_result,
                self.reward_config,
                kernel_code=parsed.kernel,
                backend=self.problem.backend,
            ),
            "metrics": metrics,
            "state": {
                "turn_idx": self.state.turn_idx,
                "done": self.state.done,
                "success": self.state.success,
                "step_scores": list(self.state.step_scores),
            },
            "timestamp": time.time(),
            "stop_condition": str(self.stop_condition),
        }

        await trace_logger.log(trace_record)

    def get_step_scores(self) -> list[float]:
        """Return per-step scores for discounted return computation."""
        return list(self.state.step_scores)


# ---------------------------------------------------------------------------
# Group builder, dataset, dataset builder (mirrors single-turn structure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiTurnKernelBenchEnvGroupBuilder(EnvGroupBuilder):
    """Builder for groups of multi-turn KernelBench environments."""

    problem: KernelBenchProblem
    renderer: renderers.Renderer
    group_size: int
    max_turns: int = 4
    eval_config: EvalConfig = field(default_factory=EvalConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    system_prompt: str | None = None
    early_stop_on_correct: bool = True
    speedup_threshold: float | None = None

    async def make_envs(self) -> Sequence[Env]:
        return [
            MultiTurnKernelBenchEnv(
                problem=self.problem,
                renderer=self.renderer,
                max_turns=self.max_turns,
                eval_config=self.eval_config,
                reward_config=self.reward_config,
                system_prompt=self.system_prompt,
                early_stop_on_correct=self.early_stop_on_correct,
                speedup_threshold=self.speedup_threshold,
            )
            for _ in range(self.group_size)
        ]

    async def compute_group_rewards(
        self,
        trajectory_group: list[Trajectory],
        env_group: Sequence[Env],
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return [
            f"level_{self.problem.level}",
            f"problem_{self.problem.problem_id}",
            "kernelbench",
            "multiturn",
        ]


class MultiTurnKernelBenchRLDataset(RLDataset):
    """RL Dataset for multi-turn KernelBench problems."""

    def __init__(
        self,
        problems: list[KernelBenchProblem],
        renderer: renderers.Renderer,
        batch_size: int,
        group_size: int,
        max_turns: int = 4,
        eval_config: EvalConfig | None = None,
        reward_config: RewardConfig | None = None,
        system_prompt: str | None = None,
        early_stop_on_correct: bool = True,
        speedup_threshold: float | None = None,
        shuffle: bool = True,
        num_epochs: int = 1,
    ):
        self.problems = problems
        self.renderer = renderer
        self.batch_size = batch_size
        self.group_size = group_size
        self.max_turns = max_turns
        self.eval_config = eval_config or EvalConfig()
        self.reward_config = reward_config or RewardConfig()
        self.system_prompt = system_prompt
        self.early_stop_on_correct = early_stop_on_correct
        self.speedup_threshold = speedup_threshold
        self.shuffle = shuffle
        self.num_epochs = num_epochs

        import random

        self._problem_indices: list[int] = []
        for _ in range(num_epochs):
            epoch_indices = list(range(len(problems)))
            if shuffle:
                random.shuffle(epoch_indices)
            self._problem_indices.extend(epoch_indices)

    def __len__(self) -> int:
        total_problems = len(self.problems) * self.num_epochs
        return (total_problems + self.batch_size - 1) // self.batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self._problem_indices))

        builders = []
        for i in range(start_idx, end_idx):
            problem_idx = self._problem_indices[i]
            problem = self.problems[problem_idx]

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
            )
            builders.append(builder)

        return builders


@chz.chz
class MultiTurnKernelBenchDatasetBuilder(RLDatasetBuilder):
    """Builder for multi-turn KernelBench RL datasets."""

    # Problem selection
    level: int = 1
    start_problem: int | None = None
    end_problem: int | None = None
    backend: str = "triton"
    dataset_src: str = "huggingface"

    # Training configuration
    batch_size: int = 4
    group_size: int = 4
    num_epochs: int = 1
    shuffle: bool = True

    # Multi-turn configuration
    max_turns: int = 4
    early_stop_on_correct: bool = True
    speedup_threshold: float | None = None

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

    # Prompt configuration
    prompt_option: str = "one_shot"
    prompt_precision: str | None = None
    prompt_include_hardware: bool = False
    prompt_gpu_name: str | None = None

    # Modal configuration
    modal_gpu_type: str = "A100"
    modal_timeout: float = 120.0

    async def __call__(
        self, tokenizer=None
    ) -> tuple[RLDataset, RLDataset | None]:
        problem_ids = get_problem_ids(
            self.level,
            start=self.start_problem,
            end=self.end_problem,
            dataset_src=self.dataset_src,
        )

        all_problems = [
            KernelBenchProblem(
                level=self.level,
                problem_id=pid,
                backend=self.backend,
                dataset_src=self.dataset_src,
                prompt_option=self.prompt_option,
                prompt_precision=self.prompt_precision or self.precision,
                prompt_include_hardware=self.prompt_include_hardware,
                prompt_gpu_name=self.prompt_gpu_name
                or (self.modal_gpu_type if self.prompt_include_hardware else None),
            )
            for pid in problem_ids
        ]

        if self.test_fraction > 0 and len(all_problems) > 1:
            n_test = max(1, int(len(all_problems) * self.test_fraction))
            train_problems = all_problems[:-n_test]
            test_problems = all_problems[-n_test:]
        else:
            train_problems = all_problems
            test_problems = None

        renderer = renderers.get_renderer(self.renderer_name, tokenizer)

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

        reward_config = RewardConfig(
            format_weight=self.reward_format_weight,
            compile_weight=self.reward_compile_weight,
            correctness_weight=self.reward_correctness_weight,
            speed_weight=self.reward_speed_weight,
            length_weight=self.reward_length_weight,
            enable_static_checker=self.reward_enable_static_checker,
            static_checker_backend=(
                self.reward_static_checker_backend or self.backend
            ),
            static_checker_precision=(
                self.reward_static_checker_precision or self.precision
            ),
            static_checker_strict=self.reward_static_checker_strict,
            static_checker_warnings=self.reward_static_checker_warnings,
        )

        from kernelbench_tinker.modal.evaluator import (
            ModalEvaluatorConfig,
            ModalKernelEvaluator,
            set_modal_evaluator,
        )

        modal_config = ModalEvaluatorConfig(
            enabled=True,
            gpu_type=eval_config.modal_gpu_type,
            timeout=int(eval_config.modal_timeout),
        )
        set_modal_evaluator(ModalKernelEvaluator(modal_config))
        logger.info(
            "Modal evaluator configured: GPU=%s, timeout=%ss",
            eval_config.modal_gpu_type,
            eval_config.modal_timeout,
        )

        train_dataset = MultiTurnKernelBenchRLDataset(
            problems=train_problems,
            renderer=renderer,
            batch_size=self.batch_size,
            group_size=self.group_size,
            max_turns=self.max_turns,
            eval_config=eval_config,
            reward_config=reward_config,
            early_stop_on_correct=self.early_stop_on_correct,
            speedup_threshold=self.speedup_threshold,
            shuffle=self.shuffle,
            num_epochs=self.num_epochs,
        )

        test_dataset = None
        if test_problems:
            test_dataset = MultiTurnKernelBenchRLDataset(
                problems=test_problems,
                renderer=renderer,
                batch_size=self.batch_size,
                group_size=self.group_size,
                max_turns=self.max_turns,
                eval_config=eval_config,
                reward_config=reward_config,
                early_stop_on_correct=self.early_stop_on_correct,
                speedup_threshold=self.speedup_threshold,
                shuffle=False,
                num_epochs=1,
            )

        return train_dataset, test_dataset
