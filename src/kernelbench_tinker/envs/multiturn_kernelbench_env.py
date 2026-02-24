"""
Multi-turn KernelBench RL environment.

Extends the single-turn KernelBenchEnv to support iterative kernel refinement.
Each episode consists of up to T turns where the model receives evaluation
feedback and can fix errors or improve performance.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Sequence

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
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
    parse_structured_response,
)
from kernelbench_tinker.envs.kernelbench_env import build_system_prompt
from kernelbench_tinker.training.reward import (
    RewardConfig,
    compute_reward,
    compute_reward_breakdown,
)
from kernelbench_tinker.training.trace_logger import get_trace_logger

logger = logging.getLogger(__name__)

# Limits for feedback content included in refinement prompts
MAX_HISTORY_CONTEXT_LEN = 8000
MAX_ERROR_LEN = 800


def build_eval_feedback(eval_result: KernelEvalResult) -> str:
    """Build a human-readable feedback string from an evaluation result.

    Each feedback string ends with the instruction to restart reasoning,
    matching the original Kevin training code's ``response_for_kernel_eval``.
    """
    error_msg = (eval_result.get("error_message") or "")[:MAX_ERROR_LEN]

    if not eval_result["format_ok"]:
        if error_msg:
            resp = (
                "Your previous answer failed to be parsed due to not adhering "
                f"to the desired formatting. Here's the error message: {error_msg}.\n"
            )
        else:
            resp = (
                "Your previous answer failed to be parsed due to not adhering "
                "to the desired formatting.\n"
            )
    elif not eval_result["compiled"]:
        resp = (
            "Your previous answer failed to compile. "
            f"Here's the error message: {error_msg}.\n"
        )
    elif not eval_result["correctness"]:
        if error_msg:
            resp = (
                "Your previous answer compiled successfully but had runtime "
                f"errors. Here's the error message: {error_msg}.\n"
            )
        else:
            resp = (
                "Your previous answer was incorrect. "
                f"Here's the error message: {error_msg}.\n"
            )
    else:
        speedup = eval_result.get("speedup") or 0.0
        resp = (
            "Your previous answer was correct but can be made faster. "
            "Here's the speedup you achieved relative to the baseline: "
            f"{speedup:.2f}.\n"
        )

    resp += "\nRestart your reasoning process and generate new, complete code."
    return resp


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
    history: list[dict]  # Per-turn: {raw_content, kernel, feedback, score}
    step_scores: list[float]
    done: bool
    success: bool


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
        early_stop_on_correct: bool = False,
        speedup_threshold: float | None = None,
        tokenizer: object | None = None,
        prompt_max_tokens: int | None = None,
    ):
        self.problem = problem
        self.renderer = renderer
        self.max_turns = max_turns
        self.eval_config = eval_config or EvalConfig()
        self.reward_config = reward_config or RewardConfig()
        self.early_stop_on_correct = early_stop_on_correct
        self.speedup_threshold = speedup_threshold
        self.tokenizer = tokenizer
        self.prompt_max_tokens = prompt_max_tokens

        self._system_prompt = system_prompt or build_system_prompt(
            problem.backend, multiturn=True
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

    def _count_message_tokens(self, messages: list[renderers.Message]) -> int:
        """Count total tokens across all messages using the tokenizer."""
        if self.tokenizer is None:
            return 0
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if hasattr(self.tokenizer, "encode"):
                total += len(self.tokenizer.encode(content))
            else:
                # Rough fallback: ~4 chars per token
                total += len(content) // 4
        return total

    def _build_refinement_messages(self) -> list[renderers.Message]:
        """Build refinement prompt using multi-turn chat format.

        Strips the one-shot example from the problem prompt and structures
        the history as alternating assistant / user turns (matching the
        original Kevin training code's ``_generate_vllm_refinement``).

        The assistant turn contains the raw model output (everything after
        ``</think>``); the user turn contains evaluation feedback with the
        "Restart your reasoning" instruction.

        When a tokenizer and ``prompt_max_tokens`` are provided, truncation
        is token-based (Kevin-style): the base messages are tokenized and
        the remaining token budget is used to keep as many recent history
        entries as possible.  Otherwise, falls back to character-based
        truncation using ``MAX_HISTORY_CONTEXT_LEN``.
        """
        messages: list[renderers.Message] = []
        messages.append({"role": "system", "content": self._system_prompt})

        # Initial user message: problem without one-shot example
        messages.append({
            "role": "user",
            "content": self.problem.base_prompt + "Here are your previous attempts:\n",
        })

        if self.state.history:
            history = list(self.state.history)

            if self.tokenizer is not None and self.prompt_max_tokens is not None:
                # Token-based truncation (Kevin-style):
                # Count tokens used by base messages, then fit history
                # into the remaining budget, keeping most recent entries.
                base_tokens = self._count_message_tokens(messages)
                budget = self.prompt_max_tokens - base_tokens

                # Walk history backwards, accumulating tokens
                kept: list[dict] = []
                used = 0
                for entry in reversed(history):
                    entry_text = entry["raw_content"] + entry["feedback"]
                    if hasattr(self.tokenizer, "encode"):
                        entry_tokens = len(self.tokenizer.encode(entry_text))
                    else:
                        entry_tokens = len(entry_text) // 4
                    if used + entry_tokens > budget and kept:
                        break
                    kept.append(entry)
                    used += entry_tokens
                history = list(reversed(kept))
            else:
                # Char-based fallback
                total_len = sum(
                    len(e["raw_content"]) + len(e["feedback"]) for e in history
                )
                while total_len > MAX_HISTORY_CONTEXT_LEN and len(history) > 1:
                    removed = history.pop(0)
                    total_len -= len(removed["raw_content"]) + len(removed["feedback"])

            # Add history as assistant/user turn pairs
            for entry in history:
                messages.append({"role": "assistant", "content": entry["raw_content"]})
                messages.append({"role": "user", "content": entry["feedback"]})

        return messages

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        self._state = MultiTurnState(
            level=self.problem.level,
            problem_id=self.problem.problem_id,
            backend=self.problem.backend,
            turn_idx=0,
            max_turns=self.max_turns,
            history=[],
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

        # Per-step reward
        step_score = compute_reward(
            eval_result,
            self.reward_config,
            kernel_code=kernel_code,
            backend=state.backend,
        )
        state.step_scores.append(step_score)

        # Extract raw response content after </think> for history context.
        # This matches the original Kevin code which stores everything after
        # </think> as the assistant turn in multi-turn chat format.
        if "</think>" in response_text:
            raw_content = response_text.split("</think>")[-1].lstrip('\n')
        else:
            raw_content = response_text

        # Build feedback and store in history
        feedback = build_eval_feedback(eval_result)
        state.history.append({
            "raw_content": raw_content,
            "kernel": kernel_code,
            "feedback": feedback,
            "score": step_score,
        })

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
                "cot_summary": parsed.cot_summary,
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
            "history": [
                {
                    "raw_content": entry["raw_content"],
                    "kernel": entry["kernel"],
                    "feedback": entry["feedback"],
                    "score": entry["score"],
                }
                for entry in self.state.history
            ],
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
    early_stop_on_correct: bool = False
    speedup_threshold: float | None = None
    tokenizer: object | None = field(default=None, hash=False, compare=False)
    prompt_max_tokens: int | None = None

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
                tokenizer=self.tokenizer,
                prompt_max_tokens=self.prompt_max_tokens,
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


