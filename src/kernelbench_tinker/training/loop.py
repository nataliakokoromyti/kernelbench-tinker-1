"""RL training loop for KernelBench (single-turn and multi-turn)."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import time
from typing import Any, Sequence

import chz
import numpy as np
import tinker
import torch
from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.completers import (
    StopCondition,
    TinkerTokenCompleter,
    TokenCompleter,
    TokensWithLogprobs,
)
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
    remove_constant_reward_groups,
)
from tinker_cookbook.rl.rollouts import do_group_rollout, do_single_rollout
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Trajectory,
    TrajectoryGroup,
)
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import timed

from kernelbench_tinker.config.configs import MultiTurnConfig
from kernelbench_tinker.envs.kernelbench_env import KernelBenchDatasetBuilder
from kernelbench_tinker.envs.multiturn_kernelbench_env import MultiTurnKernelBenchEnv
from kernelbench_tinker.training.models import get_adam_params
from kernelbench_tinker.training.reward import compute_discounted_returns
from kernelbench_tinker.training.tensorboard_logger import (
    TensorBoardConfig,
    TensorBoardLogger,
    create_tensorboard_logger,
)
from kernelbench_tinker.training.trace_logger import TraceLogger, set_trace_logger


class KernelBenchTokenCompleter(TokenCompleter):
    """Token completer with top_p and seed support."""

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ):
        self.sampling_client = sampling_client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        sample_result = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                stop=stop,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                seed=self.seed,
            ),
        )
        sampled_tokens = sample_result.sequences[0].tokens
        sampled_logprobs = sample_result.sequences[0].logprobs
        assert sampled_logprobs is not None
        return TokensWithLogprobs(tokens=sampled_tokens, maybe_logprobs=sampled_logprobs)


def prepare_datum(
    datum: tinker.Datum,
    loss_fn: str = "importance_sampling",
    clip_epsilon_low: float = 0.0,
    clip_epsilon_high: float = 0.0,
) -> tinker.Datum:
    """Prepare a datum for forward-backward, adding PPO clip thresholds if configured."""
    inputs = {k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"}

    if loss_fn == "ppo" and clip_epsilon_low > 0:
        logprobs_td = inputs["logprobs"]
        seq_len = logprobs_td.to_torch().shape[0]
        td_cls = type(logprobs_td)
        inputs["clip_low_threshold"] = td_cls.from_torch(
            torch.full((seq_len,), 1.0 - clip_epsilon_low, dtype=torch.float32)
        )
        inputs["clip_high_threshold"] = td_cls.from_torch(
            torch.full((seq_len,), 1.0 + clip_epsilon_high, dtype=torch.float32)
        )

    return tinker.Datum(model_input=datum.model_input, loss_fn_inputs=inputs)

logger = logging.getLogger(__name__)


@chz.chz
class TrainingConfig:
    """Configuration for KernelBench RL training."""

    # Model configuration
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    lora_rank: int = 32
    learning_rate: float = 1e-4

    # Generation configuration
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0  # Nucleus sampling (1.0 = disabled)
    seed: int | None = None  # Random seed for generation (null = random)

    # Response length extension: extend max_tokens mid-training as the
    # model attempts more sophisticated solutions.  Set to 0 to disable.
    max_tokens_extended: int = 0  # e.g. 22528 (22K)
    max_tokens_extend_after_step: int = 0  # Step at which to switch

    # Dataset configuration
    dataset_builder: KernelBenchDatasetBuilder = chz.field(
        default_factory=KernelBenchDatasetBuilder
    )

    # Multi-turn specific config
    multiturn: MultiTurnConfig = chz.field(default_factory=MultiTurnConfig)

    # Training configuration
    num_substeps: int = 1  # Optimizer steps per batch
    loss_fn: LossFnType = "importance_sampling"
    max_grad_norm: float = 0.0  # 0.0 = no clipping
    warmup_ratio: float = 0.0  # Fraction of batches for linear LR warmup
    clip_epsilon_low: float = 0.0  # PPO clip lower bound (0.0 = use server default)
    clip_epsilon_high: float = 0.0  # PPO clip upper bound (Clip-High: 0.28)

    # Constant length normalization (Dr. GRPO).  Tinker sums per-token
    # losses (no 1/|o_i| division), so we scale advantages by
    # 1/constant_length_norm to get uniform gradient magnitude regardless
    # of response length.  Set to 0 to disable.
    constant_length_norm: int = 0  # e.g. max_tokens (16384)

    # KL regularization
    kl_penalty_coef: float = 0.0
    kl_discount_factor: float = 0.0

    # Logging and checkpointing
    log_path: str = "./runs/kernelbench_tinker"
    save_every: int = 10  # Save checkpoint every N batches

    # Remove groups where all rewards are the same (no learning signal)
    remove_constant_reward_groups: bool = True

    # Wandb logging
    wandb_project: str | None = None
    wandb_name: str | None = None

    # TensorBoard logging
    tensorboard_enabled: bool = True
    tensorboard_log_histograms_every: int = 5
    tensorboard_log_per_level: bool = True

    # Tinker API
    base_url: str | None = None

    # Resume from checkpoint
    load_checkpoint_path: str | None = None
    resume_from_batch: int | None = None  # Resume from specific batch number (uses existing checkpoints file)
    start_batch: int = 0  # Start batch index (used with load_checkpoint_path for new runs)


async def do_group_rollout_and_filter(
    sampling_client: tinker.SamplingClient,
    env_group_builder: EnvGroupBuilder,
    max_tokens: int,
    temperature: float,
    do_remove_constant_reward_groups: bool,
    top_p: float = 1.0,
    seed: int | None = None,
) -> TrajectoryGroup | None:
    """Perform rollouts for a group and optionally filter constant reward groups."""
    policy = KernelBenchTokenCompleter(
        sampling_client,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )

    trajectory_group = await do_group_rollout(env_group_builder, policy)

    # Filter if all rewards are the same
    if do_remove_constant_reward_groups:
        trajectory_groups = remove_constant_reward_groups([trajectory_group])
        if len(trajectory_groups) == 0:
            return None
        trajectory_group = trajectory_groups[0]

    return trajectory_group


async def do_group_rollout_with_envs(
    sampling_client: tinker.SamplingClient,
    env_group_builder: EnvGroupBuilder,
    max_tokens: int,
    temperature: float,
    do_remove_constant_reward_groups: bool,
    top_p: float = 1.0,
    seed: int | None = None,
) -> tuple[TrajectoryGroup | None, Sequence[Env] | None]:
    """Perform rollouts and return both trajectory group and env references.

    Replicates do_group_rollout to retain env refs needed for reading
    per-step scores during discounted return computation.
    """
    policy = KernelBenchTokenCompleter(
        sampling_client,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )

    envs = await env_group_builder.make_envs()
    rollout_results = await asyncio.gather(
        *[do_single_rollout(policy, env) for env in envs],
        return_exceptions=True,
    )

    trajectories = []
    valid_envs: list[Env] = []
    for traj, env in zip(rollout_results, envs):
        if isinstance(traj, Exception):
            logger.warning(f"Rollout failed: {traj}")
        else:
            trajectories.append(traj)
            valid_envs.append(env)

    if not trajectories:
        logger.warning("All rollouts in group failed")
        return None, None

    rewards_and_metrics = await env_group_builder.compute_group_rewards(
        trajectories, valid_envs
    )
    rewards_G, metrics_G = zip(*rewards_and_metrics, strict=True)

    trajectory_group = TrajectoryGroup(
        trajectories,
        list(rewards_G),
        list(metrics_G),
    )

    if do_remove_constant_reward_groups:
        trajectory_groups = remove_constant_reward_groups([trajectory_group])
        if len(trajectory_groups) == 0:
            return None, None
        trajectory_group = trajectory_groups[0]

    return trajectory_group, valid_envs


def apply_discounted_returns_to_trajectories(
    trajectory_groups: list[TrajectoryGroup],
    env_groups: list[Sequence[Env]],
    gamma: float,
    aggregation: str = "sum",
) -> None:
    """Replace per-step rewards with discounted returns for multi-turn training."""
    for tg, envs in zip(trajectory_groups, env_groups):
        for traj, env in zip(tg.trajectories_G, envs):
            if isinstance(env, MultiTurnKernelBenchEnv):
                step_scores = env.get_step_scores()
            else:
                step_scores = [t.reward for t in traj.transitions]

            if not step_scores:
                continue

            returns = compute_discounted_returns(step_scores, gamma, aggregation)
            for i, trans in enumerate(traj.transitions):
                if i < len(returns):
                    trans.reward = returns[i]


def flatten_multiturn_trajectory_groups(
    trajectory_groups: list[TrajectoryGroup],
) -> list[TrajectoryGroup]:
    """Flatten multi-turn trajectories so each turn is its own single-transition trajectory."""
    flattened = []
    for tg in trajectory_groups:
        new_trajectories = []
        for traj in tg.trajectories_G:
            for trans in traj.transitions:
                new_trajectories.append(
                    Trajectory(transitions=[trans], final_ob=tinker.ModelInput.empty())
                )

        # Rewards/metrics placeholders: real values live in each transition's
        # .reward (set by apply_discounted_returns) and .metrics fields.
        new_group = TrajectoryGroup(
            new_trajectories,
            [0.0] * len(new_trajectories),
            [{}] * len(new_trajectories),
        )
        flattened.append(new_group)
    return flattened


def compute_multiturn_advantages(
    trajectory_groups: list[TrajectoryGroup],
    num_turns: int = 1,
) -> list[torch.Tensor]:
    """Dr. GRPO advantage: subtract group mean, no std division.

    Expects flattened trajectory groups (each "trajectory" = one turn).
    """
    advantages_P = []
    for tg in trajectory_groups:
        rewards = torch.tensor(tg.get_total_rewards())
        # Dr. GRPO: subtract mean across ALL samples for this problem, no std division
        mean = rewards.mean()
        advantages = rewards - mean
        advantages_P.append(advantages)
    return advantages_P


def compute_multiturn_trajectory_metrics(
    trajectory_groups: list[TrajectoryGroup],
    env_groups: list[Sequence[Env]],
) -> dict[str, Any]:
    """Compute aggregate metrics for multi-turn trajectories."""
    metrics: dict[str, Any] = {}

    turn_compiled: dict[int, list[float]] = {}
    turn_correct: dict[int, list[float]] = {}

    all_rewards = []
    all_num_turns = []
    all_success = []
    all_best_speedup = []

    all_format_ok = []
    all_compiled = []
    all_correct = []
    all_step_scores = []
    all_eval_times = []
    all_step_times = []

    for tg, envs in zip(trajectory_groups, env_groups):
        rewards = tg.get_total_rewards()
        all_rewards.extend(rewards)

        for traj, env in zip(tg.trajectories_G, envs):
            traj_speedups = []

            for trans in traj.transitions:
                if trans.metrics:
                    turn = trans.metrics.get("turn", 0)
                    if turn not in turn_compiled:
                        turn_compiled[turn] = []
                        turn_correct[turn] = []

                    compiled = trans.metrics.get("compiled", 0)
                    correct = trans.metrics.get("correctness", 0)

                    turn_compiled[turn].append(compiled)
                    turn_correct[turn].append(correct)

                    all_format_ok.append(trans.metrics.get("format_ok", 0))
                    all_compiled.append(compiled)
                    all_correct.append(correct)

                    if "step_score" in trans.metrics:
                        all_step_scores.append(trans.metrics["step_score"])
                    if "time/eval" in trans.metrics:
                        all_eval_times.append(trans.metrics["time/eval"])
                    if "time/step_total" in trans.metrics:
                        all_step_times.append(trans.metrics["time/step_total"])
                    if "speedup" in trans.metrics:
                        traj_speedups.append(trans.metrics["speedup"])

            if traj_speedups:
                all_best_speedup.append(max(traj_speedups))

            if isinstance(env, MultiTurnKernelBenchEnv):
                all_success.append(float(env.state.success))
                all_num_turns.append(env.state.turn_idx)

    if all_rewards:
        metrics["reward/discounted_mean"] = float(np.mean(all_rewards))
        metrics["reward/discounted_std"] = float(np.std(all_rewards))
        metrics["reward/discounted_min"] = float(np.min(all_rewards))
        metrics["reward/discounted_max"] = float(np.max(all_rewards))

    if all_format_ok:
        metrics["multiturn/format_rate"] = float(np.mean(all_format_ok))
    if all_compiled:
        metrics["multiturn/compile_rate"] = float(np.mean(all_compiled))
    if all_correct:
        metrics["multiturn/correct_rate"] = float(np.mean(all_correct))
    if all_format_ok:
        failures = [1.0 - (f and c and r) for f, c, r in zip(all_format_ok, all_compiled, all_correct)]
        metrics["multiturn/failure_rate"] = float(np.mean(failures))
    if all_step_scores:
        metrics["multiturn/raw_score_mean"] = float(np.mean(all_step_scores))
    if all_success:
        metrics["multiturn/success_rate"] = float(np.mean(all_success))
    if all_num_turns:
        metrics["multiturn/avg_turns"] = float(np.mean(all_num_turns))
    if all_best_speedup:
        metrics["multiturn/best_speedup_mean"] = float(np.mean(all_best_speedup))
    if all_eval_times:
        metrics["time/eval_mean"] = float(np.mean(all_eval_times))
    if all_step_times:
        metrics["time/step_mean"] = float(np.mean(all_step_times))

    for turn in sorted(turn_compiled.keys()):
        if turn_compiled[turn]:
            metrics[f"multiturn/turn_{turn}/compile_rate"] = float(
                np.mean(turn_compiled[turn])
            )
        if turn_correct[turn]:
            metrics[f"multiturn/turn_{turn}/correct_rate"] = float(
                np.mean(turn_correct[turn])
            )

    metrics["batch/num_groups"] = len(trajectory_groups)
    metrics["batch/num_trajectories"] = sum(
        len(tg.trajectories_G) for tg in trajectory_groups
    )
    metrics["batch/total_steps"] = len(all_step_scores)

    return metrics


def compute_trajectory_metrics(
    trajectory_groups: list[TrajectoryGroup],
) -> dict[str, Any]:
    """Compute aggregate metrics from single-turn trajectory groups."""
    metrics: dict[str, Any] = {}

    all_rewards = []
    all_format_ok = []
    all_compiled = []
    all_correct = []
    all_eval_times = []
    all_step_times = []
    all_ref_load_times = []
    all_modal_eval_times = []

    for tg in trajectory_groups:
        rewards = tg.get_total_rewards()
        all_rewards.extend(rewards)

        # Extract per-trajectory metrics
        for traj in tg.trajectories_G:
            for trans in traj.transitions:
                if trans.metrics:
                    all_format_ok.append(trans.metrics.get("format_ok", 0))
                    all_compiled.append(trans.metrics.get("compiled", 0))
                    all_correct.append(trans.metrics.get("correctness", 0))
                    if "time/eval" in trans.metrics:
                        all_eval_times.append(trans.metrics["time/eval"])
                    if "time/step_total" in trans.metrics:
                        all_step_times.append(trans.metrics["time/step_total"])
                    if "time/ref_load" in trans.metrics:
                        all_ref_load_times.append(trans.metrics["time/ref_load"])
                    if "time/modal_eval" in trans.metrics:
                        all_modal_eval_times.append(trans.metrics["time/modal_eval"])

    if all_rewards:
        metrics["reward/mean"] = float(np.mean(all_rewards))
        metrics["reward/std"] = float(np.std(all_rewards))
        metrics["reward/min"] = float(np.min(all_rewards))
        metrics["reward/max"] = float(np.max(all_rewards))

    if all_format_ok:
        metrics["kernel/format_rate"] = float(np.mean(all_format_ok))
    if all_compiled:
        metrics["kernel/compile_rate"] = float(np.mean(all_compiled))
    if all_correct:
        metrics["kernel/correct_rate"] = float(np.mean(all_correct))
    if all_format_ok:
        failures = [1.0 - (f and c and r) for f, c, r in zip(all_format_ok, all_compiled, all_correct)]
        metrics["kernel/failure_rate"] = float(np.mean(failures))
    if all_eval_times:
        metrics["time/eval_mean"] = float(np.mean(all_eval_times))
        metrics["time/eval_max"] = float(np.max(all_eval_times))
    if all_step_times:
        metrics["time/step_mean"] = float(np.mean(all_step_times))
        metrics["time/step_max"] = float(np.max(all_step_times))
    if all_ref_load_times:
        metrics["time/ref_load_mean"] = float(np.mean(all_ref_load_times))
        metrics["time/ref_load_max"] = float(np.max(all_ref_load_times))
    if all_modal_eval_times:
        metrics["time/modal_eval_mean"] = float(np.mean(all_modal_eval_times))
        metrics["time/modal_eval_max"] = float(np.max(all_modal_eval_times))

    metrics["batch/num_groups"] = len(trajectory_groups)
    metrics["batch/num_trajectories"] = sum(
        len(tg.trajectories_G) for tg in trajectory_groups
    )

    return metrics



async def train_step(
    data: list[tinker.Datum],
    training_client: tinker.TrainingClient,
    learning_rate: float,
    num_substeps: int,
    loss_fn: LossFnType,
    max_grad_norm: float = 0.0,
    clip_epsilon_low: float = 0.0,
    clip_epsilon_high: float = 0.0,
) -> list[torch.Tensor]:
    """Perform a training step with gradient accumulation."""
    # Split data into substeps
    substep_size = max(1, len(data) // num_substeps)
    training_logprobs = []

    for i in range(0, len(data), substep_size):
        batch = data[i : i + substep_size]

        # Prepare datums: remove mask, add PPO clip thresholds if configured
        prepared = [
            prepare_datum(d, loss_fn, clip_epsilon_low, clip_epsilon_high)
            for d in batch
        ]

        # Forward-backward pass
        fwd_bwd_future = await training_client.forward_backward_async(
            prepared, loss_fn=loss_fn
        )
        fwd_bwd_result = await fwd_bwd_future.result_async()

        # Extract logprobs
        for output in fwd_bwd_result.loss_fn_outputs:
            training_logprobs.append(output["logprobs"].to_torch())

        # Optimizer step
        adam_params = get_adam_params(learning_rate, max_grad_norm=max_grad_norm)
        optim_future = await training_client.optim_step_async(adam_params)
        await optim_future.result_async()

    return training_logprobs


async def save_checkpoint_and_get_sampling_client(
    training_client: tinker.TrainingClient,
    batch_idx: int,
    log_path: str,
    save_every: int,
    start_batch: int = 0,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    """Save checkpoint and return updated sampling client."""
    metrics: dict[str, Any] = {}

    with timed("save_checkpoint", metrics):
        # Always save checkpoint: at start (batch_idx == start_batch) or every save_every batches
        # This ensures we can resume even if training crashes early
        should_save = (
            save_every > 0 and (
                batch_idx == start_batch or  # Initial checkpoint
                (batch_idx > start_batch and batch_idx % save_every == 0)  # Periodic checkpoints
            )
        )
        if should_save:
            path_dict = await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{batch_idx:06d}",
                log_path=log_path,
                loop_state={"batch": batch_idx},
                kind="both",
            )
            return training_client.create_sampling_client(path_dict["sampler_path"]), metrics
        else:
            return await training_client.save_weights_and_get_sampling_client_async(), metrics


async def run_training_loop(
    cfg: TrainingConfig,
) -> None:
    """Main RL training loop for KernelBench (single-turn and multi-turn)."""
    is_multiturn = cfg.multiturn.enabled
    if is_multiturn:
        logger.info("Running in MULTI-TURN mode")
        logger.info(f"  n (refinement turns per trajectory): {cfg.multiturn.n}")
        logger.info(f"  group_size (parallel trajectories, m): {cfg.dataset_builder.group_size}")
        logger.info(f"  gamma (discount factor): {cfg.multiturn.gamma}")

    # Setup logging
    os.makedirs(cfg.log_path, exist_ok=True)
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )

    # Setup TensorBoard logging
    tb_logger: TensorBoardLogger | None = None
    if cfg.tensorboard_enabled:
        tb_config = TensorBoardConfig(
            log_histograms_every=cfg.tensorboard_log_histograms_every,
            log_per_level_metrics=cfg.tensorboard_log_per_level,
        )
        tb_logger = create_tensorboard_logger(cfg.log_path, tb_config)
        tb_logger.log_training_config(cfg)

    logger.info("Starting KernelBench RL training")
    logger.info(f"Multi-turn: {'enabled' if is_multiturn else 'disabled'}")
    logger.info(f"Model: {cfg.model_name}")
    logger.info(f"Log path: {cfg.log_path}")
    if tb_logger:
        logger.info(f"TensorBoard: {tb_logger.log_dir}")

    # Set up trace logger (per-step prompts/outputs/evals)
    trace_logger = TraceLogger(Path(cfg.log_path) / "traces.jsonl")
    set_trace_logger(trace_logger)

    # Check for resume
    if cfg.resume_from_batch is not None:
        # Resume from a specific batch number (uses existing checkpoints file)
        checkpoints = checkpoint_utils.load_checkpoints_file(cfg.log_path)
        resume_info = None
        for ckpt in checkpoints:
            if ckpt.get("batch") == cfg.resume_from_batch and "state_path" in ckpt:
                resume_info = ckpt
                break
        if resume_info:
            start_batch = resume_info["batch"]
            logger.info(f"Resuming from specific batch {start_batch}")
        else:
            raise ValueError(f"No checkpoint found for batch {cfg.resume_from_batch}")
    elif cfg.load_checkpoint_path:
        # Starting new run from external checkpoint
        resume_info = None  # Don't use optimizer state from checkpoints file
        start_batch = cfg.start_batch
        logger.info(f"Starting new run from external checkpoint at batch {start_batch}")
        logger.info(f"  Checkpoint: {cfg.load_checkpoint_path}")
    else:
        # Resume from last checkpoint
        resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
        if resume_info:
            start_batch = resume_info["batch"]
            logger.info(f"Resuming from batch {start_batch}")
        else:
            start_batch = cfg.start_batch

    # Create Tinker clients
    service_client = tinker.ServiceClient(base_url=cfg.base_url)

    if resume_info:
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info["state_path"]
            )
        )
    elif cfg.load_checkpoint_path:
        training_client = await service_client.create_training_client_from_state_async(
            cfg.load_checkpoint_path
        )
    else:
        training_client = await service_client.create_lora_training_client_async(
            cfg.model_name, rank=cfg.lora_rank
        )

    tokenizer = training_client.get_tokenizer()

    # Create dataset (pass tokenizer for renderer)
    if is_multiturn:
        # Override multi-turn fields from MultiTurnConfig onto the dataset builder
        dataset_builder = chz.replace(
            cfg.dataset_builder,
            max_turns=cfg.multiturn.n,
            early_stop_on_correct=cfg.multiturn.early_stop_on_correct,
            speedup_threshold=cfg.multiturn.speedup_threshold,
        )
        logger.info("Using KernelBenchDatasetBuilder (multi-turn, max_turns=%d)", cfg.multiturn.n)
    else:
        dataset_builder = cfg.dataset_builder
        logger.info("Using KernelBenchDatasetBuilder (single-turn)")

    train_dataset, test_dataset = await dataset_builder(tokenizer=tokenizer)
    num_batches = len(train_dataset)
    logger.info(f"Training on {num_batches} batches")

    # Warmup schedule
    warmup_batches = int(num_batches * cfg.warmup_ratio)
    if warmup_batches > 0:
        logger.info(f"Linear LR warmup for {warmup_batches} batches")

    # Get initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, cfg.log_path, cfg.save_every, start_batch
    )

    # Training loop
    for batch_idx in range(start_batch, num_batches):
        t_start = time.time()
        metrics: dict[str, Any] = {
            "progress/batch": batch_idx,
            "progress/done_frac": (batch_idx + 1) / num_batches,
            "mode": 1 if is_multiturn else 0,
        }

        # Get batch of env group builders
        env_group_builders = train_dataset.get_batch(batch_idx)

        # Response length extension
        effective_max_tokens = cfg.max_tokens
        if (
            cfg.max_tokens_extended > 0
            and batch_idx >= cfg.max_tokens_extend_after_step
        ):
            effective_max_tokens = cfg.max_tokens_extended
            if batch_idx == cfg.max_tokens_extend_after_step:
                logger.info(
                    f"Extending max_tokens from {cfg.max_tokens} to "
                    f"{cfg.max_tokens_extended} at step {batch_idx}"
                )

        if is_multiturn:
            # ----- Multi-turn rollouts -----
            with timed("rollout", metrics):
                try:
                    results = await asyncio.gather(
                        *[
                            do_group_rollout_with_envs(
                                sampling_client,
                                builder,
                                max_tokens=effective_max_tokens,
                                temperature=cfg.temperature,
                                do_remove_constant_reward_groups=cfg.remove_constant_reward_groups,
                                top_p=cfg.top_p,
                                seed=cfg.seed,
                            )
                            for builder in env_group_builders
                        ],
                        return_exceptions=True,
                    )
                except Exception:
                    logger.exception("Group rollout failed during gather")
                    raise

            trajectory_groups: list[TrajectoryGroup] = []
            env_groups: list[Sequence[Env]] = []
            for r in results:
                if isinstance(r, Exception):
                    logger.error("Group rollout failed", exc_info=r)
                    continue
                tg, envs = r
                if tg is not None:
                    trajectory_groups.append(tg)
                    env_groups.append(envs)

            if len(trajectory_groups) == 0:
                logger.warning(
                    f"Batch {batch_idx}: All groups filtered out, skipping"
                )
                continue

            with timed("discount_returns", metrics):
                apply_discounted_returns_to_trajectories(
                    trajectory_groups, env_groups,
                    gamma=cfg.multiturn.gamma,
                    aggregation=cfg.multiturn.aggregation,
                )

            traj_metrics = compute_multiturn_trajectory_metrics(
                trajectory_groups, env_groups
            )
            metrics.update(traj_metrics)

            # Flatten: each turn becomes its own single-transition trajectory
            # so that advantage normalization is across all m*n turn-level samples
            with timed("flatten", metrics):
                trajectory_groups = flatten_multiturn_trajectory_groups(
                    trajectory_groups
                )
        else:
            # ----- Single-turn rollouts (original path) -----
            with timed("rollout", metrics):
                try:
                    results = await asyncio.gather(
                        *[
                            do_group_rollout_and_filter(
                                sampling_client,
                                builder,
                                max_tokens=effective_max_tokens,
                                temperature=cfg.temperature,
                                do_remove_constant_reward_groups=cfg.remove_constant_reward_groups,
                                top_p=cfg.top_p,
                                seed=cfg.seed,
                            )
                            for builder in env_group_builders
                        ],
                        return_exceptions=True,
                    )
                except Exception:
                    logger.exception("Group rollout failed during gather")
                    raise

            trajectory_groups = []
            for tg in results:
                if isinstance(tg, Exception):
                    logger.error("Group rollout failed", exc_info=tg)
                elif tg is not None:
                    trajectory_groups.append(tg)

            if len(trajectory_groups) == 0:
                logger.warning(
                    f"Batch {batch_idx}: All groups filtered out, skipping"
                )
                continue

            traj_metrics = compute_trajectory_metrics(trajectory_groups)
            metrics.update(traj_metrics)

        # Compute advantages and assemble training data
        with timed("assemble_data", metrics):
            if is_multiturn:
                # Dr. GRPO (group_norm): A_t = R_t - mean(R)
                # across all m*n flattened turn-level samples, no std division
                advantages = compute_multiturn_advantages(
                    trajectory_groups, num_turns=cfg.multiturn.n
                )
            else:
                advantages = compute_advantages(trajectory_groups)

            # Constant length normalization (Dr. GRPO).
            # Tinker sums per-token losses, so scaling advantages by
            # 1/C gives each response gradient magnitude proportional to
            # |o_i|/C instead of |o_i|.
            if cfg.constant_length_norm > 0:
                for i in range(len(advantages)):
                    advantages[i] = advantages[i] / cfg.constant_length_norm

            data, _metadata = assemble_training_data(trajectory_groups, advantages)

        # Compute effective learning rate (linear warmup)
        if warmup_batches > 0 and batch_idx < warmup_batches:
            lr = cfg.learning_rate * (batch_idx + 1) / warmup_batches
        else:
            lr = cfg.learning_rate
        metrics["optim/lr"] = lr

        # Training step
        with timed("train", metrics):
            await train_step(
                data,
                training_client,
                lr,
                cfg.num_substeps,
                cfg.loss_fn,
                max_grad_norm=cfg.max_grad_norm,
                clip_epsilon_low=cfg.clip_epsilon_low,
                clip_epsilon_high=cfg.clip_epsilon_high,
            )

        # Save checkpoint and get new sampling client
        sampling_client, checkpoint_metrics = (
            await save_checkpoint_and_get_sampling_client(
                training_client, batch_idx + 1, cfg.log_path, cfg.save_every
            )
        )
        metrics.update(checkpoint_metrics)

        # Log metrics
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=batch_idx)

        # TensorBoard logging
        if tb_logger:
            tb_logger.log_training_metrics(metrics, batch_idx)
            tb_logger.log_trajectory_histograms(trajectory_groups, batch_idx)
            tb_logger.log_per_level_metrics(trajectory_groups, batch_idx)
            tb_logger.log_advantage_statistics(advantages, batch_idx)

        if is_multiturn:
            logger.info(
                f"Batch {batch_idx}/{num_batches}: "
                f"raw_score={metrics.get('multiturn/raw_score_mean', 0):.3f}, "
                f"compile={metrics.get('multiturn/compile_rate', 0):.1%}, "
                f"correct={metrics.get('multiturn/correct_rate', 0):.1%}, "
                f"success={metrics.get('multiturn/success_rate', 0):.1%}, "
                f"avg_turns={metrics.get('multiturn/avg_turns', 0):.1f}"
            )
        else:
            logger.info(
                f"Batch {batch_idx}/{num_batches}: "
                f"reward={metrics.get('reward/mean', 0):.3f}, "
                f"compile={metrics.get('kernel/compile_rate', 0):.1%}, "
                f"correct={metrics.get('kernel/correct_rate', 0):.1%}"
            )

    # Save final checkpoint
    if start_batch < num_batches:
        await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": num_batches},
        )

    # Close loggers
    if tb_logger:
        tb_logger.flush()
        tb_logger.close()
    ml_logger.close()
    logger.info("Training completed!")


async def main(cfg: TrainingConfig) -> None:
    """Entry point for training."""
    await run_training_loop(cfg)
