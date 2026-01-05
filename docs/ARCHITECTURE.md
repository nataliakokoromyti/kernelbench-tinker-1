# KernelBench-Tinker Architecture

This document explains how the integration works for training LLMs to write optimized GPU kernels.

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           YOUR MACHINE (Training)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐     ┌──────────────────┐     ┌────────────────────┐      │
│   │  YAML Config │────▶│  Training Loop   │────▶│  RL Environment    │      │
│   │              │     │  (loop.py)       │     │  (kernelbench_env) │      │
│   └──────────────┘     └────────┬─────────┘     └─────────┬──────────┘      │
│                                 │                         │                  │
│                                 ▼                         ▼                  │
│                        ┌───────────────┐         ┌───────────────────┐      │
│                        │  Tinker API   │         │  Modal Client     │      │
│                        │  (sampling)   │         │  (evaluator.py)   │      │
│                        └───────┬───────┘         └─────────┬─────────┘      │
│                                │                           │                 │
└────────────────────────────────┼───────────────────────────┼─────────────────┘
                                 │                           │
                    ─────────────┼───────────────────────────┼─────────────────
                                 ▼                           ▼
┌────────────────────────────────────────┐   ┌─────────────────────────────────┐
│           TINKER CLOUD                 │   │         MODAL CLOUD              │
├────────────────────────────────────────┤   ├─────────────────────────────────┤
│                                        │   │                                  │
│  ┌─────────────────────────────────┐   │   │  ┌─────────────────────────┐    │
│  │  LLM (Qwen/Llama + LoRA)        │   │   │  │  GPU Container (A100)   │    │
│  │  • Generates kernel code        │   │   │  │  • Runs KernelBench     │    │
│  │  • Distributed across 8 GPUs    │   │   │  │  • Isolated execution   │    │
│  └─────────────────────────────────┘   │   │  │  • Up to 32 parallel    │    │
│                                        │   │  └─────────────────────────┘    │
└────────────────────────────────────────┘   └─────────────────────────────────┘
```

## Training Flow

### 1. Batch Setup
```python
# Each batch: 8 problems × 16 rollouts = 128 kernel generations
env_group_builders = train_dataset.get_batch(batch_idx)
```

### 2. LLM Sampling (Tinker)
```python
# Tinker generates 128 kernel implementations in parallel
trajectory_group = await do_group_rollout(env_group_builder, policy)
```

### 3. Kernel Evaluation (Modal)
```python
# Each generated kernel is evaluated on Modal GPUs
eval_result = await evaluate_kernel_async(
    level=problem.level,
    problem_id=problem.problem_id,
    kernel_code=kernel_code,
    ...
)
```

### 4. Reward Computation
```python
# Reward = 0.3 × correctness + speedup (if correct)
reward = compute_reward(eval_result, reward_config)
```

### 5. Model Update (GRPO)
```python
# Advantages normalized within each problem group
advantages = compute_advantages(trajectory_groups)
await train_step(data, training_client, ...)
```

## Component Responsibilities

### envs/kernelbench_client.py
- Wraps KernelBench Python API
- Fetches prompts from HuggingFace dataset (or local files)
- Parses model responses to extract `<KERNEL>` blocks
- Routes evaluation to Modal

### envs/kernelbench_env.py
- Implements Tinker's `Env` interface
- `KernelBenchEnv`: Single problem, single rollout
- `KernelBenchEnvGroupBuilder`: Groups rollouts for GRPO
- `KernelBenchRLDataset`: Batches problems for training
- `KernelBenchDatasetBuilder`: Config → Dataset factory

### modal/app.py
**Runs on Modal (remote GPU containers)**
- Defines the Modal App and Docker image
- `KernelEvaluator.evaluate()`: Calls `kernelbench.eval.eval_kernel_against_ref()`
- Handles GPU errors with `modal.experimental.stop_fetching_inputs()`

### modal/evaluator.py
**Runs on your machine (client)**
- `ModalKernelEvaluator`: Client to call deployed Modal app
- `ModalBatcher`: Coalesces single evals into batched `starmap()` calls
- Handles retries and exceptions

### training/loop.py
- Main GRPO training loop
- Uses `tinker_cookbook.rl` utilities for rollouts and advantages
- Checkpoints after every batch for crash recovery

### training/reward.py
- `RewardConfig`: Weights for format, compile, correctness, speed
- `compute_reward()`: Total reward calculation
- Formula: `reward = 0.3 × correct + speedup` (when correct)

## Modal Batcher Design

The `ModalBatcher` exists to bridge two different execution patterns:

### The Problem

```
Tinker RL Interface:          Modal Efficiency:
─────────────────────         ──────────────────
Each Env.step() is            starmap() is more efficient
independent and async         with batched calls
      │                              │
      ▼                              ▼
128 separate calls            1 call with 128 items
(high RPC overhead)           (low RPC overhead)
```

### How the Batcher Works

```
Time ─────────────────────────────────────────▶

env.step() ──┐
env.step() ──┼──► Batcher collects for 150ms ──► starmap([all evals])
env.step() ──┤                                        │
   ...       │                                        ▼
env.step() ──┘                                  Modal containers
                                                run in parallel
```

1. Each `Env.step()` calls `evaluate_single_batched()` independently
2. `ModalBatcher` collects requests for a short window (150ms)
3. When window expires OR batch is full (32), flush to Modal
4. One `starmap()` call dispatches all collected evals
5. Results are distributed back to waiting callers

### Trade-offs

| Benefit | Cost |
|---------|------|
| Fewer RPC calls | 150ms latency waiting for batch |
| Less connection overhead | ~50 lines of async complexity |

### Alternative (Not Implemented)

Could remove the batcher and batch at training loop level:
```python
# Collect all kernels first
all_kernels = collect_kernels_from_rollouts(rollouts)
# One Modal call
all_results = await modal_evaluator.evaluate_batch(all_kernels)
# Distribute results
assign_results_to_envs(rollouts, all_results)
```
This would be simpler but requires changing the Tinker Env interface.

## Data Flow

```mermaid
sequenceDiagram
    participant Config as YAML Config
    participant Loop as Training Loop
    participant Env as RL Environment
    participant Tinker as Tinker Cloud
    participant Modal as Modal Cloud
    participant KB as KernelBench

    Config->>Loop: Load configuration
    Loop->>Env: Create dataset & envs
    
    loop Each Batch
        Loop->>Tinker: Request LLM completions
        Tinker-->>Loop: Generated kernel code
        
        Loop->>Env: env.step(kernel_code)
        Env->>Modal: evaluate_kernel_async()
        Modal->>KB: eval_kernel_against_ref()
        KB-->>Modal: compile, correctness, speedup
        Modal-->>Env: KernelEvalResult
        
        Env->>Env: compute_reward()
        Env-->>Loop: StepResult(reward, metrics)
        
        Loop->>Loop: compute_advantages()
        Loop->>Tinker: Update model weights
    end
```

## Configuration

All configuration flows from a single YAML file:

```yaml
# Model
model_name: "Qwen/Qwen3-30B-A3B"
lora_rank: 64
learning_rate: 0.000002

# Dataset
dataset_builder:
  level: 1                    # KernelBench level (1-4)
  batch_size: 8               # Problems per batch
  group_size: 16              # Rollouts per problem

# Evaluation (passed to Modal)
  num_correct_trials: 5
  measure_performance: true
  precision: "fp32"
  modal_gpu_type: "A100"
  modal_timeout: 60

# Rewards
  reward_correctness_weight: 0.3
  reward_speed_weight: 1.0
```

## Key Design Decisions

### Why Modal for Evaluation?
- **Isolation**: Bad kernels can corrupt GPU memory; Modal containers are disposable
- **Parallelism**: Up to 32 containers evaluate kernels simultaneously
- **Hard timeouts**: Infinite loops are killed by Modal, not your machine

### Why Tinker for Training?
- **Distributed LoRA**: Fine-tune 30B+ models across multiple GPUs
- **GRPO implementation**: Grouped rollouts with advantage normalization
- **Checkpointing**: Resume from any batch after crashes

### Why HuggingFace Dataset?
- **No local files needed**: Clone repo and run immediately
- **Reproducibility**: Everyone uses same problem definitions
- **Custom curriculum**: Can still add local problems via `dataset_src="local"`

## File Structure

```
src/kernelbench_tinker/
├── config/
│   └── rl_kernelbench.yaml    # Default training config
├── envs/
│   ├── kernelbench_client.py  # KB API wrapper + parsing
│   └── kernelbench_env.py     # Tinker Env implementations
├── modal/
│   ├── app.py                 # Remote: Modal App definition
│   └── evaluator.py           # Local: Modal client + batching
├── training/
│   ├── loop.py                # GRPO training loop
│   ├── reward.py              # Reward computation
│   ├── models.py              # Model/optimizer config
│   └── tensorboard_logger.py  # Metrics logging
└── scripts/
    ├── train_kernel_rl.py     # Training entrypoint
    └── eval_kernel_rl.py      # Evaluation entrypoint
```
