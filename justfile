# KernelBench ↔ Tinker Integration Commands

run_name := "run_" + `date +%Y%m%d_%H%M%S`
config := "src/kernelbench_tinker/config/rl_kernelbench.yaml"
runs_dir := "./runs"

default:
    @just --list

# === Training ===

train run=run_name:
    @mkdir -p {{runs_dir}}
    @echo "Starting training: {{run}}"
    nohup uv run python -m kernelbench_tinker.scripts.train_kernel_rl \
        --config {{config}} \
        log_path={{runs_dir}}/{{run}} \
        > {{runs_dir}}/{{run}}_nohup.log 2>&1 &
    @sleep 2
    @pgrep -f "log_path={{runs_dir}}/{{run}}" > /dev/null && echo "Training started (PID: $$(pgrep -f 'log_path={{runs_dir}}/{{run}}'))" || echo "Failed to start"
    @echo "Logs: {{runs_dir}}/{{run}}/logs.log"

resume run:
    @echo "Resuming training: {{run}}"
    nohup uv run python -m kernelbench_tinker.scripts.train_kernel_rl \
        --config {{config}} \
        log_path={{runs_dir}}/{{run}} \
        load_checkpoint_path={{runs_dir}}/{{run}} \
        > {{runs_dir}}/{{run}}_nohup.log 2>&1 &
    @sleep 2
    @pgrep -f "log_path={{runs_dir}}/{{run}}" > /dev/null && echo "Training resumed" || echo "Failed to start"

# === Monitoring ===

logs run:
    @tail -f {{runs_dir}}/{{run}}/logs.log

status:
    @echo "=== Running Training Jobs ==="
    @pgrep -fa "train_kernel_rl" | grep -v grep || echo "No training jobs running"

# === Control ===

stop run:
    @echo "Stopping training: {{run}}"
    @pkill -f "log_path={{runs_dir}}/{{run}}" && echo "Training stopped" || echo "No matching process found"

stop-all:
    @echo "Stopping all training jobs..."
    @pkill -f "train_kernel_rl" && echo "All training jobs stopped" || echo "No training jobs running"
