#!/bin/bash

# =============================================================================
# Base Model Evaluations - R2 1B Models
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

export WANDB_API_KEY="${WANDB_API_KEY:-5c696848f3dee351ef64414ecda98950697dbef4}"
# WANDB_PROJECT="jain-2b-monolithic-27jan2026" #"r2-1b-base-evals-jain-14jan"
WANDB_PROJECT="jain-2b-monolithic-feb1" 

BASE_HOST="172.18.133.131"
OUTPUT_BASE_PATH="results_base_evals"

# Evaluation settings
NUM_CONCURRENT="${NUM_CONCURRENT:-64}"
MAX_RETRIES="${MAX_RETRIES:-30}"
MAX_LENGTH="${MAX_LENGTH:-130000}"
TIMEOUT="${TIMEOUT:-6000}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# Model endpoints
declare -A MODEL_ENDPOINTS=(

    # ["32k-mono-ccv1-gpt5-glm45-3000"]="32261"
    # ["32k-mono-ccv1-gpt5-glm45-9000"]="31194"
    # ["32k-mono-ccv1-gpt5-glm45-15000"]="30515"
    # ["32k-mono-ccv2-all-models-s9000"]="32545"
    # ["32k-mono-ccv2-all-models-15000"]="32297"
    # ["32k-mono-ccv2-all-models-24000"]="32091"

    # ["32k-mono-ccv2-glm45-gpt5mini-s9000"]="30863"
    ["32k-mono-ccv2-glm45-gpt5mini-s15000"]="31347"
    ["32k-mono-ccv2-glm45-gpt5mini-s24000"]="30329"



)

# Tasks configuration (task_name:num_fewshot)
# Format: "task_name:fewshot" - use 5 as default if not specified
TASKS=(
    "ruler:0"
)

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

check_eval_completed() {
    local output_path="$1"
    # Check if results.json exists (lm-eval creates this on completion)
    if ls "${output_path}"*/results.json 1>/dev/null 2>&1; then
        return 0  # Completed
    fi
    return 1  # Not completed
}

run_eval() {
    local model_name="$1"
    local task_config="$2"
    local port="${MODEL_ENDPOINTS[$model_name]}"
    
    # Parse task config
    local task="${task_config%%:*}"
    local num_fewshot="${task_config##*:}"
    
    local base_url="http://${BASE_HOST}:${port}/v1/completions"
    local output_path="${OUTPUT_BASE_PATH}/${model_name}/${task}/"
    local wandb_name="${model_name}-${task}"

    # Check if already completed
    if check_eval_completed "$output_path"; then
        echo "⏭ Skipping: $model_name - $task (already completed)"
        return 0
    fi

    echo "----------------------------------------"
    echo "Model: $model_name | Task: $task"
    echo "Port: $port | Fewshot: $num_fewshot"
    echo "Output: $output_path"
    echo "----------------------------------------"

    mkdir -p "$output_path"

    local cmd="python -m lm_eval \
        --model local-completions \
        --tasks $task \
        --num_fewshot $num_fewshot \
        --output_path $output_path \
        --seed 42 \
        --batch_size $BATCH_SIZE \
        --model_args model=meta-llama/Meta-Llama-3-8B,base_url=${base_url},num_concurrent=${NUM_CONCURRENT},max_retries=${MAX_RETRIES},tokenized_requests=False,max_length=${MAX_LENGTH},timeout=${TIMEOUT},add_bos=True,add_eos=True,use_special=True,tokenizer=meta-llama/Meta-Llama-3-8B \
        --log_samples \
        --wandb_args project=${WANDB_PROJECT},name=${wandb_name},group=${model_name}"

    echo "Running command:"
    echo "$cmd"

    eval $cmd
}


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

echo "============================================"
echo "Base Model Evaluations - R2 1B"
echo "============================================"
echo "Models:"
for model in "${!MODEL_ENDPOINTS[@]}"; do
    echo "  - $model (port: ${MODEL_ENDPOINTS[$model]})"
done
echo ""
echo "Tasks:"
for task_config in "${TASKS[@]}"; do
    echo "  - ${task_config%%:*} (fewshot: ${task_config##*:})"
done
echo "============================================"
echo ""

# Create base output directory
mkdir -p "$OUTPUT_BASE_PATH"

# Run evaluations
for model_name in "${!MODEL_ENDPOINTS[@]}"; do
    echo ""
    echo "============================================"
    echo "Starting evaluations for: $model_name"
    echo "============================================"
    
    for task_config in "${TASKS[@]}"; do
        if run_eval "$model_name" "$task_config"; then
            echo "✓ Completed: $model_name - ${task_config%%:*}"
        else
            echo "✗ Failed: $model_name - ${task_config%%:*}"
        fi
        echo ""
    done
done

echo ""
echo "============================================"
echo "All base evaluations completed!"
echo "============================================"


