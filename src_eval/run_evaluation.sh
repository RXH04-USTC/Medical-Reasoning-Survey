#!/bin/bash

# Combined evaluation script
# This script starts the vLLM server and then runs evaluation after a 500-second wait

# Set GPU devices (as requested)
export CUDA_VISIBLE_DEVICES=1,2,3,4

# Set log_num to 0 (as in original vllm_server.sh)
log_num=0
echo "Using log_num: $log_num"

# Port number based on log_num (keeping original logic)
port=28${log_num}35

# Model path
model_name="YOUR_MODEL_PATH_HERE"  # Replace with your actual model path

# Start vLLM server in background
echo "Starting vLLM server on port $port..."
python -m vllm.entrypoints.openai.api_server \
    --model $model_name \
    --port $port \
    --gpu-memory-utilization 0.9 \
    --seed 42 \
    --tensor-parallel-size 4 > vllm_server_${log_num}.log 2>&1 &

# Save the server process PID for potential cleanup
SERVER_PID=$!
echo "vLLM server started with PID: $SERVER_PID"

# Wait for server to start and stabilize (500 seconds as requested)
echo "Waiting 300 seconds for vLLM server to stabilize before starting evaluation..."
sleep 500

# Benchmark directory and file list
bench_dir="YOUR_BENCHMARK_DIRECTORY_HERE"  # Replace with your actual benchmark directory
bench_files=(
    "GPQA_Medical_test.jsonl"
    "JMED.jsonl"
    "medmcqa.jsonl"
    "medqa_5options.jsonl"
    "medxpertqa_test_text.jsonl"
    "mmlu_pro_bio_health.jsonl"
    "pubmedqa.jsonl"
    "ReDis-QA.jsonl"
    "mimic_iv_med.jsonl"
    "mimic_iv_procedure.jsonl"
)

TEMPERATURE=0.5
USE_CHAT_TEMPLATE=true
STRICT_PROMPT=False

# Run evaluation for each mode and benchmark file
for mode in "${modes[@]}"; do
    for file_name in "${bench_files[@]}"; do
        
        # Extract benchmark name from filename
        benchmark_name=$(basename "$file_name" .jsonl)
        
        # Build local file path
        local_file_path="$bench_dir/$file_name"

        echo "-----------------------------------------------------"
        echo "Running: $benchmark_name (Mode: $mode)"
        echo "File: $local_file_path"
        echo "-----------------------------------------------------"

        cmd="python eval.py \
                --eval_dataset $local_file_path \
                --eval_benchmark $benchmark_name \
                --port $port \
                --batch_size 4 \
                --max_new_tokens 16000 \
                --temperature $TEMPERATURE"

        if [ "$STRICT_PROMPT" == "true" ]; then
            cmd="$cmd --strict_prompt"
        fi

        if [ "$mode" == "reasoning" ]; then
            cmd="$cmd --reasoning"
        fi

        if [ "$USE_CHAT_TEMPLATE" == "true" ]; then
            cmd="$cmd --use_chat_template"
        fi

        eval $cmd
    done
done

echo "Evaluation completed. vLLM server is still running with PID: $SERVER_PID"
echo "To stop the server, run: kill $SERVER_PID"