
# vllm serve saves/qwen3-vl-dpo-merged \
vllm serve saves/qwen3-vl-dpo-merged \
    --dtype auto \
    --host 0.0.0.0 \
    --port 8000 \
    --max_model_len 32768 \
    --served-model-name qwen3-2b \
    --gpu-memory-utilization 0.5
