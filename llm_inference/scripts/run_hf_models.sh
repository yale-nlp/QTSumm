export CUDA_VISIBLE_DEVICES=1,3,4,5;
export PYTHONPATH=$PYTHONPATH:$(pwd)
models=(
  "TheBloke/Llama-2-70B-chat-AWQ"
)

for model in "${models[@]}"; do
    echo "Running inference for model: $model"

    python llm_inference/hf_model_main.py \
        --model_name_or_path "$model" \
        --num_shot 1
done