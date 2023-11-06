models=(
  "gpt-35-turbo"
  "gpt-4"
)

OPENAI_API_KEY=xxxxxx
export PYTHONPATH=$PYTHONPATH:$(pwd)

for model in "${models[@]}"; do
    python llm_inference/openai_main.py \
        --model_name_or_path "$model" \
        --api_key $OPENAI_API_KEY \
        --num_shot 1
done