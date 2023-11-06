export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=1,2,3,4

# List of models
models=(
  "yale-nlp/flan-t5-large-finetuned-qtsumm"
  "yale-nlp/t5-large-finetuned-qtsumm"
  "yale-nlp/bart-large-finetuned-qtsumm"
  "yale-nlp/tapex-large-finetuned-qtsumm"
  "yale-nlp/reastap-large-finetuned-qtsumm"
  "yale-nlp/omnitab-large-finetuned-qtsumm"
)

# Loop over all models
for model in "${models[@]}"; do
    echo "Running inference for model: $model"
    
    # Run the Python command
    python finetuning/main.py \
        --yaml_file finetuning/yamls/inference.yaml \
        --model_name_or_path "$model"
done