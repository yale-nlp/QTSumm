export PYTHONPATH=$PYTHONPATH:$(pwd);
export CUDA_VISIBLE_DEVICES=0,1; 
python finetuning/main.py \
    --yaml_file finetuning/yamls/train.yaml \
    --model_name_or_path neulab/omnitab-large