import json
from utils.evaluation_utils import *
import argparse
from datasets import load_dataset
import os
from evaluation.evaluation_single_file import evaluate_prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="yale-nlp/QTSumm")
    parser.add_argument("--split_name", type=str, default="test")
    
    args = parser.parse_args()

    
    output_data = []
    for prediction_file in os.listdir(args.prediction_dir):
        if not prediction_file.endswith(".json") or "score" in prediction_file:
            continue
        model_name = ".".join(prediction_file.split("/")[-1].split(".")[:-1])
        print(model_name)
        prediction_path = os.path.join(args.prediction_dir, prediction_file)
        all_scores = evaluate_prediction(prediction_path, args.dataset_name, args.split_name)
        
        scores = {"model_name": model_name}
        for metric in all_scores:
            scores[metric] = all_scores[metric]
        output_data.append(scores)
    
    output_data = sorted(output_data, key=lambda x: x["AutoACU"], reverse=True)
    
     
    output_dir = args.prediction_dir
    output_path = os.path.join(output_dir, "scores.json")
    json.dump(output_data, open(output_path, "w"), indent=4)