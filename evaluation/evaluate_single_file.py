import json
from utils.evaluation_utils import *
import argparse
from datasets import load_dataset
import os

def evaluate_prediction(json_path, dataset_name, split_name):
    qa_data = load_dataset(dataset_name, split =split_name)
    reference_dict = {}
    for qa in qa_data:
        example_id = qa["example_id"]
        reference_dict[example_id] = qa["summary"]
    
    prediction_data = json.load(open(json_path, "r"))
    predictions, references = [], []
    
    for prediction in prediction_data:
        example_id = prediction["example_id"]
        predictions.append(prediction["prediction"].lower())
        references.append(reference_dict[example_id].lower())
        prediction["reference"] = reference_dict[example_id]

    all_scores = get_all_scores(predictions, references, json_path, dataset_name, split_name)
    return all_scores
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="yale-nlp/QTSumm")
    parser.add_argument("--split_name", type=str, default="test")
    
    args = parser.parse_args()
    
    all_scores = evaluate_prediction(args.prediction_path, args.dataset_name, args.split_name)
    
    print("Finished evaluating {}.".format(args.prediction_path))
    print(all_scores)