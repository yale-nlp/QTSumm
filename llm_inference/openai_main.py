import json
import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.openai_utils import *
from utils.llm_preprocess_utils import *
import openai
import asyncio

def prepare_model_input(prompt_text, user_turns = [], assistant_turns = []):
    chat = [{"role": "system", "content": system_prompt}]
    for user_turn, assistant_turn in zip(user_turns, assistant_turns):
        chat.append({"role": "user", "content": user_turn})
        chat.append({"role": "assistant", "content": assistant_turn})
    chat.append({"role": "user", "content": prompt_text})
    return chat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--api_key", type=str, default=None, required=True)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--num_shot", type=int, default=None, required=True)
    parser.add_argument("--dataset_name", type=str, default="yale-nlp/QTSumm")
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=int, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="test_outputs/llm_inference_outputs")
    parser.add_argument("--evaluated_qa_num", type=int, default=-1)
    args = parser.parse_args()
    
    if args.api_base is not None:
        openai.api_base = args.api_base
        openai.api_type = 'azure'
        openai.api_version = '2023-08-01-preview'

    qa_data = load_dataset(args.dataset_name, split =args.split_name)
    if args.evaluated_qa_num > 0:
        qa_data = qa_data.select(range(args.evaluated_qa_num))
    
    model_inputs = []
    prefix_user_turns, prefix_assistant_turns = prepare_n_shot_prefix(args.num_shot, qa_data)
    for qa in qa_data:
        if args.num_shot == 0:
            prompt_text = prepare_zero_shot_prompt(qa)
            model_input = prepare_model_input(prompt_text)
        else:
            user_turns, assistant_turns, prompt_text = prepare_n_shot_prompt(qa, prefix_user_turns, prefix_assistant_turns)
            model_input = prepare_model_input(prompt_text, user_turns, assistant_turns)
        model_inputs.append(model_input)
        
    model_name = args.model_name_or_path.split("/")[-1].lower()
    
    output_dir = os.path.join(args.output_dir, f"{args.num_shot}_shot")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{model_name}.json")
    
    if "turbo" in args.model_name_or_path.lower():
        requests_per_minute = 100
    else:
        requests_per_minute = 5
        
    outputs = asyncio.run(generate_from_openai_chat_completion(api_key = args.api_key, 
                                                   messages = model_inputs,
                                                   engine_name = args.model_name_or_path, 
                                                   temperature = args.temperature, 
                                                   top_p = args.top_p, 
                                                   max_tokens = args.max_tokens,
                                                   requests_per_minute = requests_per_minute,))
    
    output_data = []
    for output, qa in zip(outputs, qa_data):
        example_id = qa["example_id"]
        cur_output = {
            "example_id": example_id,
            "prediction": output,
            "reference": qa["summary"],
        }
        output_data.append(cur_output)
    
    json.dump(output_data, open(output_path, "w"), indent=4)