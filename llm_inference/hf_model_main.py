from vllm import LLM, SamplingParams
import json
import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.llm_preprocess_utils import *


def prepare_model_input(tokenizer, model_name, prompt_text, user_turns = [], assistant_turns = []):
    model_input = ""
    if "lemur" in model_name.lower() and "chat" in model_name.lower():
        model_input = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
        for user_turn, assistant_turn in zip(user_turns, assistant_turns):
            model_input += f"<|im_start|>user\n{user_turn}\n<|im_end|>\n<|im_start|>assistant\n{assistant_turn}\n<|im_end|>\n"
        model_input += f"<|im_start|>user\n{prompt_text}\n<|im_end|>\n"
        model_input += f"<|im_start|>assistant\n"
    elif "mistral" in model_name.lower() and "instruct" in model_name.lower():
        chat = []
        for user_turn, assistant_turn in zip(user_turns, assistant_turns):
            chat.append({"role": "user", "content": user_turn})
            chat.append({"role": "assistant", "content": assistant_turn})
        chat.append({"role": "user", "content": prompt_text})
        model_input = tokenizer.apply_chat_template(chat, tokenize=False)
    elif "chat" in model_name.lower():
        chat = [{"role": "system", "content": system_prompt}]
        for user_turn, assistant_turn in zip(user_turns, assistant_turns):
            chat.append({"role": "user", "content": user_turn})
            chat.append({"role": "assistant", "content": assistant_turn})
        chat.append({"role": "user", "content": prompt_text})
        model_input = tokenizer.apply_chat_template(chat, tokenize=False)
    else:
        for user_turn, assistant_turn in zip(user_turns, assistant_turns):
            model_input += f"{user_turn}\n{assistant_turn}\n"
        model_input += prompt_text
        
    return model_input
    
def process_output(output_text):
    return output_text.lstrip().split("\n\n")[0]
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--num_shot", type=int, default=None, required=True)
    parser.add_argument("--dataset_name", type=str, default="yale-nlp/QTSumm")
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=int, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="test_outputs/llm_inference_outputs")
    parser.add_argument("--evaluated_qa_num", type=int, default=-1)
    args = parser.parse_args()
    
    cuda_devices_count = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    qa_data = load_dataset(args.dataset_name, split =args.split_name)
    if args.evaluated_qa_num > 0:
        qa_data = qa_data.select(range(args.evaluated_qa_num))

    if "awq" in args.model_name_or_path.lower():
        quantization_method = "awq"
    else:
        quantization_method = None
        
    llm = LLM(args.model_name_or_path, 
            tensor_parallel_size=cuda_devices_count, 
            quantization=quantization_method,
            )
    
    sampling_params = SamplingParams(n = 1, max_tokens = args.max_tokens, top_p = args.top_p, temperature = args.temperature)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, verbose=False)
    tokenizer.use_default_system_prompt = True
    
    model_inputs = []
    prefix_user_turns, prefix_assistant_turns = prepare_n_shot_prefix(args.num_shot, qa_data)
    for qa in tqdm(qa_data):
        if args.num_shot == 0:
            prompt_text = prepare_zero_shot_prompt(qa)
            model_input = prepare_model_input(tokenizer, args.model_name_or_path, prompt_text)
        else:
            user_turns, assistant_turns, prompt_text = prepare_n_shot_prompt(qa, prefix_user_turns, prefix_assistant_turns)
            model_input = prepare_model_input(tokenizer, args.model_name_or_path, prompt_text, user_turns, assistant_turns)
        model_inputs.append(model_input)
    
    model_name = args.model_name_or_path.split("/")[-1].lower()
    
    output_dir = os.path.join(args.output_dir, f"{args.num_shot}_shot")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{model_name}.json")
    
    outputs = llm.generate(model_inputs, sampling_params)
    output_data = []
    for output, qa in zip(outputs, qa_data):
        example_id = qa["example_id"]
        output_text = output.outputs[0].text
        cur_output = {
            "example_id": example_id,
            "prediction": process_output(output_text),
            "reference": qa["summary"],
        }
        output_data.append(cur_output)
                
    json.dump(output_data, open(output_path, "w"), indent=4)
    

    




