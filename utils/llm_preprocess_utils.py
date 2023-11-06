import json
import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

system_prompt = "You are a helpful, respectful and honest assistant. Below is an instruction that describes a query-focused summarization task. Write a summary that appropriately response to the user query."

def process_table_data(table):
    header_str = " | ".join(table["header"])
    rows_str = "\n".join([" | ".join(row) for row in table["rows"]])
    return f"{header_str}\n{rows_str}\n"

def prepare_zero_shot_prompt(example):
    table_str = process_table_data(example["table"])
    table_title = example["table"]["title"]
    query = example["query"]
    prompt_text = f"Table Title: {table_title}\n\n{table_str}\nUsing the information from the table, generate a paragraph-long summary to response to the following user query:\n{query}\n\nSummary:\n"

    return prompt_text

def prepare_n_shot_prefix(n, qa_data):
    user_turns = []
    assistant_turns = []
    for qa in qa_data.select(range(n)):
        table_str = process_table_data(qa["table"])
        table_title = qa["table"]["title"]
        query = qa["query"]
        
        user_text = f"Table Title: {table_title}\n\n{table_str}\nUsing the information from the table, generate a paragraph-long summary to response to the following user query:\n{query}\n\nSummary:\n"
        assistant_text = f"{qa['summary']}\n"
        
        user_turns.append(user_text)
        assistant_turns.append(assistant_text)
    return user_turns, assistant_turns

def prepare_n_shot_prompt(example, user_turns, assistant_turns):    
    table_str = process_table_data(example["table"])
    table_title = example["table"]["title"]
    query = example["query"]
    prompt_text = f"Table Title: {table_title}\n\n{table_str}\nUsing the information from the table, generate a paragraph-long summary to response to the following user query:\n{query}\n\nSummary:\n"

    return user_turns, assistant_turns, prompt_text