import argparse
import pickle
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import torch
import torch.nn.functional as F
from tqdm import tqdm

import string
import re
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_f1(prediction, ground_truth):
    prediction = normalize_answer(prediction).split()
    ground_truth = normalize_answer(ground_truth).split()
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
 
    return f1



def format_triviaqa(example):
    question = example["question"]
    answer = example["answer"]["value"]
    answer = f" {answer}"
    prompt = f"Question: {question}\nAnswer:"
    return prompt, answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='facebook/opt-1.3b')
    args = parser.parse_args()

    model_name = args.model_name

    data_name = 'triviaqa'
    base_path = './'
    save_result_path = "{}/result/{}/{}".format(base_path, data_name, model_name)
    os.makedirs(save_result_path, exist_ok=True) 


    print(f"Loading model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


    print(f"Loading dataset {data_name}")
    dataset = load_dataset("trivia_qa", "unfiltered.nocontext", split="validation")
    format_fn = format_triviaqa

    
    f1_list = []
    temp_count = 0
    for example in tqdm(dataset):
        prompt, target = format_fn(example)

        outputs = generator(prompt, max_new_tokens=32, return_full_text=False, do_sample=False)
        gen_answer = outputs[0]["generated_text"].strip()
        f1 = compute_f1(gen_answer, target)
        f1_list.append(f1)
        temp_count += 1
        if temp_count % 100 == 0:
            save_file_name = f"{save_result_path}/f1_score.pt"
            with open(save_file_name, 'wb') as f:
                pickle.dump(f1_list, f)

    save_file_name = f"{save_result_path}/f1_score.pt"
    with open(save_file_name, 'wb') as f:
        pickle.dump(f1_list, f)

    avg_ll = np.mean(f1_list)
    print(f"Average f1 over {data_name}: {avg_ll:.4f}")

        
if __name__ == "__main__":
    main()