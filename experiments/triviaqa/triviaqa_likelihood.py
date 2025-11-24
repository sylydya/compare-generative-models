import argparse
import pickle
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import torch.nn.functional as F
from tqdm import tqdm


def compute_log_likelihood(model, tokenizer, prompt: str, target: str):

    input_text = prompt + target
    input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)


    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

        # Only consider target token likelihoods
        target_ids = input_ids[0, len(prompt_ids[0]):]
        target_logits = logits[0, len(prompt_ids[0]) - 1:-1]

        log_probs = F.log_softmax(target_logits, dim=-1)
        target_log_probs = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)

        return target_log_probs.sum().item()


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

    print(f"Loading dataset")
    dataset = load_dataset("trivia_qa", "unfiltered.nocontext", split="validation")
    format_fn = format_triviaqa

    log_likelihood_list = []
    for example in tqdm(dataset):
        prompt, target = format_fn(example)
        ll = compute_log_likelihood(model, tokenizer, prompt, target)
        log_likelihood_list.append(ll)

    save_file_name = f"{save_result_path}/log_likelihood_list.pt"
    with open(save_file_name, 'wb') as f:
        pickle.dump(log_likelihood_list, f)

    avg_ll = np.mean(log_likelihood_list)
    print(f"Average log-likelihood over {data_name}: {avg_ll:.4f}")


if __name__ == "__main__":
    main()
