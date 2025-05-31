import os
import torch
import pickle
import argparse
import tiktoken
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
from datasets import load_dataset

parser = argparse.ArgumentParser(description='gpt-wikitext')
parser.add_argument('--name', default='gpt2', type = str, help = 'gpt name')
parser.add_argument('--flag', default=0, type = int, help = 'quantization flag')


args = parser.parse_args()

# load model
model_name = args.name
flag = args.flag
print(model_name)
if flag:
    model = GPT2LMHeadModel.from_pretrained(model_name, torch_dtype=torch.float16)
else:
    model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# load data
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
enc = tiktoken.get_encoding("gpt2")
token_list = []
for line_index, line in enumerate(dataset['test']['text']):
    if len(line) == 0 or (line[0:2] == ' =' and line[-3:] == '= \n'):
        continue
    tokens = enc.encode(line)
    token_list.append(tokens)


# save results
if flag:
    result_path = './result/wikitext/'
    result_file_name = f"{result_path}{model_name}_quantized_loglikelihood.pt"
else:
    result_path = './result/wikitext/'
    result_file_name = f"{result_path}{model_name}_loglikelihood.pt"

if not os.path.isdir(result_path):
    try:
        os.makedirs(result_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(result_path):
            pass
        else:
            raise


neg_log_likelihood_list = []

for token_index, tokens in enumerate(token_list):
    with torch.no_grad():
        input_ids = torch.LongTensor(tokens).unsqueeze(0).to(device)
        outputs = model(input_ids)
        logits = outputs.logits[0,:-1,:]
        labels = input_ids[0, 1:]
        length = input_ids.shape[1] - 1
        loss = F.cross_entropy(logits, labels)
        neg_log_likelihood = loss * length
    neg_log_likelihood_list.append(neg_log_likelihood.item())
    if token_index % 20 == 0:
        print(token_index, neg_log_likelihood_list[-1])
        
    with open(result_file_name, 'wb') as f:
        pickle.dump(neg_log_likelihood_list, f)

with open(result_file_name, 'wb') as f:
    pickle.dump(neg_log_likelihood_list, f)