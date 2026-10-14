import sys
import os
import glob as glob_module
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
from tqdm import tqdm
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, required=False, default="instruction_rank-128_lm_mrqa",help='Directory including the configuration file')
    return parser.parse_args()


def get_examples_list(instruction_dataset_repo, split):
    instruction_dataset_repo_name = instruction_dataset_repo.split('/')[-1]
    # cache long text for preventing full dataset traversal on each preparation.
    if os.path.exists(f'output/{instruction_dataset_repo_name}_{split}_instruction_dataset.json'):
        with open(f'output/{instruction_dataset_repo_name}_{split}_instruction_dataset.json', 'r', encoding='utf-8') as f:
            examples_list =  json.load(f)
        return examples_list
#############################gather all in-domain and out-of-domain dataset for testing##############################################
    examples_list = []
    if split == "train":
        dataset = load_dataset(instruction_dataset_repo, split=split, streaming=True)
        for example in tqdm(dataset, desc="Processing examples"):
            examples_list.append(example)
    else:
        dataset = load_dataset(instruction_dataset_repo, split="test", streaming=True)
        for example in tqdm(dataset, desc="Processing examples"):
            examples_list.append(example)
        dataset = load_dataset(instruction_dataset_repo, split="validation", streaming=True)
        for example in tqdm(dataset, desc="Processing examples"):
            examples_list.append(example)    

        
    with open(f'output/{instruction_dataset_repo_name}_{split}_instruction_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(examples_list, f, ensure_ascii=False)

    return examples_list

    
_BOS_ID = None


def get_ids(instruction_dataset_repo_name, examples_list, tokenizer, split, max_context_tokens=0):

    examples = []
    # info_list = []
    minn = 999999
    maxn = 0
    for example in tqdm(examples_list, desc="Processing examples"):

        ################ one reference answer is enough for training ################
        # answer = ""
        # for i in range(len(example["answers"])):
        #     if i == len(example["answers"])-1:
        #         answer += example["answers"][i]
        #     else:
        #         if example[("answers")][i] == example[("answers")][i+1]:
        #             answer += example["answers"][i]
        #             break
        #         answer += example[("answers")][i] + " "
        ################################################################################
        answer = example["answers"][0]

        context = tokenizer(example["context"], add_special_tokens=False)["input_ids"]
        prompt = tokenizer(example["question"], add_special_tokens=False)["input_ids"]
        answer = tokenizer(answer, add_special_tokens=False)["input_ids"]
        
        context_ids = [_BOS_ID] + tokenizer("### Context:\n", add_special_tokens=False)["input_ids"] + context
        # Truncate context if max_context_tokens is set
        if max_context_tokens > 0 and len(context_ids) > max_context_tokens:
            context_ids = context_ids[:max_context_tokens]
        question_ids = tokenizer("\n### Question:\n", add_special_tokens=False)["input_ids"] + prompt \
                       + tokenizer("\n### Answer:\n", add_special_tokens=False)["input_ids"]
        answer_ids = answer + [tokenizer.eos_token_id] # tokenizer("</s>", add_special_tokens=False)["input_ids"]

        tot_len = len(context_ids) + len(question_ids) + len(answer_ids)
        minn = min(minn,tot_len)
        maxn = max(maxn,tot_len)

        # build instruction_target: the label the model is trained to predict
        instruction_target = [-100 for x in question_ids] + [x for x in answer_ids]
        instruction_target = instruction_target[1:]

        inputs = torch.LongTensor(context_ids)
        # training-time branch
        if split == 'train':
            lm_target = torch.LongTensor(question_ids + answer_ids)
        else:
            lm_target = torch.LongTensor(question_ids)

        instruction_target = torch.LongTensor(instruction_target)

        if split == "test":
            examples.append({"input_ids":inputs,"lm_targets":lm_target})
        else:
            examples.append({"input_ids":inputs,"lm_targets":lm_target,
                            "instruction_target":instruction_target})
        # info_list.append(example)
    print(f"len range: [{minn}:{maxn}]")
    # with open(f'output/{instruction_dataset_repo_name}_{split}_instruction_dataset.json', 'w', encoding='utf-8') as f:
    #     json.dump(info_list, f, ensure_ascii=False)
    return examples

def get_examples(model_id, instruction_dataset_repo, samples_num, min_len, max_len, dataset_repo, max_context_tokens=0):
    
    model_name = model_id.split('/')[-1]
    instruction_dataset_repo_name = instruction_dataset_repo.split('/')[-1]
    train_data_name = f"output/{instruction_dataset_repo_name}_train_"+model_name+f"_{samples_num}samples_instruction.pt"
    eval_data_name = f"output/{instruction_dataset_repo_name}_eval_"+model_name+f"_{samples_num}samples_instruction.pt"

    print(f"in:train_data_name:{train_data_name}")
    if os.path.exists(train_data_name):
        print("loading data...")
        return torch.load(train_data_name), torch.load(eval_data_name)

    # Glob fallback: match any samples_num (for pre-generated datasets like r500x)
    train_glob = glob_module.glob(f"output/{instruction_dataset_repo_name}_train_{model_name}_*samples_instruction.pt")
    eval_glob = glob_module.glob(f"output/{instruction_dataset_repo_name}_eval_{model_name}_*samples_instruction.pt")
    if train_glob and eval_glob:
        train_data_name = sorted(train_glob)[-1]
        eval_data_name = sorted(eval_glob)[-1]
        print(f"loading data (glob match): {train_data_name}")
        return torch.load(train_data_name), torch.load(eval_data_name)

    print(f"preparing data :train_data_name:{train_data_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Backbone-agnostic BOS (Qwen2.5 has bos_token_id=None -> use eos).
    global _BOS_ID
    _BOS_ID = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    
    train_examples_list = get_examples_list(instruction_dataset_repo, split="train")
    test_examples_list = get_examples_list(instruction_dataset_repo, split="test")

    random.seed(0)
    random.shuffle(train_examples_list)
    train_examples_list = train_examples_list[:samples_num]

    train_data = get_ids(instruction_dataset_repo_name, train_examples_list, tokenizer, split='train', max_context_tokens=max_context_tokens)
    test_data = get_ids(instruction_dataset_repo_name, test_examples_list, tokenizer, split='test', max_context_tokens=max_context_tokens)


    torch.save(train_data, train_data_name)
    torch.save(test_data, eval_data_name)
    
    return train_data, test_data
    
if __name__ == "__main__":

    args = parse_args()
    with open(args.work_dir+"/config.json") as f:
        config=json.load(f)

    if not os.path.exists("output"):
        os.makedirs("output")
    
    training_config = config["sft_training_config"]
    config["data_config"]["model_id"] = training_config["model_id"]
    
    print(config["data_config"])
    train_examples, eval_examples = get_examples(**config["data_config"])
    print(len(train_examples))
    print(train_examples[50])
    print(len(eval_examples))
    print(eval_examples[50])

"""
python instruction_prepare_data.py --work_dir '<work_dir>'
"""