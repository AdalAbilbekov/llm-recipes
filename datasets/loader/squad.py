import os
import sys
from datasets import load_from_disk, load_dataset

sys.path.append(f"{os.getenv('HOME')}/llm-distillation")
from prompt.prompt import create_chat_prompt
from prompt.prompt import create_prompt

import pdb

def tokenize(item, tokenizer, encoder_decoder=False):
    is_chat = True if 'chat' in tokenizer.name_or_path.lower() or "instruct" in tokenizer.name_or_path.lower() else False
    task = "qa"

    if tokenizer.name_or_path == "meta-llama/Llama-2-7b-chat-hf":
        shot = 1
        title = False
    elif tokenizer.name_or_path == "mistralai/Mistral-7B-Instruct-v0.2":
        shot = 3
        title = item['title']
    elif tokenizer.name_or_path == "tiiuae/falcon-7b-instruct":
        shot = 4
        title = False

    # Temporary added, need to identify a model based on the conditions above
    shot=1
    is_chat=False
    title=False
    
    if is_chat:
        prompt = create_chat_prompt(
            task, shot,
            title = title,
            context = item['context'],
            question = item['question'],
            sys_user = True if "mistralai/Mistral-7B-Instruct-v0.2" in tokenizer.name_or_path else False,
            chat_template = tokenizer.apply_chat_template
        )
    else:
        prompt = create_prompt(
            task, 0, 
            context = item['context'],
            question = item['question'],
        )

    context_tokens = tokenizer.encode(f"{tokenizer.bos_token} {prompt}", add_special_tokens=False)
   
    if not encoder_decoder:
        if 'chat' in tokenizer.name_or_path.lower() or "instruct" in tokenizer.name_or_path.lower():# Added additional part based on the tokenizer.
            # context_tokens = tokenizer.encode(f"{prompt}", add_special_tokens=False)
            context_tokens = tokenizer.encode(f"{tokenizer.bos_token}{prompt}", add_special_tokens=False) #Added bos eos tokens due the lack of these tokens during tokenization. Recheck for each tokenizer.
            if tokenizer.name_or_path == "tiiuae/falcon-7b-instruct":
                answer_tokens = tokenizer.encode(f" {item['answers_generated']}", add_special_tokens=False)
            else:
                answer_tokens = tokenizer.encode(f"{item['answers_generated'].strip()}{tokenizer.eos_token}", add_special_tokens=False)
        else:
            context_tokens = tokenizer.encode(f"{tokenizer.bos_token}{prompt}", add_special_tokens=False)
            answer_tokens = tokenizer.encode(f" {item['answers_generated']}{tokenizer.eos_token}", add_special_tokens=False)

        prompt_tokens = context_tokens+answer_tokens
        labels_tokens = (len(context_tokens)*[-100,])+answer_tokens

        combined_tokens = {
            "input_ids": prompt_tokens,
            "labels": labels_tokens
        }
        return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))
    else:
        input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")[0]
        labels = tokenizer.encode(item['answers_generated'], add_special_tokens=True, return_tensors="pt")[0]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1]*len(input_ids)
        }

def get_split(dataset_config, tokenizer, split):
    # dataset = load_dataset('Nicolas-BZRD/uld_loss_Llama-2-7b-chat-hf-squad', data_dir='data') The origina Dataset
    # Based on my datasets columns have to be chnaged as follows: instruction -> context | input -> question | output -> answers_generated
    dataset = load_dataset('AdalAbilbekov/arc_distill_test_ENKK', data_dir='data')
    dataset = dataset[split]
    if dataset_config.training_size < 1: dataset = dataset.select(range(int(len(dataset)*dataset_config.training_size)))
    dataset = dataset.map(lambda item: tokenize(item, tokenizer, dataset_config.encoder_decoder), remove_columns=list(dataset.features))
    return dataset