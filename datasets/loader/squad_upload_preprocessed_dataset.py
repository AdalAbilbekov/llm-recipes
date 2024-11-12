import os
import sys
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict

def get_split(dataset_config, tokenizer, split):
    
    if split is "train":
        if "batyr" in tokenizer.name_or_path.lower():
            dataset = load_dataset("json", data_files="/data/nvme3n1p1/adal_workspace/v2_distillation/datasets/student_dataset_train.json")
        else:
            dataset = load_dataset("json", data_files="/data/nvme3n1p1/adal_workspace/v2_distillation/datasets/teacher_dataset_train.json")
    else:
        if "batyr" in tokenizer.name_or_path.lower():
            dataset = load_dataset("json", data_files="/data/nvme3n1p1/adal_workspace/v2_distillation/datasets/student_dataset_val.json")
        else:
            dataset = load_dataset("json", data_files="/data/nvme3n1p1/adal_workspace/v2_distillation/datasets/teacher_dataset_val.json")

    return dataset['train']

