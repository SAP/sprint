"""
Tokenize test and train datasets for MRPC dataset and save them as pickle files
Uses bert-base-uncased tokenizer with max_length of 128 taken from the BERT paper
"""
import os
from datasets import load_dataset
from transformers import BertTokenizerFast, DataCollatorWithPadding, AutoTokenizer
import torch
from torch.utils.data import TensorDataset, random_split
import pickle
import sys

import pandas as pd
from datasets import Dataset

import argparse

from sprint_core.constants import TOKENIZED_DATASETS_PATH

parser = argparse.ArgumentParser(description="Tokenize dataset for SPRINT")
parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
parser.add_argument('--model_type', type=str, required=True, help='Model name')
parser.add_argument('--test', action='store_true', help='Run in test mode')


dataset_folder_name = {
    "mrpc": "MRPC",
    "stsb": "STS-B",
    "sst2": "SST-2",
    "rte": "RTE",
    "mnli": "MNLI",
    "qqp": "QQP",
    "qnli": "QNLI"
}

model_names = {
    "bert" : "bert-base-uncased",
    "roberta": "roberta-base",
}


def tokenize_labeled_test_dataset(dataset_name, model_type="bert"):
    # read the .tsv file
    home = os.getenv("HOME")
    dataset_path = f"{home}/finetuning/experiments/transformers/tokenized_datasets/roberta/sst2/label_test.tsv"
    test_dataset = pd.read_csv(dataset_path, delimiter='\t', header=None, names=['label', 'sentence'])

    # add the header as it was a dataset from huggingface
    test_dataset = Dataset.from_pandas(test_dataset)
    print(test_dataset[:10])

    # tokenize the text
    tokenizer = AutoTokenizer.from_pretrained(model_names[model_type], do_lower_case=True)

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length",max_length=128)
    
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    # save the tokenized dataset
    tokenized_test_dataset.save_to_disk(f"{TOKENIZED_DATASETS_PATH}/{model_type}/{dataset_name}/{dataset_name}_test_dataset_labeled")


def tokenize_dataset(dataset_name, model_type="bert"):
    print(f"Dataset name: {dataset_name}")
    print(f"Tokenized dataset directory: {TOKENIZED_DATASETS_PATH}")
    
    max_seq_len = 128

    if dataset_name in ["mrpc", "stsb", "rte"]:
        raw_dataset = load_dataset(f"SetFit/{dataset_name}")
        # remap column name text1 and text2 to sentence1 and sentence2
        raw_dataset = raw_dataset.rename_column("text1", "sentence1")
        raw_dataset = raw_dataset.rename_column("text2", "sentence2")
    elif dataset_name == "sst2":
        raw_dataset = load_dataset("stanfordnlp/sst2")
    elif dataset_name == "mnli":
        raw_dataset = load_dataset("nyu-mll/glue", "mnli")
    elif dataset_name == "qqp":
        raw_dataset = load_dataset("nyu-mll/glue", "qqp")
    elif dataset_name == "qnli":
        raw_dataset = load_dataset("nyu-mll/glue", "qnli")
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported")

    tokenizer = AutoTokenizer.from_pretrained(model_names[model_type], do_lower_case=True)

    print(raw_dataset)

    def tokenize_function_2_sentences(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length",max_length=max_seq_len)
    
    def tokenize_function_1_sentence(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length",max_length=max_seq_len)
    
    def tokenize_function_premise_hypothesis(examples):
        return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length",max_length=max_seq_len)
    
    def tokenize_function_2_questions(examples):
        return tokenizer(examples["question1"], examples["question2"], truncation=True, padding="max_length",max_length=max_seq_len)
    
    def tokenize_function_question_sentence(examples):
        return tokenizer(examples["question"], examples["sentence"], truncation=True, padding="max_length",max_length=max_seq_len)
    

    if dataset_name in ["mrpc", "stsb", "rte"]:
        tokenized_dataset = raw_dataset.map(tokenize_function_2_sentences, batched=True)
    elif dataset_name in ["sst2"]:
        tokenized_dataset = raw_dataset.map(tokenize_function_1_sentence, batched=True)
    elif dataset_name in ["mnli"]:
        tokenized_dataset = raw_dataset.map(tokenize_function_premise_hypothesis, batched=True)
    elif dataset_name in ["qqp"]:
        tokenized_dataset = raw_dataset.map(tokenize_function_2_questions, batched=True)
    elif dataset_name in ["qnli"]:
        tokenized_dataset = raw_dataset.map(tokenize_function_question_sentence, batched=True)

    print(tokenized_dataset)
    
    
    # Save the tokenized dataset
    tokenized_train_dataset = tokenized_dataset["train"]

    if dataset_name in ["mnli"]:
        tokenized_val_dataset = tokenized_dataset["validation_matched"]
        tokenized_val_dataset_mismatched = tokenized_dataset["validation_mismatched"]
        tokenized_test_dataset = tokenized_dataset["test_matched"]
        tokenized_test_dataset_mismatched = tokenized_dataset["test_mismatched"]
    else:
        tokenized_val_dataset = tokenized_dataset["validation"]
        tokenized_test_dataset = tokenized_dataset["test"]


    tokenized_train_dataset.save_to_disk(f"{TOKENIZED_DATASETS_PATH}/{model_type}/{dataset_name}/{dataset_name}_train_dataset")

    tokenized_val_dataset.save_to_disk(f"{TOKENIZED_DATASETS_PATH}/{model_type}/{dataset_name}/{dataset_name}_val_dataset")

    tokenized_test_dataset.save_to_disk(f"{TOKENIZED_DATASETS_PATH}/{model_type}/{dataset_name}/{dataset_name}_test_dataset")

    if dataset_name in ["mnli"]:
        tokenized_val_dataset_mismatched.save_to_disk(f"{TOKENIZED_DATASETS_PATH}/{model_type}/{dataset_name}/{dataset_name}_val_dataset_mismatched")
        tokenized_test_dataset_mismatched.save_to_disk(f"{TOKENIZED_DATASETS_PATH}/{model_type}/{dataset_name}/{dataset_name}_test_dataset_mismatched")
    

if __name__ == '__main__':
    args = parser.parse_args()

    dataset_name = args.dataset 
    model_type = args.model_type
    test=args.test
    
    if model_type not in model_names.keys():
        raise ValueError(f"Model {model_type} is not supported, supported models types are {model_names.keys()}")


    if dataset_name not in dataset_folder_name.keys():
        raise ValueError(f"Dataset {dataset_name} is not supported")
    
    print(f"Tokenizing {dataset_name} dataset")
    if test:
        print("Tokenizing test dataset")
        tokenize_labeled_test_dataset(dataset_name, model_type)
    else:
        tokenize_dataset(dataset_name,model_type)
    




