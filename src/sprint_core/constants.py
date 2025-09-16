# Copyright (c) 2025 SAP SE or an SAP affiliate company and sprint contributors
# SPDX-License-Identifier: Apache-2.0

"""Constants and configuration for SPRINT fine-tuning."""

import os
from typing import Dict

# Dataset configurations
LABELS_PER_DATASET: Dict[str, int] = {
    "sst2": 2,
    "mnli": 3,
    "cola": 2,
    "mrpc": 2,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "stsb": 1,
    "toy": 2
}

MODEL_DICT: Dict[str, str] = {
    "bert-base": "bert-base-uncased",
    "bert-medium": "prajjwal1/bert-medium",
    "bert-tiny": "prajjwal1/bert-tiny",
    "bert-small": "prajjwal1/bert-small",
    "tinybert": "huawei-noah/TinyBERT_General_4L_312D",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
}

# Model types
SUPPORTED_MODEL_TYPES = ["bert", "roberta"]

# Default paths
HOME_PATH = os.getenv("HOME")
BASE_DATA_PATH = f"{HOME_PATH}/sprint/data"
TOKENIZED_DATASETS_PATH = f"{BASE_DATA_PATH}/tokenized_datasets"
MODELS_PATH = f"{BASE_DATA_PATH}/models"
FINETUNING_RESULTS_PATH = f"{BASE_DATA_PATH}/finetuning"
SAVED_MODELS_PATH = f"{BASE_DATA_PATH}/models"

# Default LoRA configuration
DEFAULT_LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "freeze_A": False,
    "modules_to_save": ["classifier"],
    "bias": "none",
    "target_modules": ['query', 'key', 'value', 'dense'],
    "fan_in_fan_out": False,
    "init_method": "normal"
}

# Default fine-tuning parameters
DEFAULT_TRAINING_PARAMS = {
    "batch_size": 512,
    "microbatch_size": 16,
    "epochs": 10,
    "learning_rate": 5e-5,
    "epsilon": 8.0,
    "clip_threshold": 1.0,
    "delta": 1e-5,
    "weight_decay": 0.01,
    "logits_cap_attention": 50.0,
    "logits_cap_classifier": 50.0,
    "embeddings_cap": 50.0,
}
