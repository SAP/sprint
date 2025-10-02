# Copyright (c) 2025 SAP SE or an SAP affiliate company and sprint contributors
# SPDX-License-Identifier: Apache-2.0

"""Constants and configuration for SPRINT fine-tuning."""

import os
from typing import Dict


def _get_sprint_path():
    """Get SPRINT path from environment or detect automatically."""
    # 1. Check environment variable first
    if os.getenv('SPRINT_PATH'):
        return os.getenv('SPRINT_PATH')
    
    print("⚠️ SPRINT_PATH environment variable not set, attempting to detect automatically.")
    
    # 2. Try to detect based on current file location
    current_file = os.path.abspath(__file__)
    # Go up from src/sprint_core/constants.py to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    
    # Check if this looks like the right directory
    if os.path.exists(os.path.join(project_root, "src", "sprint_core")):
        print(f"✅ Detected SPRINT_PATH as: {project_root}")
        return project_root
    
    # 3. Fallback to current working directory
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, "src", "sprint_core")):
        print(f"✅ Using current working directory as SPRINT_PATH: {cwd}")
        return cwd
    
    # 4. Final fallback to home directory
    print("⚠️ Could not detect SPRINT_PATH automatically, defaulting to $HOME/sprint.")
    return f"{os.getenv('HOME')}/sprint"



SPRINT_PATH = _get_sprint_path()

# Default paths
CONFIG_PATH = f"{SPRINT_PATH}/src/configs"
BASE_DATA_PATH = f"{SPRINT_PATH}/data"

TOKENIZED_DATASETS_PATH = f"{BASE_DATA_PATH}/tokenized_datasets"
MODELS_PATH = f"{BASE_DATA_PATH}/models"
FINETUNING_RESULTS_PATH = f"{BASE_DATA_PATH}/finetuning"
INFERENCE_RESULTS_PATH = f"{BASE_DATA_PATH}/inference"


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
