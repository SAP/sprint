# Copyright (c) 2025 SAP SE or an SAP affiliate company and sprint contributors
# SPDX-License-Identifier: Apache-2.0

"""Configuration management for SPRINT inference."""

import os
import yaml
from dataclasses import dataclass
from typing import List, Optional, Any, Dict
from .constants import CONFIG_PATH


@dataclass
class InferenceConfig:
    """Configuration for inference parameters."""
    # Model settings
    model_name: str = "roberta-base"
    test_dataset: str = "sst2" # if toy, then random data to evaluate runtime and communication costs
    batch_size: int = 1
    devices: List[str] = None
    n_samples: int = -1

    # Model configuration
    hidden_act: Optional[str] = None
    softmax: Optional[str] = None
    cap: Optional[float] = None
    
    # Execution settings
    encrypted: bool = False
    world_size: int = 1
    multiprocess: bool = False
    debug: bool = False

    profile: bool = False
    verbose: bool = False
    
    # For runtime and communication evaluation (if none, leaves the model as it is)
    lora_type: Optional[str] = None
    target_modules: Optional[List[str]] = None
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    modules_to_save: Optional[List[str]] = None


class InferenceConfigManager:
    """Manages inference configuration loading and validation."""
    
    def __init__(self):
        super().__init__()

    def load_from_yaml(self, config_file: str) -> InferenceConfig:
        """Load configuration directly from a YAML file."""
        config_path = os.path.join(CONFIG_PATH, config_file)
        try:
            with open(config_path, "r") as file:
                config_dict = yaml.safe_load(file) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
            
        return self._parse_config(config_dict)

    def _parse_config(self, config_dict: Dict[str, Any]) -> InferenceConfig:
        """Parse and validate configuration dictionary."""
        return InferenceConfig(
            model_name=config_dict.get('model_name'),
            test_dataset=config_dict.get('test_dataset', 'sst2'),
            batch_size=config_dict.get('batch_size', 1),
            devices=config_dict.get('devices'),
            hidden_act=config_dict.get('hidden_act'),
            softmax=config_dict.get('softmax'),
            cap=config_dict.get('cap'),
            encrypted=config_dict.get('encrypted', False),
            world_size=config_dict.get('world_size', 1),
            debug=config_dict.get('debug', False),
            lora_type=config_dict.get('lora_type'),
            target_modules=config_dict.get('target_modules'),
            modules_to_save=config_dict.get('modules_to_save'),
            lora_rank=config_dict.get('lora_rank'),
            lora_alpha=config_dict.get('lora_alpha'),
            profile=config_dict.get('profile', False),
            verbose=config_dict.get('verbose', False),
            n_samples=config_dict.get('n_samples', -1),
            multiprocess=config_dict.get('multiprocess', False)
        )
