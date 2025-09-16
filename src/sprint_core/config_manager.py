# Copyright (c) 2025 SAP SE or an SAP affiliate company and sprint contributors
# SPDX-License-Identifier: Apache-2.0

"""Configuration management for SPRINT fine-tuning."""

import os
import yaml
from dataclasses import dataclass
from typing import List, Optional, Union, Any, Dict
from .constants import DEFAULT_TRAINING_PARAMS, DEFAULT_LORA_CONFIG


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_sizes: List[int]
    clipping_thresholds: List[float]
    learning_rates: List[float]
    epochs: int
    epsilon: float
    delta: float
    microbatch_size: int
    weight_decay: List[float]
    lora_ranks: List[int]
    lora_alphas: List[int]
    cap_thresholds: List[Optional[int]]
    exponential_scheduler_gammas: List[float]
    
    # Model and dataset
    pretrained_model: str
    dataset_name: str
    device: str
    
    # Optimization settings
    lora_types: List[str]
    schedulers: List[str]
    optimizers: List[str]
    hidden_acts: List[str]
    softmaxes: List[str]
    target_modules: List[List[str]]
    lora_init: List[str]
    modules_to_save: List[List[str]]
    # Training behavior
    save_best_model: bool
    random_seed_for_training: bool
    training_seed: Optional[int]
    init_seeds: List[int]
    subsampling_type: str


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, base_path: str = None):
        self.base_path = base_path or f"{os.getenv('HOME')}/sprint/src/configs"
    
    def load_from_yaml(self, config_path: Optional[str] = None) -> TrainingConfig:
        """Load configuration from YAML file."""
        config_path = f"{self.base_path}/{config_path}"
        
        try:
            with open(config_path, "r") as file:
                config_dict = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
        
        return self._parse_config(config_dict)
    
    def _parse_config(self, config_dict: Dict[str, Any]) -> TrainingConfig:
        """Parse and validate configuration dictionary."""
        try:
            return TrainingConfig(
                batch_sizes=[int(x) for x in config_dict['batch_sizes']],
                clipping_thresholds=[float(x) for x in config_dict['clipping_thresholds']],
                learning_rates=[float(x) for x in config_dict['learning_rates']],
                epochs=int(config_dict['epochs']),
                epsilon=float(config_dict['epsilon']),
                delta=float(config_dict['delta']),
                microbatch_size=int(config_dict['microbatch_size']),
                weight_decay=[float(x) for x in config_dict['weight_decay']],
                lora_ranks=[int(x) for x in config_dict['lora_ranks']],
                lora_alphas=[int(x) for x in config_dict['lora_alphas']],
                cap_thresholds=[None if x is None or x == 'None' else int(x) 
                               for x in config_dict['cap_thresholds']],
                exponential_scheduler_gammas=[float(x) for x in config_dict['exponential_scheduler_gammas']],
                
                pretrained_model=config_dict['pretrained_model'],
                dataset_name=config_dict['dataset_name'],
                device=config_dict.get('device', 'cuda:0'),
                
                lora_types=config_dict['lora_types'],
                schedulers=config_dict['schedulers'],
                optimizers=config_dict['optimizers'],
                hidden_acts=config_dict['hidden_acts'],
                softmaxes=config_dict['softmaxes'],
                target_modules=config_dict['target_modules'],
                lora_init=config_dict['lora_init'],
                modules_to_save=config_dict['modules_to_save'] if 'modules_to_save' in config_dict else [["classifier"]],

                save_best_model=config_dict['save_best_model'],
                random_seed_for_training=config_dict['random_seed_for_training'],
                training_seed=config_dict.get('training_seed'),
                init_seeds=[int(x) for x in config_dict['init_seeds']],
                subsampling_type=config_dict['subsampling_type']
            )
        except KeyError as e:
            raise ValueError(f"Missing required configuration key: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid configuration value: {e}")
