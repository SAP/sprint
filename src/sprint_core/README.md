<!-- 
# Copyright (c) 2025 SAP SE or an SAP affiliate company and sprint contributors
# SPDX-License-Identifier: Apache-2.0
-->
# SPRINT Core

Core components for SPRINT fine-tuning and inference.

## Structure

```
sprint_core/
├── __init__.py                 # Package initialization
├── constants.py               # Constants and defaults
├── config_manager.py          # Config loading and validation
├── data_loaders.py           # Data loading utilities
├── model_factory.py          # Model creation
├── training_manager.py       # Training execution
├── experiment_runner.py      # Multi-experiment orchestration
├── eval_utils.py             # Evaluation utilities
├── inference_config.py       # Inference configuration
├── inference_manager.py      # Inference execution
├── multiprocess_launcher.py  # Multi-process utilities
└── README.md                 # Documentation
```

## Components

**constants.py** - Dataset configs, model types, default paths and LoRA parameters

**config_manager.py** - `TrainingConfig` dataclass and `ConfigManager` for YAML loading

**data_loaders.py** - `DataLoaderFactory` with Poisson subsampling, shuffling, and validation loaders

**model_factory.py** - `ModelFactory` for CrypTen configs, LoRA setup, and model instantiation

**training_manager.py** - `TrainingManager` handles privacy engine, training loop, validation, and saving

**experiment_runner.py** - `ExperimentRunner` orchestrates multiple experiments with hyperparameter combinations

**eval_utils.py** - Model evaluation functions and metrics

**inference_config.py** - Inference configuration classes and validation

**inference_manager.py** - `InferenceManager` for model loading, batch processing, and result collection

**multiprocess_launcher.py** - Multi-process execution for MPC inference