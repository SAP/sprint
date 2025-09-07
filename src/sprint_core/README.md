# SPRINT Core - Fine-tuning Module

This folder contains code for SPRINT fine-tuning

## Structure

```
sprint_core/
├── __init__.py                 # Package initialization and exports
├── constants.py               # Constants and default configurations
├── config_manager.py          # Configuration loading and validation
├── data_loaders.py           # Data loading utilities and factories
├── model_factory.py          # Model creation and configuration
├── training_manager.py       # Training process management
└── experiment_runner.py      # Multi-experiment orchestration
```

## Key Components

### Constants (`constants.py`)
- Dataset configurations (labels per dataset)
- Supported model types
- Default paths for data and models
- Default LoRA and training parameters

### Configuration Manager (`config_manager.py`)
- `TrainingConfig` dataclass for type-safe configuration
- `ConfigManager` class for loading and validating YAML configs
- Comprehensive error handling and validation

### Data Loaders (`data_loaders.py`)
- `DataLoaderFactory` with methods for different loader types:
  - Poisson subsampling for differential privacy
  - Standard shuffling loaders
  - Validation loaders
- Custom collate functions for proper batching

### Model Factory (`model_factory.py`)
- `ModelFactory` for creating models and configurations:
  - CrypTen configuration creation
  - LoRA configuration setup
  - Model instantiation and device placement
  - LoRA application logic

### Training Manager (`training_manager.py`)
- `TrainingManager` class that handles:
  - Privacy engine setup and configuration
  - Optimizer and scheduler initialization
  - Training loop execution
  - Validation and model saving
  - NaN detection and error handling

### Experiment Runner (`experiment_runner.py`)
- `ExperimentRunner` for orchestrating multiple experiments:
  - Hyperparameter combination generation
  - Experiment deduplication
  - Result saving and management
  - Error handling and recovery