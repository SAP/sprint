# Copyright (c) 2025 SAP SE or an SAP affiliate company and sprint contributors
# SPDX-License-Identifier: Apache-2.0

"""Experiment runner for managing multiple training runs."""

import os
import json
import torch
import numpy as np
import copy
import math
from typing import Dict, Any, List, Iterator
from .config_manager import TrainingConfig
from .model_factory import ModelFactory
from .data_loaders import DataLoaderFactory
from .training_manager import TrainingManager
from .constants import FINETUNING_RESULTS_PATH


class ExperimentRunner:
    """Manages multiple training experiments with different hyperparameters."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.results_dir = FINETUNING_RESULTS_PATH
        
    def run_all_experiments(self, device: str = None, dataset_name: str = None) -> None:
        """Run all experiment combinations."""
        device = device or self.config.device
        dataset_name = dataset_name or self.config.dataset_name
        
        device_for_file = device.replace(":", "")
        
        for experiment_params in self._generate_experiment_combinations():
            self._run_single_experiment(experiment_params, device, dataset_name, device_for_file)
    
    def _generate_experiment_combinations(self) -> Iterator[Dict[str, Any]]:
        """Generate all combinations of hyperparameters."""
        for init_seed in self.config.init_seeds:
            for exponential_scheduler_gamma in self.config.exponential_scheduler_gammas:
                for scheduler in self.config.schedulers:
                    for optimizer in self.config.optimizers:
                        for batch_size in self.config.batch_sizes:
                            for lora_rank in self.config.lora_ranks:
                                for lora_type in self.config.lora_types:
                                    for lora_init_method in self.config.lora_init:
                                        for decay in self.config.weight_decay:
                                            for target_module in self.config.target_modules:
                                                for layers_to_save in self.config.modules_to_save:
                                                    for lora_alpha in self.config.lora_alphas:
                                                        lora_alpha_scaled = lora_alpha * lora_rank
                                                        for caps in self.config.cap_thresholds:
                                                            for learning_rate in self.config.learning_rates:
                                                                for clipping_threshold in self.config.clipping_thresholds:
                                                                    for softmax in self.config.softmaxes:
                                                                        for hidden_act in self.config.hidden_acts:
                                                                            yield {
                                                                                'init_seed': init_seed,
                                                                                'exponential_scheduler_gamma': exponential_scheduler_gamma,
                                                                                'scheduler': scheduler,
                                                                                'optimizer': optimizer,
                                                                                'batch_size': batch_size,
                                                                                'lora_rank': lora_rank,
                                                                                'lora_type': lora_type,
                                                                                'lora_init_method': lora_init_method,
                                                                                'weight_decay': decay,
                                                                                'target_module': target_module,
                                                                                'lora_alpha': lora_alpha_scaled,
                                                                                'caps': caps,
                                                                                'learning_rate': learning_rate,
                                                                                'clipping_threshold': clipping_threshold,
                                                                                'softmax': softmax,
                                                                                'hidden_act': hidden_act,
                                                                                'modules_to_save': layers_to_save
                                                                            }
    
    def _run_single_experiment(
        self,
        experiment_params: Dict[str, Any],
        device: str,
        dataset_name: str,
        device_for_file: str
    ) -> None:
        """Run a single experiment with given parameters."""
        # Generate file name for results
        file_name = self._generate_results_filename(
            experiment_params,
            device_for_file,
            dataset_name
        )
        file_path = os.path.join(self.results_dir, file_name)
        
        # Check if experiment already exists
        if self._experiment_exists(file_path, experiment_params):
            print(f"Skipping existing experiment: {experiment_params}")
            return
        
        # Set seeds
        self._set_seeds(experiment_params['init_seed'])
        
        # Run the experiment
        try:
            results = self._execute_experiment(experiment_params, device, dataset_name)
            self._save_results(file_path, results, experiment_params)
        except Exception as e:
            print(f"Error in experiment {experiment_params}: {e}")
    
    def _execute_experiment(
        self,
        params: Dict[str, Any],
        device: str,
        dataset_name: str
    ) -> Dict[str, Any]:
        """Execute a single training experiment."""
        # Create model configuration
        model_config = ModelFactory.create_model_config(
            softmax_act=params['softmax'],
            hidden_act=params['hidden_act'],
            classifier_act="relu",
            logits_cap_attention=params['caps'],
            logits_cap_classifier=None,
            embeddings_cap=params['caps']
        )
        
        # Create model
        model = ModelFactory.create_model(
            pretrained_model=self.config.pretrained_model,
            dataset_name=dataset_name,
            model_config=model_config,
            device=device
        )
        
        # Apply LoRA if needed
        if params['lora_type'] != "none":
            lora_config = ModelFactory.create_lora_config(
                r=params['lora_rank'],
                lora_alpha=params['lora_alpha'],
                init_method=params['lora_init_method'],
                target_modules=params['target_module'],
                modules_to_save=params['modules_to_save']
            )
            model = ModelFactory.apply_lora(model, lora_config, params['lora_type'])
        
        # Create data loaders
        if self.config.subsampling_type == "poisson":
            train_dataloader = DataLoaderFactory.create_poisson_dataloader(
                dataset_name=dataset_name,
                model_type=self.config.pretrained_model,
                batch_size=params['batch_size']
            )
        else:
            train_dataloader = DataLoaderFactory.create_shuffle_dataloader(
                dataset_name=dataset_name,
                model_type=self.config.pretrained_model,
                batch_size=params['batch_size']
            )
        
        val_dataloader = DataLoaderFactory.create_validation_dataloader(
            dataset_name=dataset_name,
            model_type=self.config.pretrained_model,
            batch_size=self.config.microbatch_size
        )
        
        # Setup training manager
        trainer = TrainingManager(model, device)
        
        # Setup privacy engine
        sample_size = len(train_dataloader.dataset)
        trainer.setup_privacy_engine(
            epochs=self.config.epochs,
            batch_size=params['batch_size'],
            sample_size=sample_size,
            clip_threshold=params['clipping_threshold'],
            epsilon=self.config.epsilon,
            delta=self.config.delta
        )
        
        # Setup optimizer
        trainer.setup_optimizer(
            optimizer_type=params['optimizer'],
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            batch_size=params['batch_size'],
            clip_threshold=params['clipping_threshold']
        )
        
        # Setup scheduler
        trainer.setup_lr_scheduler(
            scheduler_type=params['scheduler'],
            epochs=self.config.epochs,
            sample_size=sample_size,
            batch_size=params['batch_size'],
            exponential_gamma=params['exponential_scheduler_gamma']
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_accuracy = -1
        best_val_loss = math.inf
        best_model_dict = None
        best_epoch = 0
        n_batches = sample_size // params['batch_size']
        
        for epoch in range(self.config.epochs):
            # Train
            epoch_loss, batch_losses = trainer.train_epoch(
                train_dataloader,
                self.config.microbatch_size,
                epoch,
                n_batches
            )
            train_losses.extend(batch_losses)
            
            # Validate
            val_loss, val_accuracy = trainer.validate(val_dataloader)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # Check for best model
            if val_accuracy > best_val_accuracy or (val_accuracy == best_val_accuracy and val_loss < best_val_loss):
                best_val_accuracy = val_accuracy
                best_val_loss = val_loss
                best_model_dict = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                print(f"Best model found at epoch {best_epoch}: Val_loss={val_loss}, val_accuracy={val_accuracy}")
            
            print(f"Epoch {epoch}: Loss={epoch_loss}, Val_loss={val_loss}, val_accuracy={val_accuracy}")
            
            # Check for NaN parameters
            # trainer.check_for_nan_parameters()
        
        # Save best model if requested
        if self.config.save_best_model and best_model_dict is not None:
            model_name = self._generate_model_name(params, dataset_name)
            trainer.save_best_model(
                state_dict=best_model_dict,
                model_name=model_name,
                config_dict=model_config.to_dict(),
                lora_config_dict=lora_config.to_dict() if params['lora_type'] != "none" else {},
                best_metrics={
                    "best_val_accuracy": best_val_accuracy,
                    "best_val_loss": best_val_loss,
                    "best_val_epoch": best_epoch
                },
                eval_params={
                    "batch_size": params['batch_size'],
                    "lr": params['learning_rate'],
                    "clip_threshold": params['clipping_threshold'],
                    "epochs": self.config.epochs,
                    "optimizer": params['optimizer'],
                    "scheduler": params['scheduler'],
                    "weight_decay": params['weight_decay'],
                    "train_seed": self.config.training_seed,
                    "init_seed": params['init_seed']
                }
            )
        
        # Return results
        privacy_spent = trainer.get_privacy_spent()
        return {
            "noise_multiplier": trainer.privacy_engine.noise_multiplier if trainer.privacy_engine else None,
            "effective_noise_multiplier": trainer.privacy_engine.effective_noise_multiplier if trainer.privacy_engine else None,
            "privacy_spent": privacy_spent,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracy": val_accuracies,
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch
        }
    
    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _generate_results_filename(self, experiment_params: Dict[str, Any], device_for_file: str, dataset_name: str) -> str:
        """Generate filename for results."""
        return f"sprint_ft_stats_{experiment_params['init_seed']}_{self.config.pretrained_model}_{experiment_params['softmax']}_{experiment_params['hidden_act']}_{dataset_name}_{device_for_file}.json"

    def _generate_model_name(self, params: Dict[str, Any], dataset_name: str) -> str:
        """Generate model name for saving."""
        components = [
            self.config.pretrained_model,
            f"_{params['optimizer']}_",
            'lora_' if params['lora_type'] == 'lora' else '',
            'falora_' if params['lora_type'] == 'fa_lora' else '',
            f"r{params['lora_rank']}_" if params['lora_type'] != 'none' else '',
            f"a{params['lora_alpha']}_" if params['lora_type'] != 'none' else '',
            f"{params['lora_init_method']}_",
            f"{params['hidden_act']}_",
            f"{params['softmax']}_",
            f"cap_{params['caps']}_" if params['caps'] is not None else '',
            f"{dataset_name}.pth"
        ]
        return ''.join(components)
    
    def _experiment_exists(self, file_path: str, params: Dict[str, Any]) -> bool:
        """Check if experiment already exists in results file."""
        if not os.path.exists(file_path):
            self._create_empty_results_file(file_path)
            return False
        
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Create parameters dict for comparison
            experiment_params = self._create_parameter_dict(params)
            
            for run in data.get("run", []):
                if run.get("parameters") == experiment_params:
                    return True
            return False
        except (json.JSONDecodeError, KeyError):
            return False
    
    def _create_parameter_dict(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create parameter dictionary for saving/comparing."""
        return {
            "batch_size": params['batch_size'],
            "epochs": self.config.epochs,
            "learning_rate": params['learning_rate'],
            "clipping_threshold": params['clipping_threshold'],
            "epsilon": self.config.epsilon,
            "delta": self.config.delta,
            "lora_rank": params['lora_rank'],
            "lora_alpha": params['lora_alpha'],
            "lora_type": params['lora_type'],
            "optimizer": params['optimizer'],
            "scheduler": params['scheduler'],
            "lora_init_method": params['lora_init_method'],
            "init_seed": params['init_seed'],
            "random_seed_for_training": self.config.random_seed_for_training,
            "training_seed": self.config.training_seed,
            "weight_decay": params['weight_decay'],
            "logits_cap_attention": params['caps'],
            "logits_cap_classifier": None,
            "embeddings_cap": params['caps'],
            "target_module": params['target_module'],
            "exponential_scheduler_gamma": params['exponential_scheduler_gamma'],
            "subsampling_type": self.config.subsampling_type
        }
    
    def _create_empty_results_file(self, file_path: str) -> None:
        """Create empty results file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        empty_json_list = {"run": []}
        with open(file_path, "w") as f:
            json.dump(empty_json_list, f)
    
    def _save_results(self, file_path: str, results: Dict[str, Any], params: Dict[str, Any]) -> None:
        """Save experiment results."""
        results["parameters"] = self._create_parameter_dict(params)
        
        with open(file_path, "r+") as f:
            data = json.load(f)
            data["run"].append(results)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
