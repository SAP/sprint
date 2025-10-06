# Copyright (c) 2025 SAP SE or an SAP affiliate company and sprint contributors
# SPDX-License-Identifier: Apache-2.0

"""Training management and execution."""

import os
import math
import copy
import time
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from transformers.optimization import get_linear_schedule_with_warmup
from private_transformers import PrivacyEngine
from modeling.optimizers.adam_bc import AdamCorr
from .constants import MODELS_PATH


class TrainingManager:
    """Manages the training process for SPRINT models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        precision: str = "fp32"
    ):
        self.model = model
        self.device = device
        self.precision = precision
        self.privacy_engine: Optional[PrivacyEngine] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def setup_privacy_engine(
        self,
        epochs: int,
        batch_size: int,
        sample_size: int,
        clip_threshold: float,
        epsilon: float,
        delta: float
    ) -> None:
        """Setup the privacy engine for differential privacy."""
        self.privacy_engine = PrivacyEngine(
            module=self.model,
            epochs=epochs * (sample_size // batch_size),
            batch_size=batch_size,
            sample_size=sample_size,
            max_grad_norm=clip_threshold,
            target_epsilon=epsilon,
            target_delta=delta,
            clipping_mode="ghost",
            record_snr=True,
            accounting_mode="rdp",
            skip_checks=True,
        )

    def setup_optimizer(
        self,
        optimizer_type: str,
        learning_rate: float,
        weight_decay: float = 0.01,
        batch_size: Optional[int] = None,
        clip_threshold: Optional[float] = None
    ) -> None:
        """Setup the optimizer."""
        if optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=weight_decay
            )
        elif optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate
            )
        elif optimizer_type == "adam_bc":
            if not all([batch_size, clip_threshold, self.privacy_engine]):
                raise ValueError("adam_bc requires batch_size, clip_threshold, and privacy_engine")
            
            self.optimizer = AdamCorr(
                params=self.model.parameters(),
                dp_batch_size=batch_size,
                dp_noise_multiplier=self.privacy_engine.noise_multiplier,
                dp_l2_norm_clip=clip_threshold,
                eps_root=0.000000007,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        if self.privacy_engine:
            self.privacy_engine.attach(self.optimizer)

    def setup_lr_scheduler(
        self,
        scheduler_type: str,
        epochs: int,
        sample_size: int,
        batch_size: int,
        exponential_gamma: float = 1.0
    ) -> None:
        """Setup the learning rate scheduler."""
        if not self.optimizer:
            raise ValueError("Optimizer must be setup before scheduler")

        total_steps = epochs * (sample_size // batch_size)
        
        if scheduler_type == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
            )
        elif scheduler_type == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=0
            )
        elif scheduler_type == "exponential":
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=exponential_gamma
            )
        elif scheduler_type in [None, "none"]:
            self.lr_scheduler = None
        else:
            raise ValueError(f"Unknown learning rate scheduler: {scheduler_type}")

    def train_epoch(
        self,
        train_dataloader,
        microbatch_size: int,
        epoch: int,
        n_batches: int
    ) -> Tuple[float, list]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0
        batch_losses = []
        self.model.zero_grad(set_to_none=True)

        for count_batch, batch in enumerate(train_dataloader, 1):
            batch_loss = self._process_batch(batch, microbatch_size)
            
            batch_losses.append(batch_loss)
            epoch_loss += batch_loss
            
            print(f"Epoch {epoch} processed {count_batch}/{n_batches} batches:\n\tBatch Loss={batch_loss}")
            if self.privacy_engine:
                print(f"\tPrivacy spent: {self.privacy_engine.get_privacy_spent()}")

        epoch_loss /= n_batches
        return epoch_loss, batch_losses

    def _process_batch(self, batch: Dict[str, torch.Tensor], microbatch_size: int) -> float:
        """Process a single batch with microbatching."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)
        labels = batch["label"].to(self.device)

        n_subsamples = len(input_ids)
        n_microbatches = math.ceil(n_subsamples / microbatch_size)
        batch_loss = 0

        for i in range(n_microbatches):
            start_idx = i * microbatch_size
            end_idx = (i + 1) * microbatch_size
            
            micro_ids = input_ids[start_idx:end_idx]
            micro_mask = attention_mask[start_idx:end_idx]
            micro_token_type = token_type_ids[start_idx:end_idx]
            micro_labels = labels[start_idx:end_idx]

            logits = self.model(
                input_ids=micro_ids,
                attention_mask=micro_mask,
                token_type_ids=micro_token_type
            )

            logits = logits["logits"] if isinstance(logits, dict) else logits

            # Check for NaN values
            #if torch.isnan(logits).sum() > 0:
            #    print(f"Warning: {torch.isnan(logits).sum()} NaN values in logits")

            loss = self.loss_fn(logits, micro_labels)
            batch_loss += loss.sum().item()

            if i == n_microbatches - 1:
                self.optimizer.step(loss=loss)
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                self.model.zero_grad(set_to_none=True)
            else:
                self.optimizer.virtual_step(loss=loss)

        return batch_loss / n_subsamples

    def validate(self, val_dataloader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                
                logits = self.model(input_ids, attention_mask=attention_mask)
                logits = logits["logits"] if isinstance(logits, dict) else logits
                
                loss = self.loss_fn(logits, labels)
                val_loss += loss.sum().item()
                
                pred_labels = logits.argmax(1)
                val_correct += (pred_labels == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_dataloader.dataset)
        val_accuracy = val_correct / val_total
        
        return val_loss, val_accuracy

    def save_best_model(
        self,
        state_dict: Dict[str, Any],
        dataset_name: str,
        model_name: str,
        config_dict: Dict[str, Any],
        lora_config_dict: Dict[str, Any],
        best_metrics: Dict[str, Any],
        eval_params: Dict[str, Any]
    ) -> str:
        """Save the best model."""
        model_path = os.path.join(MODELS_PATH, dataset_name, model_name)
        print(f"Saving model to {model_path}...")
        # Check if the path exists, otherwise create it
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        object_to_save = {
            "model": model_name.split('_')[0],  # Extract base model name
            "state_dict": state_dict,
            "config": config_dict,
            "lora_config": lora_config_dict,
            **best_metrics,
            "eval_params": eval_params
        }
        
        torch.save(object_to_save, model_path)
        print(f"Model saved at {model_path}")
        return model_path

    def check_for_nan_parameters(self) -> None:
        """Check if any parameters contain NaN values."""
        for name, param in self.model.named_parameters():
            if torch.isnan(param).sum() > 0:
                print(f"Warning: {torch.isnan(param).sum()} NaN values in {name}")

    def get_privacy_spent(self) -> Optional[Tuple[float, float]]:
        """Get privacy budget spent."""
        if self.privacy_engine:
            return self.privacy_engine.get_privacy_spent()
        return None
