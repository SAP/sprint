"""Inference management and execution for SPRINT models."""

import logging
from pyexpat import model
import torch
import crypten
import json
import os
from typing import Dict, List, Optional, Tuple, Any

from .constants import MODEL_DICT
from torch import device
from .inference_config import InferenceConfig
from .model_factory import ModelFactory
from .data_loaders import DataLoaderFactory
from modeling.lora.lora_utils import get_lora_config_from_model, decrypt_non_lora_parameters, LoraConfig


class InferenceManager:
    """Manages the inference process for SPRINT models."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.rank = 0
        self.device = "cpu"
        self.model = None
        self.test_dataloader = None
        
    def setup_environment(self) -> None:
        """Setup CrypTen environment and device assignment."""
        if self.config.encrypted:
            config_file_path = f"{self.config.src_path}/configs/{self.config.crypten_config}"
            crypten.init(config_file=config_file_path)
            self.rank = crypten.communicator.get().get_rank()
            print(f"Crypten initialized on rank {self.rank}")
        else:
            self.rank = 0
            
        # Assign device based on rank
        if self.rank < len(self.config.devices):
            self.device = self.config.devices[self.rank]
        else:
            self.device = self.config.devices[0]  # Fallback to first device
            
        # Validate device availability
        if "cuda" in self.device and not torch.cuda.is_available():
            print(f"Warning: CUDA not available, falling back to CPU")
            self.device = "cpu"
            
        self._debug_print(f"Device: {self.device}")
    
    def load_model(self) -> None:
        """Load the fine-tuned model."""
        model_type = self._get_model_type()
        # Get model classes based on encryption mode
        model_type_to_class = self._get_model_classes()
        
        # Load model
        try:
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"Model file not found: {self.config.model_path}")
            self.model = ModelFactory.load_finetuned_model(
                model_name=self.config.model_name,
                model_path=self.config.model_path,
                test_dataset=self.config.test_dataset,
                device=self.device,
                model_type_to_class=model_type_to_class,
                softmax=self.config.softmax,
                hidden_act=self.config.hidden_act,
                cap=self.config.cap
            )
        except FileNotFoundError as e:
            print(f"The model file was not found, loading a pre-trained {model_type} model with 2 labels")
            model_config = ModelFactory.create_model_config(
                softmax_act=self.config.softmax,
                hidden_act=self.config.hidden_act,
                classifier_act="relu",
                logits_cap_attention=self.config.cap,
                logits_cap_classifier=None,
                embeddings_cap=self.config.cap
            )
        
            # Create model
            self.model = model_type_to_class[model_type].from_pretrained(
                pretrained_model_name_or_path=MODEL_DICT[self.config.model_name],
                num_labels=2,
                model_config=model_config,
            )
        
            # Apply LoRA if needed
            if self.config.lora_type != "none":
                lora_config = ModelFactory.create_lora_config(
                    r=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    target_modules=self.config.target_modules,
                    modules_to_save=self.config.modules_to_save,
                    freeze_A=self.config.lora_type=="falora"
                )
                self.model = ModelFactory.apply_lora(self.model, lora_config, self.config.lora_type)
            
            self.model.to(self.config.devices[self.rank])

        self._debug_print(f"Retrieved model of type {type(self.model)}")
        
        # Encrypt model if needed
        if self.config.encrypted:
            self.model.encrypt()
            self._decrypt_non_lora_parameters()
            self._debug_print("Model encrypted")
    
    def load_data(self) -> None:
        """Load test data."""
        model_type = self._get_model_type()
        
        self.test_dataloader = DataLoaderFactory.create_test_dataloader(
            dataset_name=self.config.test_dataset,
            model_type=model_type,
            batch_size=self.config.batch_size,
            n_samples=self.config.n_samples,
            shuffle=False
        )
        
        self._debug_print("Retrieved data loader")
    
    def run_inference(self) -> Dict[str, Any]:
        """Run inference and return results."""
        if self.model is None or self.test_dataloader is None:
            raise ValueError("Model and data must be loaded before inference")
        
        predictions = []
        total_labels = 0
        correct_predictions = 0
        inference_steps = len(self.test_dataloader)
        cnt_overflows = 0
        corrected_overflows = 0
        
        with torch.no_grad():
            with crypten.no_grad() if self.config.encrypted else torch.no_grad():
                for i, batch in enumerate(self.test_dataloader, 1):
                    result = self._process_batch(batch, i, inference_steps)
                    
                    if result['overflow_detected']:
                        cnt_overflows += 1
                        if result['overflow_corrected']:
                            corrected_overflows += 1
                        else:
                            self._debug_print(f"Overflows persist after retrying: {cnt_overflows}")
                            break
                    
                    predictions.append(result['predictions'])
                    correct_predictions += result['correct_count']
                    total_labels += result['total_count']
        
        accuracy = correct_predictions / total_labels if total_labels > 0 else 0
        
        # Convert all predictions to lists for JSON serialization
        results = {
            'accuracy': accuracy,
            'total_labels': total_labels,
            'correct_predictions': correct_predictions,
            'predictions': [p.tolist() if hasattr(p, 'tolist') else p for p in predictions]
        }
        
        if self.config.encrypted:
            results.update({
                'overflows_detected': cnt_overflows,
                'overflows_corrected': corrected_overflows
            })
        
        # Print results (only rank 0 for encrypted mode)
        if not self.config.encrypted or self.rank == 0:
            print(f"Accuracy: {accuracy}")
            if self.config.encrypted:
                print(f"Overflows detected: {cnt_overflows}, overflows corrected: {corrected_overflows}")

        home_path = os.path.expanduser("~")
        experiment_name = f'{"clear" if not self.config.encrypted else f"mpc{self.config.world_size}p"}_{self.config.model_name}_{self.config.test_dataset}'
        FINETUNING_RESULTS_PATH = f"{home_path}/sprint/data/inference/accuracy/{self.config.test_dataset}"

        with open(f"{FINETUNING_RESULTS_PATH}/{experiment_name}_results.json", "w") as f:
            json.dump(results, f)

        return results
    

    def run_inference_profiled(self) -> Dict[str, Any]:
        if not self.config.encrypted or not self.config.profile:
            raise ValueError("Profiling is only supported in encrypted mode.")
        
        """Run profiled inference and return results."""
        if self.model is None or self.test_dataloader is None:
            raise ValueError("Model and data must be loaded before inference")
        
        predictions = []
        total_labels = 0
        correct_predictions = 0
        inference_steps = len(self.test_dataloader)
        
        crypten.communicator.get().reset_communication_stats()
        with torch.no_grad():
            with crypten.no_grad() if self.config.encrypted else torch.no_grad():
                for i, batch in enumerate(self.test_dataloader, 1):
                    result = self._process_batch(batch, i, inference_steps, max_retries_overflows=0)
                    
                    predictions.append(result['predictions'])
                    correct_predictions += result['correct_count']
                    total_labels += result['total_count']
        
        accuracy = correct_predictions / total_labels if total_labels > 0 else 0
        
        print(f"Accuracy: {accuracy}")
        
        # Communication costs
        communication_stats = crypten.communicator.get().get_communication_stats()
        logging.info(f"Communication costs:")
        logging.info(communication_stats)
        print(f"Communication_stats: {communication_stats}")

        # Model timing
        logging.info(f"Model timing:")
        logging.info(self.model.timing)
        print(f"Model timing: {self.model.timing}")

        logging.info(f"Model communication:")
        logging.info(self.model.communication)
        print(f"Model communication: {self.model.communication}")

        # Convert all predictions to lists for JSON serialization
        results = {
            'accuracy': accuracy,
            'total_labels': total_labels,
            'correct_predictions': correct_predictions,
            'predictions': [p.tolist() if hasattr(p, 'tolist') else p for p in predictions],
            "runtime": self.model.timing,
            "communication": self.model.communication
        }

        home_path = os.path.expanduser("~")
        FINETUNING_RESULTS_PATH = f"{home_path}/sprint/data/inference/runtime"
        os.makedirs(FINETUNING_RESULTS_PATH, exist_ok=True)
        with open(f"{FINETUNING_RESULTS_PATH}/results.json", "w") as f:
            json.dump(results, f)

        return results


    def _process_batch(
        self, 
        batch: Dict[str, torch.Tensor], 
        step: int, 
        total_steps: int,
        max_retries_overflows: int = 5
    ) -> Dict[str, Any]:
        """Process a single batch and handle overflows."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)
        labels = batch["label"].cpu()
        
        overflow_detected = False
        overflow_corrected = False
        retry_count = 0
        max_retries = max_retries_overflows
        
        while retry_count <= max_retries:
            # Forward pass
            logits = self.model(input_ids, attention_mask, token_type_ids)
            logits = logits.logits.cpu() if isinstance(logits, dict) else logits.cpu()
            
            if self.config.encrypted:
                logits = logits.get_plain_text()
            
            # Check for overflows in encrypted mode
            if self.config.encrypted and ((logits > 1e5).any() or (logits < -1e5).any()):
                overflow_detected = True
                if retry_count < max_retries:
                    retry_count += 1
                    self._debug_print(f"Overflows detected, retrying step {step}/{total_steps} (attempt {retry_count})")
                    continue
                else:
                    break
            else:
                if overflow_detected and retry_count > 0:
                    overflow_corrected = True
                break
        
        # Calculate predictions and accuracy
        predictions = logits.argmax(-1).cpu()
        correct_count = (predictions == labels).sum().item()
        total_count = len(labels)
        
        return {
            'predictions': predictions,
            'correct_count': correct_count,
            'total_count': total_count,
            'overflow_detected': overflow_detected,
            'overflow_corrected': overflow_corrected
        }
    
    def _get_model_type(self) -> str:
        """Determine model type from model name."""
        if "roberta" in self.config.model_name:
            return "roberta"
        elif "bert" in self.config.model_name:
            return "bert"
        else:
            raise ValueError("Model type not supported, it must contain either 'bert' or 'roberta'")
    
    def _get_model_classes(self) -> Dict[str, Any]:
        """Get appropriate model classes based on encryption mode."""
        if self.config.encrypted:
            if self.config.profile:
                from modeling.models.profiled.modeling_bert import BertForSequenceClassification
                from modeling.models.profiled.modeling_roberta import RobertaForSequenceClassification
                return {
                    "bert": BertForSequenceClassification,
                    "roberta": RobertaForSequenceClassification
                }
            else:
                from modeling.models.modeling_bert import BertForSequenceClassification
                from modeling.models.modeling_roberta import RobertaForSequenceClassification
                return {
                    "bert": BertForSequenceClassification,
                    "roberta": RobertaForSequenceClassification
                }
        else:
            from modeling.models.modeling_bert_clear import BertForSequenceClassification
            from modeling.models.modeling_roberta_clear import RobertaForSequenceClassification
            return {
                "bert": BertForSequenceClassification,
                "roberta": RobertaForSequenceClassification
            }
    
    def _decrypt_non_lora_parameters(self) -> None:
        """Decrypt non-LoRA parameters for efficiency."""
        state_dict = self.model.state_dict()
        if any("lora" in key for key in state_dict.keys()):
            lora = True
            lora_config = get_lora_config_from_model(self.model, self._get_model_type())
        
            decrypt_non_lora_parameters(
                model=self.model,
                peft_config=lora_config
            )
        
    def _debug_print(self, message: str) -> None:
        """Print debug messages if debug mode is enabled."""
        if self.config.debug:
            if self.config.encrypted:
                crypten.print(f"[Process {self.rank}]: {message}", in_order=True)
            else:
                print(f"[Process {self.rank}]: {message}")
