# Copyright (c) 2025 SAP SE or an SAP affiliate company and sprint contributors
# SPDX-License-Identifier: Apache-2.0

"""Model creation and configuration utilities."""

import torch
from typing import Optional, Dict, Any
from .constants import LABELS_PER_DATASET, MODEL_DICT
from modeling.models.modeling_bert_clear import BertForSequenceClassification
from modeling.models.modeling_bert_like import CryptenBertLikeConfig
from modeling.models.modeling_roberta_clear import RobertaForSequenceClassification
from modeling.lora.lora_utils import LoraConfig, get_peft_model, get_lora_config_from_model
        


class ModelFactory:
    """Factory for creating and configuring models."""
    
    @staticmethod
    def create_model_config(
        softmax_act: str = "softmax_nn",
        hidden_act: str = "relu",
        classifier_act: str = "relu",
        logits_cap_attention: float = 50.0,
        logits_cap_classifier: Optional[float] = None,
        embeddings_cap: float = 50.0,
        **kwargs
    ) -> CryptenBertLikeConfig:
        """Create CrypTen configuration."""
        return CryptenBertLikeConfig(
            layer_norm_eps=1e-8,
            softmax_act=softmax_act,
            hidden_act=hidden_act,
            classifier_act=classifier_act,
            logits_cap_attention=logits_cap_attention,
            logits_cap_classifier=logits_cap_classifier,
            embeddings_cap=embeddings_cap
        )

    @staticmethod
    def create_lora_config(
        r: int = 8,
        lora_alpha: int = 16,
        freeze_A: bool = False,
        init_method: str = "normal",
        target_modules: list = None,
        modules_to_save: list = None,
        **kwargs
    ) -> LoraConfig:
        """Create LoRA configuration."""
        if target_modules is None:
            target_modules = ['query', 'key', 'value', 'dense']
        if modules_to_save is not None:
            if any(m not in ("classifier", "layernorm") for m in modules_to_save):
                raise ValueError("Only 'classifier' and 'layernorm' are allowed in modules_to_save.")

        return LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            freeze_A=freeze_A,
            modules_to_save=modules_to_save if modules_to_save is not None else ["classifier"],
            bias="none",
            target_modules=target_modules,
            fan_in_fan_out=False,
            init_method=init_method
        )

    @staticmethod
    def create_model(
        pretrained_model: str,
        dataset_name: str,
        model_config: CryptenBertLikeConfig,
        device: str = "cuda:0",
        precision: str = "fp32"
    ) -> torch.nn.Module:
        """Create and configure the model."""
        num_labels = LABELS_PER_DATASET.get(dataset_name)
        if num_labels is None:
            raise ValueError(f"Unknown dataset: {dataset_name}")


        pretrained_model = MODEL_DICT.get(pretrained_model) if pretrained_model in MODEL_DICT.keys() else pretrained_model

        # Create model based on type
        if "roberta" in pretrained_model:
            model = RobertaForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=pretrained_model,
                num_labels=num_labels,
                model_config=model_config
            )
        elif "bert" in pretrained_model:
            model = BertForSequenceClassification.from_pretrained(
                pretrained_model_name=pretrained_model,
                num_labels=num_labels,
                model_config=model_config
            )
        else:
            raise ValueError(f"Model: {pretrained_model} NOT supported")

        model.to(device)
        
        if precision == "fp16":
            model.half()
            
        return model

    @staticmethod
    def apply_lora(
        model: torch.nn.Module,
        lora_config: LoraConfig,
        lora_type: str = "none"
    ) -> torch.nn.Module:
        """Apply LoRA to the model if requested."""
        if lora_type in ["lora", "fa_lora"]:
            lora_config.freeze_A = (lora_type == "fa_lora")
            get_peft_model(model, lora_config)
        return model

    @staticmethod
    def load_finetuned_model(
        model_name: str,
        model_path: str,
        test_dataset: str,
        device: str,
        model_type_to_class: Dict[str, Any],
        softmax: Optional[str] = None,
        hidden_act: Optional[str] = None,
        cap: Optional[float] = None
    ) -> torch.nn.Module:
        """Load a fine-tuned model from checkpoint."""
        # Extract model type from model name
        torch_name = model_name.split("_")[0] if model_name.split("_")[0] != "best" else model_name.split("_")[1]
        model_type = "roberta" if "roberta" in torch_name else "bert"
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract configuration and state dict
        if isinstance(checkpoint, dict):
            model_config = CryptenBertLikeConfig().from_dict(checkpoint["config"])
            state_dict = checkpoint["state_dict"]
            lora_config = None
            if "lora_config" in checkpoint and any("lora" in key for key in state_dict.keys()):
                lora_config = LoraConfig().from_dict(checkpoint["lora_config"])
        else:
            model_config = checkpoint.config
            state_dict = checkpoint.state_dict()
            lora_config = None
            if any("lora" in key for key in state_dict.keys()):
                lora_config = get_lora_config_from_model(checkpoint, model_type)
        
        # Override configuration if specified
        if cap is not None:
            model_config.logits_cap_attention = cap
            model_config.logits_cap_classifier = cap
            model_config.embeddings_cap = cap
        
        if softmax is not None:
            model_config.softmax_act = softmax
        
        if hidden_act is not None:
            model_config.hidden_act = hidden_act
            
        # Handle softmax_max validation
        if (model_config.softmax_act == "softmax_max" and 
            cap is None and model_config.logits_cap_attention is None):
            print("Warning: Cap must be specified when using softmax_max, setting softmax to softmax")
            model_config.softmax_act = "softmax"
        
        # Create model
        model = model_type_to_class[model_type].from_pretrained(
            pretrained_model_name_or_path=torch_name,
            num_labels=LABELS_PER_DATASET[test_dataset],
            model_config=model_config,
        )
        
        # Apply LoRA if present
        if lora_config is not None:
            get_peft_model(model, lora_config)
        
        # Load state dict and prepare model
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        
        print(f"Model loaded from {model_path}, model type: {model_type}, "
              f"softmax: {model_config.softmax_act}, hidden_act: {model_config.hidden_act}, cap: {cap}")
        
        return model
