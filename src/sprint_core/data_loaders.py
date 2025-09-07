"""Data loading utilities for SPRINT fine-tuning."""

import os
import torch
import datasets
from torch.utils.data import DataLoader, TensorDataset
from opacus.data_loader import DPDataLoader as PoissonDataLoader
from typing import Dict, Any, Literal
from .constants import TOKENIZED_DATASETS_PATH, SUPPORTED_MODEL_TYPES
from .eval_utils import get_toy_tokenized_data


def custom_collate_fn(batch):
    """Custom collate function for batching data."""
    input_ids = torch.stack([item[0].clone().detach() for item in batch])
    attention_mask = torch.stack([item[1].clone().detach() for item in batch])
    token_type_ids = torch.stack([item[2].clone().detach() for item in batch])
    labels = torch.stack([item[3].clone().detach() for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "label": labels
    }


def _validate_model_type(model_type: str) -> str:
    """Validate and normalize model type."""
    if model_type not in SUPPORTED_MODEL_TYPES:
        if "roberta" in model_type:
            return "roberta"
        elif "bert" in model_type:
            return "bert"
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    return model_type


def _load_dataset(
    dataset_name: str, 
    model_type: str, 
    dataset_type: str,
    n_samples: int = -1,
) -> TensorDataset:
    """Load and prepare dataset as TensorDataset."""
    model_type = _validate_model_type(model_type)
    
    if "toy" in dataset_name:
        n_samples = n_samples if n_samples > 0 else 100
        dataset = get_toy_tokenized_data(
            num_samples=n_samples,
            num_labels=2,
            model_type=model_type,
            seq_length=128,
        )
    else:
        file_name = f"{dataset_name}_{dataset_type}_dataset/"
        dataset_path = f"{TOKENIZED_DATASETS_PATH}/{model_type}/{dataset_name}/{file_name}"
        
        try:
            dataset = datasets.load_from_disk(dataset_path)
            if n_samples > 0:
                dataset = dataset.select(range(min(n_samples, len(dataset))))
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset from {dataset_path}: {e}")

    # Convert to tensors
    input_ids = torch.tensor(dataset["input_ids"])
    attention_mask = torch.tensor(dataset["attention_mask"])
    labels = torch.tensor(dataset["label"])
    
    # Handle token_type_ids based on model type
    if model_type == "bert":
        token_type_ids = torch.tensor(dataset["token_type_ids"])
    elif model_type == "roberta":
        token_type_ids = torch.zeros_like(input_ids)
    else:
        raise ValueError(f"Model type: {model_type} NOT supported")

    return TensorDataset(input_ids, attention_mask, token_type_ids, labels)


class DataLoaderFactory:
    """Factory for creating different types of data loaders."""
    
    @staticmethod
    def create_poisson_dataloader(
        dataset_name: str = "sst2",
        model_type: str = "bert",
        batch_size: int = 512,
    ) -> PoissonDataLoader:
        """Create a DataLoader that performs Poisson subsampling for each epoch."""
        dataset = _load_dataset(dataset_name, model_type, "train")
        q = batch_size / len(dataset)
        
        return PoissonDataLoader(
            dataset, 
            sample_rate=q, 
            collate_fn=custom_collate_fn
        )

    @staticmethod
    def create_shuffle_dataloader(
        dataset_name: str = "sst2",
        model_type: str = "bert",
        batch_size: int = 2048,
        shuffle: bool = True,
        n_samples: int = -1
    ) -> DataLoader:
        """Create a DataLoader with shuffling (for comparison purposes)."""
        dataset = _load_dataset(dataset_name, model_type, "train", n_samples=n_samples)

        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=custom_collate_fn, 
            drop_last=True
        )

    @staticmethod
    def create_validation_dataloader(
        dataset_name: str = "sst2",
        model_type: str = "bert",
        batch_size: int = 32,
        shuffle: bool = False,
        n_samples: int = -1
    ) -> DataLoader:
        """Create a DataLoader for validation."""
        dataset = _load_dataset(dataset_name, model_type, "val", n_samples=n_samples)

        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=custom_collate_fn
        )
    
    @staticmethod
    def create_test_dataloader(
        dataset_name: str = "sst2",
        model_type: str = "bert",
        n_samples: int = -1,
        batch_size: int = 32,
        shuffle: bool = False
    ) -> DataLoader:
        """Create a DataLoader for testing."""
        if "toy" in dataset_name:
            dataset = _load_dataset(dataset_name, model_type, "test", n_samples=n_samples)
        else:
            dataset = _load_dataset(dataset_name, model_type, "val", n_samples=n_samples)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=custom_collate_fn
        )

    @staticmethod
    def get_dataloader(
        dataset_name: str,
        model_type: str,
        dataloader_type: Literal["poisson", "shuffle", "validation", "test"],
        batch_size: int,
        **kwargs
    ) -> DataLoader:
        """Get appropriate dataloader based on type."""
        if dataloader_type == "poisson":
            return DataLoaderFactory.create_poisson_dataloader(
                dataset_name, model_type, batch_size
            )
        elif dataloader_type == "shuffle":
            return DataLoaderFactory.create_shuffle_dataloader(
                dataset_name, model_type, batch_size, **kwargs
            )
        elif dataloader_type == "validation":
            return DataLoaderFactory.create_validation_dataloader(
                dataset_name, model_type, batch_size, **kwargs
            )
        elif dataloader_type == "test":
            return DataLoaderFactory.create_test_dataloader(
                dataset_name, model_type, batch_size, **kwargs
            )
        else:
            raise ValueError(f"Unknown dataloader type: {dataloader_type}")
