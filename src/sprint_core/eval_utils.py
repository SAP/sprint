import json
import pickle
import crypten.communicator
import torch
import crypten
from time import time

import os

from crypten.config import cfg

import datasets



zeroed_comm_stats = {
    "rounds": 0,
    "bytes": 0,
    "time": 0,
}


non_train_communication_keys = [
    "experiment_name", 
    "train", 
    "per_sample_test", 
    "test", 
    "batch"
]

class EvalConfig():
    def __init__(
            self,
            lr=0.001,
            epsilon=1.0,
            delta=1e-5,
            clipping_threshold=10.0,
            smoothing_factor=1e-3,
            per_class_samples=1000,
            num_epochs=2,
            batch_size=10,
            num_labels=2,
            vocab_size=30522,
            softmax_act="softmax",
            hidden_act="relu",
            classifier_act="relu",
            layer_norm_eps=1e-5,
            sample_rate=1.0
            ):
        self.lr = lr
        self.epsilon = epsilon
        self.delta = delta
        self.clipping_threshold = clipping_threshold
        self.smoothing_factor = smoothing_factor
        self.per_class_samples = per_class_samples
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.vocab_size = vocab_size
        self.softmax_act = softmax_act
        self.hidden_act = hidden_act
        self.classifier_act = classifier_act
        self.layer_norm_eps = layer_norm_eps
        self.sample_rate = sample_rate

    def __str__(self) -> str:
        return f"EvalConfig(lr={self.lr}, epsilon={self.epsilon}, delta={self.delta}, clipping_threshold={self.clipping_threshold}, smoothing_factor={self.smoothing_factor}, per_class_samples={self.per_class_samples}, num_epochs={self.num_epochs}, batch_size={self.batch_size}, num_labels={self.num_labels}, vocab_size={self.vocab_size}, softmax_act={self.softmax_act}, hidden_act={self.hidden_act}, classifier_act={self.classifier_act}, layer_norm_eps={self.layer_norm_eps}, sample_rate={self.sample_rate})"


class Timing():
    def __init__(
        self,
        experiment_name=""
    ):
        self.experiment_name = experiment_name
        self.forward = 0.0
        self.backward = 0.0
        self.clip = 0.0
        self.noise_sample = 0.0
        self.param_update = 0.0
        self.epoch = 0.0
        self.batch = 0.0
        self.loss = 0.0
        self.train = 0.0
        self.pert_param = 0.0
        self.test = 0.0
        self.per_sample_test = 0.0
        self.subsampling = 0.0
        self.grad_accum = 0.0
        self.validation = 0.0

    def reset(self):
        for key in self.__dict__:
            if key != "experiment_name":
                self[key] = 0.0

    def update(self, key, value):
        self[key] = value

    def add(self, key, value):
        self[key] += value

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        return setattr(self, key, value)
    
    def __str__(self) -> str:
        return f"Timing for {self.experiment_name}:\n\tforward: {self.forward}\n\tbackward: {self.backward}\n\tclip: {self.clip}\n\tgrad_accu: {self.grad_accum}\n\tnoise_sample: {self.noise_sample}\n\tparam_update: {self.param_update}\n\tepoch: {self.epoch}\n\tbatch: {self.batch}\n\tloss: {self.loss}\n\ttrain: {self.train}\n\tpert_param: {self.pert_param}\n\tsubsampling: {self.subsampling}\n\t\tvalidation: {self.validation}\n\n\ttest: {self.test}\n\t per_sample_test: {self.per_sample_test}\n"
    
    def __repr__(self) -> str:
        return f"Timing for {self.experiment_name}:\n\tforward: {self.forward}\n\tbackward: {self.backward}\n\tclip: {self.clip}\n\tgrad_accu: {self.grad_accum}\n\tnoise_sample: {self.noise_sample}\n\tparam_update: {self.param_update}\n\tepoch: {self.epoch}\n\tbatch: {self.batch}\n\tloss: {self.loss}\n\ttrain: {self.train}\n\tpert_param: {self.pert_param}\n\tsubsampling: {self.subsampling}\n\t\tvalidation: {self.validation}\n\n\ttest: {self.test}\n\t per_sample_test: {self.per_sample_test}\n"
    
    # json serialization
    def to_json(self):
        return json.dumps(self.__dict__)
    
    def to_dict(self):
        return self.__dict__
    
    def compute_per_epochs_timing(self, num_epochs=1, num_batches=1):
        for key in self.__dict__:
            if key not in ["experiment_name", "train", "per_sample_test", "test", "validation", "batch"]:
                self[key] = self[key] / num_epochs
            if key == "batch":
                self["batch"] = self["batch"] / (num_epochs * num_batches)
    
    def compute_per_sample_test_timing(self, num_samples=1):
        self.per_sample_test = self.test / num_samples
        

class Communication():
    def __init__(
        self,
        experiment_name=""
    ):
        # Each communication stats 
        self.experiment_name = experiment_name
        self.forward = zeroed_comm_stats.copy()
        self.backward = zeroed_comm_stats.copy()
        self.clip = zeroed_comm_stats.copy()
        self.noise_sample = zeroed_comm_stats.copy()
        self.param_update = zeroed_comm_stats.copy()
        self.epoch = zeroed_comm_stats.copy()
        self.batch = zeroed_comm_stats.copy()
        self.loss = zeroed_comm_stats.copy()
        self.train = zeroed_comm_stats.copy()
        self.pert_param = zeroed_comm_stats.copy()
        self.test = zeroed_comm_stats.copy()
        self.per_sample_test = zeroed_comm_stats.copy()
        self.subsampling = zeroed_comm_stats.copy()
        self.grad_accum = zeroed_comm_stats.copy()

    def reset(self):
        for key in self.__dict__:
            if key != "experiment_name":
                self[key] = {
                    "rounds": 0,
                    "bytes": 0,
                    "time": 0
                }

    def update(self, key, value):
        self[key] = value

    def add(self, key, value):
        self[key]["rounds"] += value["rounds"]
        self[key]["bytes"] += value["bytes"]
        self[key]["time"] += value["time"]

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        return setattr(self, key, value)
    
    def __str__(self) -> str:
        return f"Communication for {self.experiment_name}:\n\tForward: {self.forward}\n\tBackward: {self.backward}\n\tClip: {self.clip}\n\tGrad Accu: {self.grad_accum}\n\tNoise Sample: {self.noise_sample}\n\tParam Update: {self.param_update}\n\tEpoch: {self.epoch}\n\tBatch: {self.batch}\n\tLoss: {self.loss}\n\tTrain: {self.train}\n\tPert Param: {self.pert_param}\n\tSubsampling: {self.subsampling}\n\n\tTest: {self.test}\n\tPer Sample Test: {self.per_sample_test}\n"
    
    def __repr__(self) -> str:
        return f"Communication for {self.experiment_name}:\n\tForward: {self.forward}\n\tBackward: {self.backward}\n\tClip: {self.clip}\n\tGrad Accu: {self.grad_accum}\n\tNoise Sample: {self.noise_sample}\n\tParam Update: {self.param_update}\n\tEpoch: {self.epoch}\n\tBatch: {self.batch}\n\tLoss: {self.loss}\n\tTrain: {self.train}\n\tPert Param: {self.pert_param}\n\tSubsampling: {self.subsampling}\n\n\tTest: {self.test}\n\tPer Sample Test: {self.per_sample_test}\n"
    
    # json serialization
    def to_json(self):
        return json.dumps(self.__dict__)
    
    def to_dict(self):
        return self.__dict__
    
    def compute_per_epochs_communication(self, num_epochs=1, num_batches = 1):
        # train communication stats are the sum of all the communication stats
        self.train["rounds"] = sum([self[key]["rounds"] for key in self.__dict__ if key not in non_train_communication_keys])
        self.train["bytes"] = sum([self[key]["bytes"] for key in self.__dict__ if key not in non_train_communication_keys])
        self.train["time"] = sum([self[key]["time"] for key in self.__dict__ if key not in non_train_communication_keys])
        for key in self.__dict__:
            if key != "experiment_name" and key != "train":
                self[key]["rounds"] = self[key]["rounds"] / num_epochs
                self[key]["bytes"] = self[key]["bytes"] / num_epochs
                self[key]["time"] = self[key]["time"] / num_epochs
            if key == "batch":
                self["batch"]["rounds"] = self["batch"]["rounds"] / (num_epochs * num_batches)
                self["batch"]["bytes"] = self["batch"]["bytes"] / (num_epochs * num_batches)
                self["batch"]["time"] = self["batch"]["time"] / (num_epochs * num_batches)


def get_tokenized_data(
        dataset_type="train",
        model_type="bert",
        dataset_name="mrpc",
        per_class_samples=-1, 
        num_labels=None,
        type_vocab_size=None, 
        vocab_size=30522,
        one_hot=False,
        dynamic_padding=False,
        no_crypten=False,
        seq_length=128
    ):
    if model_type not in ["bert", "roberta"]:
        if "roberta" in model_type:
            model_type = "roberta"
        elif "bert" in model_type:
            model_type = "bert"
        else:
            raise ValueError("Model type must be one of 'bert', 'roberta'")
    if dataset_name == "toy":
        return get_toy_tokenized_data(
            num_samples= per_class_samples * 2 if per_class_samples != -1 else 512,
            num_labels=num_labels if num_labels is not None else 2,
            model_type=model_type,
            seq_length=seq_length,
            no_crypten=no_crypten
        )

    supported_datasets = ["cola", "sst2", "mrpc", "stsb", "rte", "toy", "qqp", "mnli", "qnli"]

    if dataset_type not in ["train", "val", "test"]:
        raise ValueError("dataset_type must be one of 'train', 'val', 'test'")
    if dataset_name not in supported_datasets:
        raise ValueError(f"dataset_name must be one of {supported_datasets}")

    file_name = f"{dataset_name}_{dataset_type}_dataset/"
    try:
        #with open(file_name, 'rb') as f:
        dataset = datasets.load_from_disk(file_name)
            #dataset = pickle.load(f)
    except FileNotFoundError:
        try:
            dataset = datasets.load_from_disk(f'{dataset_name}/{file_name}')
        except FileNotFoundError:
            try:
                dataset = datasets.load_from_disk(f'tokenized_datasets/{dataset_name}/{file_name}')
            except FileNotFoundError:
                try:
                    home_path = os.getenv("HOME")

                    dataset = datasets.load_from_disk(f'{home_path}/finetuning/experiments/transformers/tokenized_datasets/{model_type}/{dataset_name}/{file_name}')
                        #dataset = pickle.load(f)
                except FileNotFoundError:
                    try:
                        aws_path = os.getenv("HOME") + "/aws-launcher-tmp"
                        dataset = datasets.load_from_disk(f'{aws_path}/{model_type}/{dataset_name}/{file_name}')
                    except FileNotFoundError:
                        raise FileNotFoundError(f"Tokenized {dataset_name} {dataset_type} dataset not found")

    if dataset == "cola":
        input_ids, attention_mask, labels = dataset.dataset.tensors
        #token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        type_vocab_size = 2
    else:
        input_ids = torch.tensor(dataset["input_ids"])
        attention_mask = torch.tensor(dataset["attention_mask"])
        labels = torch.tensor(dataset["label"])
    
    if model_type == "bert" and dataset != "cola":
        token_type_ids = torch.tensor(dataset["token_type_ids"])
    else:
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)


    if no_crypten==False and  per_class_samples != -1:
        if crypten.communicator.get().get_rank() == 0:
            if dataset_type == "test":
                # Test sets have -1 labels
                selected_indices = torch.tensor(list(range(2*per_class_samples)))
            else:
                indices_1 = (labels == 1).nonzero().squeeze()
                indices_0 = (labels == 0).nonzero().squeeze()
                indices_1 = indices_1[torch.randperm(len(indices_1))[:per_class_samples]]
                indices_0 = indices_0[torch.randperm(len(indices_0))[:per_class_samples]]
                # Here you need to concatenate and shuffle the indices
                selected_indices = torch.cat((indices_1, indices_0)) 
                # Shuffle the indices
                # Simulate the single party which will shuffle the indices on its own
        
                selected_indices = selected_indices[torch.randperm(len(selected_indices))]
        # To avoid IndexError: tuple index out of range when the communicator is 
        # verbose, since it works only with encrypted tensors
            with cfg.temp_override({"communicator.verbose": False}):
                crypten.communicator.get().broadcast(selected_indices, src=0)
        else:
            selected_indices = torch.zeros(2*per_class_samples, dtype=torch.long)
            with cfg.temp_override({"communicator.verbose": False}):
                crypten.communicator.get().broadcast(selected_indices, src=0)

        print(selected_indices.shape)


        selected_ids = input_ids[selected_indices]
        selected_type_ids = token_type_ids[selected_indices]
        selected_masks = attention_mask[selected_indices]
        selected_labels = labels[selected_indices]
    else:
        selected_ids = input_ids
        selected_type_ids = token_type_ids
        selected_masks = attention_mask
        selected_labels = labels

    if type_vocab_size is None:
        type_vocab_size = torch.unique(selected_type_ids).size(0)
    
    if num_labels is None:
        num_labels = torch.unique(selected_labels).size(0)


    if no_crypten==False:
        if one_hot:
            input_id_one_hot = torch.nn.functional.one_hot(selected_ids, num_classes=vocab_size)
            input_id_enc = crypten.cryptensor(input_id_one_hot)
            type_ids_one_hot = torch.nn.functional.one_hot(selected_type_ids, num_classes=type_vocab_size)
            type_ids_enc = crypten.cryptensor(type_ids_one_hot)
        else:
            input_id_enc = crypten.cryptensor(selected_ids)
            type_ids_enc = crypten.cryptensor(selected_type_ids)

        attention_mask_enc = crypten.cryptensor(selected_masks)

    # Here need post processing since if we do pre-processing each party will have different data
    # due to the shuffling
    if dynamic_padding:
        selected_ids = [t for t in selected_ids]
        selected_type_ids = [t for t in selected_type_ids]
        selected_masks = [t for t in selected_masks]
        input_id_enc = [t for t in input_id_enc]
        type_ids_enc = [t for t in type_ids_enc]
        attention_mask_enc = [t for t in attention_mask_enc]

        for i in range(len(selected_ids)):
            padding_idx = (selected_ids[i] == 0).nonzero()[0]
            selected_ids[i] = selected_ids[i][:padding_idx]
            selected_type_ids[i] = selected_type_ids[i][:padding_idx]
            selected_masks[i] = selected_masks[i][:padding_idx]
            input_id_enc[i] = input_id_enc[i][:padding_idx]
            type_ids_enc[i] = type_ids_enc[i][:padding_idx]
            attention_mask_enc[i] = attention_mask_enc[i][:padding_idx]

    if no_crypten:
        return {
            "input_ids": selected_ids,
            "attention_mask": selected_masks,
            "token_type_ids": selected_type_ids,
            "label": selected_labels
        }
    else:
        if dataset_type == "test":
            labels_enc = None
        else:
            labels_one_hot = torch.nn.functional.one_hot(selected_labels, num_classes=num_labels)
            labels_enc = crypten.cryptensor(labels_one_hot)


        return {
            "input_ids": selected_ids,
            "attention_mask": selected_masks,
            "token_type_ids": selected_type_ids,
            "label": selected_labels,
            "input_id_enc": input_id_enc,
            "attention_mask_enc": attention_mask_enc,
            "attention_mask": selected_masks,
            "token_type_ids_enc": type_ids_enc,
            "token_type_ids": selected_type_ids,
            "labels_enc": labels_enc
        }


def get_toy_tokenized_data(
    num_samples=100,
    num_labels=2,
    model_type="roberta",
    seq_length=128,
    no_crypten=False
):
    if model_type not in ["bert", "roberta"]:
        raise ValueError("Model type must be one of 'bert', 'roberta'")
    
    vocab_size = 30522 if model_type == "bert" else 50265

    input_ids = torch.randint(0, vocab_size, (num_samples, seq_length), dtype=torch.long)
    attention_mask = torch.ones((num_samples, seq_length), dtype=torch.long)
    token_type_ids = torch.zeros((num_samples, seq_length), dtype=torch.long)
    labels = torch.randint(0, num_labels, (num_samples,))

    if no_crypten:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "label": labels
        }
    else:
        type_ids_enc = crypten.cryptensor(token_type_ids)
        attention_mask_enc = crypten.cryptensor(attention_mask)
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_labels)
        labels_enc = crypten.cryptensor(labels_one_hot)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "label": labels,
            "attention_mask_enc": attention_mask_enc,
            "token_type_ids_enc": type_ids_enc,
            "labels_enc": labels_enc
        }



def print_and_rank_timing_from_model(timing, keys = ["Block", "Layer"], experiment="BERT"):
    """Print and rank the timing of the model for each of the keys provided"""
    print(f"Timing for {experiment}")
    for key in keys:
        print("\n")
        print(f"\t{key}")
        timing_dict = dict(timing[key])
        comm_dict = {k: v for k, v in timing_dict.items() if "Comm" in k}
        time_dict = {k: v for k, v in timing_dict.items() if k not in comm_dict}
        
        # sort the timing dict according to the values
        sorted_timing = sorted(time_dict.items(), key=lambda x: x[1], reverse=True)
        for k, v in sorted_timing:
            print(f"\t\t{k}: {v}")


def train_and_profile_tiny_bert(
    model,
    training_data,
    timing: Timing,
    optimizer,
    loss_fn,
    plaintext_embedding=False,
    verbose=False,
    eval_config=EvalConfig(),
):
    if plaintext_embedding:
        ids = training_data["input_ids"]
    else:
        ids = training_data["input_id_enc"]
    attention_mask = training_data["attention_mask_enc"]
    labels = training_data["labels_enc"]
    losses = []
    n_batches = len(ids) // eval_config.batch_size
    
    
    for epoch in range(eval_config.num_epochs):
        epoch_loss = 0.0
        tic_epoch = time()
        for i in range(0, len(ids), eval_config.batch_size):
            tic_batch = time()
            batch_ids = ids[i:i+eval_config.batch_size]
            batch_masks = attention_mask[i:i+eval_config.batch_size]
            batch_labels = labels[i:i+eval_config.batch_size]

            optimizer.zero_grad()
            tic_fw = time()
            logits = model(batch_ids, attention_mask=batch_masks)
            toc_fw = time()
            timing.add("forward", toc_fw - tic_fw)
            tic_loss = time()
            loss = loss_fn(logits, batch_labels)
            toc_loss = time()
            timing.add("loss", toc_loss - tic_loss)

            tic_grad = time()
            loss.backward()
            toc_grad = time()
            timing.add("backward", toc_grad - tic_grad)
            
            tic_param = time()
            optimizer.step()
            toc_param = time()
            timing.add("param_update", toc_param - tic_param)
            if verbose:
                print(f"Epoch: {epoch} Batch: {i} Loss: {loss.get_plain_text()}")
            epoch_loss += loss.get_plain_text()
            toc_batch = time()
            timing.add("batch", toc_batch - tic_batch)

        toc_epoch = time()
        timing.add("epoch", toc_epoch - tic_epoch)
        final_loss = epoch_loss/eval_config.batch_size
        print(f"Epoch: {epoch} Loss: {final_loss} Time: {toc_epoch - tic_epoch} s")
        losses.append(final_loss)

    timing.update("train", timing.__getitem__("epoch"))
    timing.compute_per_epochs_timing(eval_config.num_epochs, num_batches=n_batches)

    return losses



