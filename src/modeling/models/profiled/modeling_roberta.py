# Copyright (c) 2025 SAP SE or an SAP affiliate company and sprint contributors
# SPDX-License-Identifier: Apache-2.0

from time import time

import torch

import crypten
import crypten.nn as cnn
import crypten.communicator as comm

from typing import Optional

from transformers import AutoModel, AutoModelForSequenceClassification
from collections import defaultdict

from transformers.models.roberta.modeling_roberta import RobertaEmbeddings as PlainTextRobertaEmbeddings

from modeling.models.profiled.modeling_bert_like import BertLikeEncoder, BertLikePooler, logits_soft_capping

# This is not used in the classification task, the classifier already has a linear layer for pooling
    

class RobertaPreTrainedModel(cnn.Module):
    config_class = None
    base_model_prefix = "roberta"

    def __init__(self, config, **kwargs):
        super(RobertaPreTrainedModel, self).__init__()
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs
    ):
        if cls is None:
            raise ValueError(
                "The model class should be specified"
            )
        if cls not in MODELS:
            raise ValueError(
                f"The model class should be one of {MODELS}"
            )
        #model_config = None, timing = None):
        if pretrained_model_name_or_path is None:
            raise ValueError(
                "You have to specify a pretrained model name"
            )
        if "bert" not in pretrained_model_name_or_path.lower():
            raise ValueError(
                "The pre-trained model you are loading is not a Bert model. Please make sure that your pre-trained model is a Bert model."
            )
        model_config = kwargs.pop("model_config", None)
        if model_config is None:
            print(f'''No crypten config provided, using default values:\n
                  \t softmax activation: softmax\n
                  \t hidden activation: relu\n
                  \t classifier activation: relu\n
                  \t layer norm eps: 1e-5\n
                  \t plaintext embedding: False\n''')
            model_config = {
                "softmax_act": "softmax",
                "hidden_act": "relu",
                "classifier_act": "relu",
                "layer_norm_eps": 1e-5,
                "plaintext_embedding": False
            }
        timing = kwargs.pop("timing", None)
        if timing is None:
            print(f"No timing provided, using default float dictionary:\n" )
            timing = defaultdict(float)

        communication = kwargs.pop("communication", None)
        if communication is None:
            print(f"No communication provided, using default float dictionary:\n" )
            communication = defaultdict(float)

        torch_dtype = kwargs.pop("torch_dtype", torch.float32)

        if cls == RobertaModel:
            torch_model = AutoModel.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype, *model_args, **kwargs)
        else:
            torch_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype, *model_args, **kwargs)

        config = torch_model.config

        for key, value in model_config.items():
            setattr(config, key, value)

        # Load the model
        model = cls(config, timing, communication, **kwargs)
        model.load_state_dict(torch_model.state_dict(), strict=False)
        return model
    

class RobertaModel(RobertaPreTrainedModel):
    def __init__(self, config, timing, communication,  add_pooling_layer=True, **kwargs):
        super(RobertaModel, self).__init__(config)
        self.config = config
        self.timing = timing
        self.communication = communication
        self.embeddings = PlainTextRobertaEmbeddings(config)
        self.encoder = BertLikeEncoder(config, self.timing, self.communication)
        self.pooler = BertLikePooler(config, self.timing, self.communication) if add_pooling_layer else None
        
        self.embeddings_cap = config.embeddings_cap if hasattr(config, "embeddings_cap") else None

        if not timing.keys().__contains__("Block"):
            timing["Block"] = defaultdict(float)
        if not timing.keys().__contains__("Layer"):
            timing["Layer"] = defaultdict(float)
        if not timing.keys().__contains__("Encoder"):
            timing["Encoder"] = defaultdict(float)

        if not communication.keys().__contains__("Block"):
            communication["Block"] = defaultdict(float)
        if not communication.keys().__contains__("Layer"):
            communication["Layer"] = defaultdict(float)
        


    def load_state_dict(self, state_dict, strict=False):
        super().load_state_dict(state_dict, strict)

    def train(self, mode=True):
        super().train(mode)


    def to(self, device = "cpu"):
        self.embeddings.to(device=device)
        return super().to(device)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[crypten.CrypTensor] = None,
        token_type_ids: Optional[crypten.CrypTensor] = None
    ):
        t0 = time()
        comm0 = comm.get().get_communication_stats()
        embedding_output = self.embeddings(input_ids, token_type_ids)
        if self.embeddings_cap is not None:
            embedding_output = logits_soft_capping(embedding_output, max_cap=self.embeddings_cap)

        embedding_output = crypten.cryptensor(embedding_output)
        t1_emb = time()
        comm1 = comm.get().get_communication_stats()
        self.timing["Block"]["Embedding"] += (t1_emb - t0)
        self.timing["Block"]["EmbeddingComm"] += (comm1["time"] - comm0["time"])
        self.communication["Block"]["EmbeddingRounds"] += (comm1["rounds"] - comm0["rounds"])
        self.communication["Block"]["EmbeddingBytes"] += (comm1["bytes"] - comm0["bytes"])

        encoded_layers = self.encoder(embedding_output, attention_mask)
        sequence_output = encoded_layers
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else sequence_output
        t1 = time()
        self.timing["RoBERTaTime"] += (t1-t0)

        return pooled_output


class RobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config, timing, communication, **kwargs):
        super(RobertaForSequenceClassification, self).__init__(config)
        self.num_labels = kwargs.pop("num_labels", 2)
        self.roberta = RobertaModel(config, timing, communication, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config, timing, communication)

        self.timing = timing
        self.communication = communication
        #self.post_init()

    def encrypt(self, mode=True, src=0):
        super().encrypt(mode, src)
        self.roberta.encrypt(mode, src)
        self.classifier.encrypt(mode, src)

    def load_state_dict(self, state_dict, strict=False):
        # the target state dict should not have classifier
        # embeddings copy, since they are stored in cleartext (locally)

        embed_dict = self.roberta.embeddings.state_dict()
        prepend_string = "roberta.embeddings."
        for k in embed_dict.keys():
            embed_dict[k] = state_dict[prepend_string + k]

        self.roberta.embeddings.load_state_dict(embed_dict)

        state_dict_keys = state_dict.keys()
        target_state_dict = self.state_dict().copy()
        for k in state_dict_keys:
            target_state_dict[k] = state_dict[k]

        super().load_state_dict(target_state_dict, strict)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        t0 = time()
        comm0 = comm.get().get_communication_stats()
        outputs = self.roberta(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            token_type_ids = token_type_ids
        )
        comm0_class = comm.get().get_communication_stats()
        t0_class = time()
        logits = self.classifier(outputs)
        t1 = time()
        comm1 = comm.get().get_communication_stats()

        self.timing["RobertaForSequenceClassification"] += (t1 - t0)
        self.timing["RobertaForSequenceClassificationComm"] += (comm1["time"] - comm0["time"])
        self.communication["RobertaForSequenceClassificationRounds"] += (comm1["rounds"] - comm0["rounds"])
        self.communication["RobertaForSequenceClassificationBytes"] += (comm1["bytes"] - comm0["bytes"])

        self.timing["RobertaForSequenceClassificationEncrypted"] = self.timing["RobertaForSequenceClassification"] - self.timing["Block"]["Embedding"]
        self.timing["RobertaForSequenceClassificationEncryptedComm"] = self.timing["RobertaForSequenceClassificationComm"] - self.timing["Block"]["EmbeddingComm"]

        self.communication["RobertaForSequenceClassificationEncryptedRounds"] += self.communication["RobertaForSequenceClassificationRounds"] - self.communication["Block"]["EmbeddingRounds"]
        self.communication["RobertaForSequenceClassificationEncryptedBytes"] += self.communication["RobertaForSequenceClassificationBytes"] - self.communication["Block"]["EmbeddingBytes"]

        self.timing["Block"]["Classifier"] += (t1 - t0_class)
        self.timing["Block"]["ClassifierComm"] += (comm1["time"] - comm0_class["time"])
        self.communication["Block"]["ClassifierRounds"] += (comm1["rounds"] - comm0_class["rounds"])
        self.communication["Block"]["ClassifierBytes"] += (comm1["bytes"] - comm0_class["bytes"])

        return logits



class RobertaClassificationHead(cnn.Module):
    def __init__(self, config, timing, communication):
        super(RobertaClassificationHead, self).__init__()
        self.dense = cnn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = cnn.Dropout(config.hidden_dropout_prob)
        self.out_proj = cnn.Linear(config.hidden_size, config.num_labels)
        self.timing = timing
        self.communication = communication
        self.logits_cap = config.logits_cap_classifier

    def forward(self, features, **kwargs):
        x = features.narrow(1, 0, 1).squeeze(1)
        x = self.dropout(x)
        x = self.dense(x)

        comm_tan0 = comm.get().get_communication_stats()
        x = x.tanh()
        comm_tan1 = comm.get().get_communication_stats()

        x = self.dropout(x)
        x = self.out_proj(x)

        
        if self.logits_cap is not None:
            t0_capping = time()
            comm0_capping = comm.get().get_communication_stats()
            x = logits_soft_capping(x, max_cap=self.logits_cap)

        comm1 = comm.get().get_communication_stats()
        t1 = time()

        if self.logits_cap is not None:
            self.timing["Layer"]["ClassifierSoftcapping"] += (t1 - t0_capping)
            self.timing["Layer"]["ClassifierSoftcappingCommunication"] += (comm1["time"] - comm0_capping["time"])
            self.communication["Layer"]["ClassifierSoftcappingRounds"] += (comm1["rounds"] - comm0_capping["rounds"])
            self.communication["Layer"]["ClassifierSoftcappingBytes"] += (comm1["bytes"] - comm0_capping["bytes"])

            self.timing["Layer"]["Softcapping"] += (t1 - t0_capping)
            self.timing["Layer"]["SoftcappingCommunication"] += (comm1["time"] - comm0_capping["time"])
            self.communication["Layer"]["SoftcappingRounds"] += (comm1["rounds"] - comm0_capping["rounds"])
            self.communication["Layer"]["SoftcappingBytes"] += (comm1["bytes"] - comm0_capping["bytes"])


        self.timing["Layer"]["Tanh"] += (comm_tan1["time"] - comm_tan0["time"])
        self.timing["Layer"]["TanhCommunication"] += (comm_tan1["time"] - comm_tan0["time"])
        self.communication["Layer"]["TanhRounds"] += (comm_tan1["rounds"] - comm_tan0["rounds"])
        self.communication["Layer"]["TanhBytes"] += (comm_tan1["bytes"] - comm_tan0["bytes"])
        return x

MODELS = [ 
    RobertaModel,
    RobertaForSequenceClassification
]