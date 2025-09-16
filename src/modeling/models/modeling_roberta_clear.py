# Copyright (c) 2025 SAP SE or an SAP affiliate company and sprint contributors
# SPDX-License-Identifier: Apache-2.0


from time import time

import torch
import torch.nn as nn

from typing import Optional, Tuple

from collections import defaultdict

from transformers import RobertaModel as TorchRoberta
from transformers import RobertaForSequenceClassification as TorchRobertaForSequenceClassification

from transformers import PreTrainedModel

from transformers.models.roberta.modeling_roberta import RobertaEmbeddings

from modeling.models.modeling_bert_like_clear import BertLikeEncoder, BertLikePooler, logits_soft_capping
    

class MixedFP32Embedding(RobertaEmbeddings):
    def forward(self, input_ids, token_type_ids=None, half_precision=False):
        output = super().forward(input_ids, token_type_ids)
        if half_precision:
            output = output.half()
        return output
            

class RobertaPreTrainedModel(PreTrainedModel):
    config_class = None
    base_model_prefix = "roberta"

    def __init__(self, config, **kwargs):
        super(RobertaPreTrainedModel, self).__init__(config=config, **kwargs)
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
        #model_config= None, timing = None):
        if pretrained_model_name_or_path is None:
            raise ValueError(
                "You have to specify a pretrained model name"
            )
        if "roberta" not in pretrained_model_name_or_path.lower():
            raise ValueError(
                "The pre-trained model you are loading is not a RoBerta model. Please make sure that your pre-trained model is a Bert model."
            )
        model_config= kwargs.pop("model_config", None)
        if model_config is None:
            print(f'''No model config provided, using default values:\n
                  \t softmax activation: softmax\n
                  \t hidden activation: relu\n
                  \t classifier activation: relu\n
                  \t layer norm eps: 1e-5\n
                  \t plaintext embedding: False\n
                  \t apply embeddings capping: False\n
                  \t apply logits capping attention: False
                  \t apply logits capping classifier: False\n''')
            model_config= {
                "softmax_act": "softmax",
                "hidden_act": "relu",
                "classifier_act": "relu",
                "layer_norm_eps": 1e-5,
                "plaintext_embedding": False,
                "apply_embeddings_capping": False,
                "apply_logits_cap_attention": False,
                "apply_logits_cap_classifier": False,
                "embeddings_cap": 100.0,
                "logits_cap_attention": 100.0,
                "logits_cap_classifier": 100.0
            }
        timing = kwargs.pop("timing", None)
        if timing is None:
            print(f"No timing provided, using default float dictionary:\n" )
            timing = defaultdict(float)

        if cls == RobertaModel:
            torch_model = TorchRoberta.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        else:
            torch_model = TorchRobertaForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        config = torch_model.config

        for key, value in model_config.items():
            setattr(config, key, value)

        # Load the model
        model = cls(config, timing,**kwargs)
        model.load_state_dict(torch_model.state_dict(), strict=False)
        return model
    

class RobertaModel(RobertaPreTrainedModel):
    def __init__(self, config, timing, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.timing = timing
        self.embeddings = MixedFP32Embedding(config)
        self.encoder = BertLikeEncoder(config, self.timing)
        self.pooler = BertLikePooler(config, self.timing) if add_pooling_layer else None

        self.embeddings_cap = config.embeddings_cap if hasattr(config, "embeddings_cap") else None

        self.half_precision = False
        
        if not timing.keys().__contains__("Block"):
            timing["Block"] = defaultdict(float)
        if not timing.keys().__contains__("Layer"):
            timing["Layer"] = defaultdict(float)
        if not timing.keys().__contains__("Encoder"):
            timing["Encoder"] = defaultdict(float)

    def get_extended_attention_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int], device: torch.device = None, dtype: torch.float = None
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if dtype is None:
            dtype = self.dtype

        if not (attention_mask.dim() == 2 and self.config.is_decoder):
            # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
            if device is not None:
                warnings.warn(
                    "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
                )
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask


    def load_state_dict(self, state_dict, strict=False):
        super().load_state_dict(state_dict, strict)

    def train(self, mode=True):
        super().train(mode)

    def half(self):
        # Embeddings in full precision
        self.encoder.half()
        self.half_precision = True
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ):
        t0 = time()
        embedding_output = self.embeddings(input_ids, token_type_ids, self.half_precision)
        t1_emb = time()
        self.timing["Block"]["Embedding"] += (t1_emb - t0)

        if self.embeddings_cap is not None:
            embedding_output = logits_soft_capping(embedding_output, self.embeddings_cap)
            
        
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.size(), device=input_ids.device)

        
        input_shape = input_ids.size()
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, dtype=torch.float16 if self.half_precision else torch.float32)

        encoded_layers = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = encoded_layers
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else sequence_output
        t1 = time()
        self.timing["BertTime"] += (t1-t0)

        return pooled_output


class RobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config, timing, **kwargs):
        super().__init__(config)
        self.num_labels = kwargs.pop("num_labels", 2)
        self.roberta = RobertaModel(config, timing, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config, timing)
        self.half_precision = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        outputs = self.roberta(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            token_type_ids = token_type_ids
        )
        if self.half_precision:
            outputs = outputs.to(torch.float32)
        logits = self.classifier(outputs)
        
        if self.half_precision:
            logits = logits.to(torch.float16)

        return logits
        

    def load_state_dict(self, state_dict, strict=False):
        super().load_state_dict(state_dict, strict)

    def half(self):
        self.half_precision = True
        self.roberta.half()


class RobertaClassificationHead(nn.Module):
    def __init__(self, config, timing):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.timing = timing

        self.logits_cap = config.logits_cap_classifier

    def forward(self, features, **kwargs):
        t0 = time()
        # HERE modified for CrypTen
        # x = features.narrow(1, 0, 1).squeeze(1)
        x = features[:, 0, :] # take <s> token (equiv. to [CLS]) (batch_size, hidden_size) from (batch_size, seq_len, hidden_size)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        if self.logits_cap is not None:
            x = logits_soft_capping(x, self.logits_cap)

        t1 = time()
        self.timing["Block"]["ClassificationHead"] += (t1 - t0)

        return x

MODELS = [ 
    RobertaModel,
    RobertaForSequenceClassification
]