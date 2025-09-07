import time

import torch
import torch.nn as nn

from typing import Optional, Tuple

from transformers import AutoModel, AutoModelForSequenceClassification
from collections import defaultdict

from transformers.models.bert.modeling_bert import BertEmbeddings

from modeling.models.modeling_bert_like_clear import BertLikeEncoder, BertLikePooler, logits_soft_capping



class BertPreTrainedModel(nn.Module):
    config_class = None
    load_tf_weights = None
    base_model_prefix = "bert"
    authorized_missing_keys = None
    authorized_unexpected_keys = None

    def __init__(self, config, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        self.config = config

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name, 
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
        if pretrained_model_name is None:
            raise ValueError(
                "You have to specify a pretrained model name"
            )
        if "bert" not in pretrained_model_name.lower():
            raise ValueError(
                "The pre-trained model you are loading is not a Bert model. Please make sure that your pre-trained model is a Bert model."
            )
        model_config= kwargs.pop("model_config", None)
        if model_config is None:
            print(f'''No model config provided, using default values:\n
                  \t softmax activation: softmax\n
                  \t hidden activation: relu\n
                  \t classifier activation: relu\n
                  \t layer norm eps: 1e-5\n
                  \t plaintext embedding: False\n''')
            model_config= {
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

       
        #torch_model = AutoModel.from_pretrained(pretrained_model_name)
        if cls == BertModel:
            torch_model = AutoModel.from_pretrained(pretrained_model_name)
        elif cls == BertForSequenceClassification:
            torch_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name)

        config = torch_model.config
        for key, value in model_config.items():
            setattr(config, key, value)
        
        # Load model
        model = cls(config, timing, *model_args, **kwargs)
        model.load_state_dict(torch_model.state_dict(), strict=False)
        return model
    


class BertModel(BertPreTrainedModel):
    def __init__(self, config, timing):
        super(BertModel, self).__init__(config)
        #self.plaintext_embedding = config.plaintext_embedding
    
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertLikeEncoder(config, timing)
        self.pooler = BertLikePooler(config, timing)
        if not timing.keys().__contains__("Block"):
            timing["Block"] = defaultdict(float)
        if not timing.keys().__contains__("Layer"):
            timing["Layer"] = defaultdict(float)
        if not timing.keys().__contains__("Encoder"):
            timing["Encoder"] = defaultdict(float)

        self.timing = timing

        self.embeddings_cap = config.embeddings_cap if hasattr(config, "embeddings_cap") else 100.0

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

    def encrypt(self, mode=True, src=0):
        super().encrypt(mode, src)
        #if self.plaintext_embedding:
        #    self.embeddings.encrypt(mode=False, src = src)

    def train(self, mode=True):
        super().train(mode)
        #if self.plaintext_embedding:
        #    self.embeddings.train(mode=False)
    

    
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        t0 = time.time()
        embedding_output = self.embeddings(input_ids, token_type_ids)
        t1_emb = time.time()
        self.timing["Block"]["Embedding"] += (t1_emb - t0)

        if self.embeddings_cap is not None:
            embedding_output = logits_soft_capping(embedding_output, max_cap=self.embeddings_cap)

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.size(), device=input_ids.device)
        
        input_shape = input_ids.size()
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        encoded_layers = self.encoder(embedding_output, attention_mask)
        sequence_output = encoded_layers
        pooled_output = self.pooler(sequence_output)
        t1 = time.time()
        self.timing["BertTime"] += (t1-t0)
        return pooled_output
        #return sequence_output, pooled_output
    

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, timing, *model_args, **kwargs):
        super().__init__(config)
        self.num_labels = kwargs.pop("num_labels", 2)
        self.fit_size = kwargs.pop("fit_size", 768)

        self.bert = BertModel(config, timing)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.logits_cap = config.logits_cap_classifier
        # dropout not used
        #self.dropout = nn.Dropout(config.dropout)
        #self.classifier_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        #self.classifier_act = config.classifier_act
        #self.classifier_act_fn = ACT2FN[config.classifier_act]
        # Only for teacher - student training
        #self.fit_dense = nn.Linear(config.hidden_size, fit_size)

    def load_state_dict(self, state_dict, strict=False):
        # the target state dict should not have classifier
        #state_dict_keys = state_dict.keys()
        #target_state_dict = self.state_dict().copy()
        #for k in target_state_dict.keys():
        #    target_state_dict[k] = state_dict[k]

        super().load_state_dict(state_dict, strict)

        #super().load_state_dict(target_state_dict, strict)

        #try :
        #    super().load_state_dict(state_dict, strict)
        #except KeyError as e:
        #    if "classifier" not in e.args[0]:
        #        raise e

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        pooled_output = super().forward(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask,
            )
        # dropout not used in original implementation
        #pooled_output = self.dropout(pooled_output)
        # HERE Add layer norm to avoid overflow in the loss
        #pooled_output = self.classifier_layer_norm(pooled_output)
        
        #t0 = time.time()
        #logits = self.classifier_act_fn(pooled_output)
        #t1_act = time.time()

        t0_lin = time.time()
        logits = self.classifier(pooled_output)
        t1_lin = time.time()

        if self.logits_cap is not None:
            logits = logits_soft_capping(logits, max_cap=self.logits_cap)

        #self.timing["Layer"][f"Classifier_{self.classifier_act}"] += (t1_act - t0)
        
        self.timing["Layer"]["Linear"] += (t1_lin - t0_lin)
        #logits_clear = logits.get_plain_text()
        return logits


    

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
    ):
        return self.embeddings(input_ids)


MODELS = [ 
    BertModel,
    BertForSequenceClassification
]
