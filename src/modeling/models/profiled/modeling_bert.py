import time

import crypten.communicator
import torch
import torch.nn.functional as F

import crypten
import crypten.nn as cnn
import crypten.communicator as comm

from typing import Optional

from transformers import AutoModel
from collections import defaultdict

# Import BertEmbeddings from transformers as PlainTextBertEmbeddings
from transformers.models.bert.modeling_bert import BertEmbeddings as PlainTextBertEmbeddings

from modeling.models.profiled.modeling_bert_like import BertLikeEncoder, BertLikePooler, logits_soft_capping


"""
Knott on limitations of reciprocal function in CrypTen:
    Because of the limitations of encryption, functions in CrypTen are all constructed 
    from compositions of linear operations and logical comparisons. Because of this, 
    some functions are implemented using functional approximations. In this case, we 
    use Newton-Raphson iterations to implement the reciprocal function.

    Because of how Newton-Raphson (and other iterative methods work), convergence is only 
    guaranteed within a range of inputs. The range for reciprocal is approximately [-500, 500]. 
    So, the denominator of any division (by a cryptensor) must be within this range.

    Here, you set the value of a to 666, which is outside of this range. If you reduce this value, 
    division should work. We understand that this puts limitations on the kinds of inputs you can 
    use, but there is currently no known way to make this work for all inputs and we had to make 
    design choices here, and values in many machine learning workflows tend to be limited to a 
    small range (especially when using normalization techniques).
"""


class BertEmbeddings(cnn.Module):
    def __init__(self, config, timing, communication):
        super(BertEmbeddings, self).__init__()
        #self.padding_idx = 0
        self.word_embeddings = cnn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = cnn.Embedding(config.max_position_embeddings, config.hidden_size)        
        self.token_type_embeddings = cnn.Embedding(config.type_vocab_size, config.hidden_size)
        
        self.LayerNorm = cnn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = cnn.Dropout(config.hidden_dropout_prob)
        self.apply_LoRA = config.apply_LoRA
        
        self.config = config
        self.timing = timing
        self.communication = communication

    def encrypt(self, mode=True, src=0):
        super().encrypt(mode, src)
        return super().encrypt(mode, src)

    def forward(
            self, 
            input_ids: crypten.CrypTensor, 
            token_type_ids: Optional[crypten.CrypTensor] = None):
        t0_emb = time.time()
        comm0_emb = comm.get().get_communication_stats()
        
        #if input_ids is not None:
        input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        #input_embeds_clear = input_embeds.get_plain_text()
        #input_embeds_weight_clear = self.word_embeddings.weight.get_plain_text()
        

        # The input is one hot encoded, need to remove last dimension that is vocab size
        input_shape = input_ids.size()[:-1]
        seq_length = input_shape[1]
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(input_shape)
        position_ids = F.one_hot(position_ids, num_classes=self.config.max_position_embeddings)
        # If no cryptensor, position embeddings weights are not updated
        
        position_ids = crypten.cryptensor(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long)
            token_type_ids = F.one_hot(token_type_ids, num_classes=self.config.type_vocab_size).to(torch.float32)
            token_type_ids = crypten.cryptensor(token_type_ids)

        position_embeddings = self.position_embeddings(position_ids) # (bs, max_seq_length, dim)
        #position_embeddings_clear = position_embeddings.get_plain_text()

        token_type_embeddings = self.token_type_embeddings(token_type_ids) # (bs, max_seq_length, dim)
        #token_type_embeddings_clear = token_type_embeddings.get_plain_text()

        t1_emb = time.time()
        comm1_emb = comm.get().get_communication_stats()

        embeddings = input_embeds + position_embeddings + token_type_embeddings # (bs, max_seq_length, dim)
        #token_type_embeddings_weight_clear = self.token_type_embeddings.weight.get_plain_text()
        #embeddings_clear_pre_norm = embeddings.get_plain_text()
        
        t0_ln = time.time()
        comm0_ln = comm.get().get_communication_stats()
        embeddings = self.LayerNorm(embeddings) # (bs, max_seq_length, dim)
        t1_ln = time.time()
        comm1_ln = comm.get().get_communication_stats()

        embeddings = self.dropout(embeddings)

        #embeddings_clear = embeddings.get_plain_text()
        #embeddings_clear_max = embeddings_clear.abs().max()
        

        self.timing["Layer"]["LayerNorm"] += (t1_ln - t0_ln)
        self.timing["Layer"]["LayerNormCommunication"] += (comm1_ln["time"] - comm0_ln["time"])
        self.communication["Layer"]["LayerNormRounds"] += (comm1_ln["rounds"] - comm0_ln["rounds"])
        self.communication["Layer"]["LayerNormBytes"] += (comm1_ln["bytes"] - comm0_ln["bytes"])

        self.timing["Layer"]["Embedding"] += (t1_emb - t0_emb)
        self.timing["Layer"]["EmbeddingCommunication"] += (comm1_emb["time"] - comm0_emb["time"])
        self.communication["Layer"]["EmbeddingRounds"] += (comm1_emb["rounds"] - comm0_emb["rounds"])
        self.communication["Layer"]["EmbeddingBytes"] += (comm1_emb["bytes"] - comm0_emb["bytes"])
        return embeddings


class BertPreTrainedModel(cnn.Module):
    config_class = None
    load_tf_weights = None
    base_model_prefix = ""
    authorized_missing_keys = None
    authorized_unexpected_keys = None

    def __init__(self, config, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        self.config = config

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path, 
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
        communication = kwargs.pop("communication", None)
        if timing is None:
            print(f"No timing provided, using default float dictionary:\n" )
            timing = defaultdict(float)
        
        if communication is None:
            print(f"No communication provided, using default float dictionary:\n" )
            communication = defaultdict(float)
       

        torch_model = AutoModel.from_pretrained(pretrained_model_name_or_path)

        config = torch_model.config
        for key, value in model_config.items():
            setattr(config, key, value)
        
        # Load model
        model = cls(config, timing, communication, *model_args, **kwargs)
        model.load_state_dict(torch_model.state_dict(), strict=False)
        return model
    


class BertModel(BertPreTrainedModel):
    def __init__(self, config, timing, communication):
        super(BertModel, self).__init__(config)
        self.embeddings = PlainTextBertEmbeddings(config)
        self.encoder = BertLikeEncoder(config, timing, communication)
        self.pooler = BertLikePooler(config, timing, communication)

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

        self.timing = timing
        self.communication = communication
        self.apply_LoRA = config.apply_LoRA
        self.embeddings_cap = config.embeddings_cap


    def load_state_dict(self, state_dict, strict=False):
        super().load_state_dict(state_dict, strict)

    def encrypt(self, mode=True, src=0):
        super().encrypt(mode, src)
        #if self.plaintext_embedding:
        #    self.embeddings.encrypt(mode=False, src = src)

    def train(self, mode=True):
        super().train(mode)
        self.embeddings.train(mode=False)
    
    def forward(
        self,
        input_ids: crypten.CrypTensor,
        token_type_ids: Optional[crypten.CrypTensor] = None,
        attention_mask: Optional[crypten.CrypTensor] = None,
        device: str = "cpu"
    ):
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        embedding_output = self.embeddings(input_ids, token_type_ids)
        t1_emb = time.time()
        comm1_emb = comm.get().get_communication_stats()
        self.timing["Block"]["Embedding"] += (t1_emb - t0)
        self.timing["Block"]["EmbeddingCommunication"] += (comm1_emb["time"] - comm0["time"])
        self.communication["Block"]["EmbeddingRounds"] += (comm1_emb["rounds"] - comm0["rounds"])
        self.communication["Block"]["EmbeddingBytes"] += (comm1_emb["bytes"] - comm0["bytes"])


        if not isinstance(embedding_output, crypten.CrypTensor):
            if self.embeddings_cap is not None:
                embedding_output = logits_soft_capping(embedding_output, max_cap=self.embeddings_cap)
            embedding_output = crypten.cryptensor(embedding_output).to(device)

        encoded_layers = self.encoder(embedding_output, attention_mask)
        #encoded_layers_clear = encoded_layers.get_plain_text()
        sequence_output = encoded_layers
        pooled_output = self.pooler(sequence_output)
        #pooled_output_clear = pooled_output.get_plain_text()
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["BertTime"] += (t1-t0)
        self.timing["BertCommTime"] += (comm1["time"] - comm0["time"])
        self.communication["BertRounds"] +=  (comm1["rounds"] - comm0["rounds"])
        self.communication["BertBytes"] +=  (comm1["bytes"] - comm0["bytes"])
        return pooled_output
        #return sequence_output, pooled_output
    

class BertForSequenceClassification(BertModel):
    def __init__(self, config, timing, communication, *model_args, **kwargs):
        super().__init__(config, timing, communication)
        self.num_labels = kwargs.pop("num_labels", 2)
        self.fit_size = kwargs.pop("fit_size", 768)

        self.bert = BertModel(config, timing, communication)
        self.classifier = cnn.Linear(config.hidden_size, self.num_labels)

        self.logits_cap = config.logits_cap_classifier

    def load_state_dict(self, state_dict, strict=False):
        # the target state dict should not have classifier
        if state_dict.keys().__contains__("bert.embeddings.word_embeddings.weight"):
            prepend_string = "bert.embeddings."
            embed_dict = self.bert.embeddings.state_dict()
            for k in embed_dict.keys():
                embed_dict[k] = state_dict[prepend_string + k]

            self.bert.embeddings.load_state_dict(embed_dict)

        state_dict_keys = state_dict.keys()
        target_state_dict = self.state_dict().copy()
        for k in state_dict_keys:
            target_state_dict[k] = state_dict[k]

        super().load_state_dict(target_state_dict, strict)


    def encrypt(self, mode=True, src=0):
        super().encrypt(mode, src)
        self.bert.encrypt(mode, src)
        self.classifier.encrypt(mode, src)

    def forward(
        self,
        input_ids: crypten.CrypTensor,
        token_type_ids: Optional[crypten.CrypTensor] = None,
        attention_mask: Optional[crypten.CrypTensor] = None,
        device: str = "cpu"
    ):
        pooled_output = super().forward(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask,
            device=device
            )
        # dropout not used in original implementation
        #pooled_output = self.dropout(pooled_output)
        # HERE Add layer norm to avoid overflow in the loss
        #pooled_output = self.classifier_layer_norm(pooled_output)
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        #logits = self.classifier_act_fn(pooled_output)
        #t1_act = time.time()
        #comm1_act = comm.get().get_communication_stats()

        t0_lin = time.time()
        comm0_lin = comm.get().get_communication_stats()
        logits = self.classifier(pooled_output)
        t1_lin = time.time()
        comm1_lin = comm.get().get_communication_stats()

        if self.logits_capping is not None:
            logits = logits_soft_capping(logits, max_cap=self.logits_capping)

        self.timing["Layer"]["Linear"] += (t1_lin - t0_lin)
        self.timing["Layer"]["LinearComm"] += (comm1_lin["time"] - comm0_lin["time"])
        self.communication["Layer"]["LinearRounds"] +=  (comm1_lin["rounds"] - comm0_lin["rounds"])
        self.communication["Layer"]["LinearBytes"] +=  (comm1_lin["bytes"] - comm0_lin["bytes"])

        return logits

    def get_embeddings(
        self,
        input_ids: crypten.CrypTensor,
    ):
        return self.embeddings(input_ids)


MODELS = [ 
    BertModel,
    BertForSequenceClassification
]