# Copyright (c) 2025 SAP SE or an SAP affiliate company and sprint contributors
# SPDX-License-Identifier: Apache-2.0

import math
from collections import defaultdict
from time import time
from typing import Optional

import torch

import crypten
import crypten.nn as cnn
import crypten.communicator as comm

from modeling.activations.activations_crypten import (
    softmax_2RELU, 
    activation_quad, 
    softmax_2QUAD, 
    hard_GELU, 
    GELU, 
    erf_GELU, 
    Softmax_NN, 
    ReLU_optim, 
    bolt_GELU, 
    SoftmaxMaxCap
)

def logits_soft_capping(logits, max_cap=1):
    # logits soft capping for numerical stability (interval -10,10)
    # according to https://huggingface.co/blog/gemma2#soft-capping-and-attention-implementations
    return (logits / max_cap).tanh() * max_cap

ACT2FN = {
    "relu": cnn.ReLU(),
    "relu6": cnn.ReLU6(),
    "gelu": GELU(),
    "quad": activation_quad(),
    "hard_gelu": hard_GELU(),
    "erf_gelu": erf_GELU(),
    "relu_optim": ReLU_optim(),
    "bolt_gelu": bolt_GELU(),
}

SOFTMAX2FN = {
    "softmax": cnn.Softmax(dim=-1),
    "softmax_2RELU": softmax_2RELU(dim=-1),
    "softmax_2QUAD": softmax_2QUAD(dim=-1),
    "softmax_nn" : Softmax_NN(input_size=128, hidden_size=128),
    "softmax_max": SoftmaxMaxCap(dim=-1)
}


class LayerNorm(cnn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)

        # Do not use cryptensor here, otherwise you will get an error in encrypt
        pytorch_module = torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
        for param in ["weight", "bias"]:
            self.register_parameter(param, getattr(pytorch_module, param))
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.threshold = 1e2
        self.scale = 1e3


    def forward(self, input):
        debug = False
        #if (input.get_plain_text() > 1e2).any() or (input.get_plain_text() < -1e2).any():
        #    print("Debug LayerNorm input is out of bounds")
        #    print(input.get_plain_text().abs().max())
        #    debug = True

        end_index = input.dim()
        start_index = end_index - len(self.normalized_shape)
        stats_dimensions = list(range(start_index, end_index))
        

        mean = input.mean(dim = stats_dimensions, keepdim=True) 
        # save computation of the mean again
        # also the backpropagation of the variance is not 
        # efficient since they compute the mean but they do not use it
        # for the forward pass
        variance = (input - mean).pow(2).mean(dim = stats_dimensions, keepdim=True)

        #if debug:
        #    if (variance.get_plain_text() > 1e2).any() or (variance.get_plain_text() < -1e2).any():
        #        print("Debug LayerNorm variance is out of bounds")
        #        print("Variance abs max:, "variance.get_plain_text().abs().max())

        scaling_cond = variance.le(self.threshold)
        variance = variance + self.eps
        scaled_var = variance/self.scale


        inv_sqrt_arg = scaled_var + (variance - scaled_var)*(scaling_cond)
	
        inv_sqrt_out = inv_sqrt_arg.inv_sqrt()
        inv_var = inv_sqrt_out + (inv_sqrt_out*(1/(math.sqrt(self.scale))) - inv_sqrt_out)*(1-scaling_cond)



        #scale_factor = 10
        #inv_var = (1/math.sqrt(scale_factor)) * ( (variance + self.eps)/scale_factor).inv_sqrt()
        # HERE: saving pre computations for backward
        
        x_norm = (input - mean) * inv_var
        if debug:
            if (x_norm.get_plain_text() > 1e2).any() or (x_norm.get_plain_text() < -1e2).any():
                print("Debug LayerNorm x_norm is out of bounds")
                print("Normalized input max: ",x_norm.get_plain_text().abs().max())
                print("Variance abs max: ", variance.get_plain_text().abs().max())
                print("Input abs max: ",input.get_plain_text().abs().max())
        #    print("Debug LayerNorm")
        return x_norm * self.weight + self.bias


class CryptenBertLikeConfig:
    def __init__(
            self,
            layer_norm_eps = 1e-5,
            softmax_act = "softmax",
            hidden_act = "relu",
            classifier_act = "relu", 
            plaintext_embedding = False,
            apply_LoRA = False,
            logits_cap_attention = 10.0,
            logits_cap_classifier = 10.0,
            embeddings_cap = 10.0,
    ):
        self.layer_norm_eps = layer_norm_eps
        self.softmax_act = softmax_act
        self.hidden_act = hidden_act
        self.classifier_act = classifier_act
        self.plaintext_embedding = plaintext_embedding
        self.apply_LoRA = apply_LoRA
        self.logits_cap_attention = logits_cap_attention
        self.logits_cap_classifier = logits_cap_classifier
        self.embeddings_cap = embeddings_cap

    def items(self):
        return self.__dict__.items()
    
    def to_dict(self):
        return self.__dict__
    
    def from_dict(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
        return self


class BertLikeSelfAttention(cnn.Module):
    def __init__(self, config, timing, communication):
        super(BertLikeSelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.timing = timing
        self.communication = communication

        self.query = cnn.Linear(config.hidden_size, self.all_head_size)
        self.key = cnn.Linear(config.hidden_size, self.all_head_size)
        self.value = cnn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = cnn.Dropout(config.attention_probs_dropout_prob)

        self.softmax = SOFTMAX2FN[config.softmax_act]
        self.softmax_str = config.softmax_act

        self.logits_cap = config.logits_cap_attention if hasattr(config, "logits_cap_attention") else None

        if self.logits_cap is None and self.softmax_str == "softmax_max":
            raise ValueError("SoftmaxMaxCap requires a maximum value for the logits")

    
    def forward(
        self,
        hidden_states : torch.Tensor,
        attention_mask : Optional[torch.Tensor] = None
    ): 
        t0_sa = time()
        comm0_sa = comm.get().get_communication_stats()
        bs, seq_len, dim = hidden_states.shape
        dim_per_head = dim // self.num_attention_heads
        
        mask_reshape = (bs, 1, 1, seq_len)

        def transpose_for_scores(x):
            return x.reshape(bs, -1, self.num_attention_heads, dim_per_head).transpose(1, 2)
        
        def un_transpose_for_scores(x):
            return x.transpose(1, 2).reshape(bs, -1, self.all_head_size)
        
        def masked_fill(scores, mask):
            #mod_mask = (mask == 0).view(mask_reshape).expand(scores.size())
            #mod_mask = mod_mask * value
            # HERE for tanh, max value is around 250, so we can use as min
            # -200 * self.logits_cap to be safe
            if self.logits_cap is not None:
                value = -2e2 * self.logits_cap
            else:
                value = -2e2

            mod_mask = mask.view(mask_reshape).expand(scores.size())
            mod_mask = (1 - mod_mask) * value
            return scores + mod_mask
        
        t0_lin = time()
        comm0_lin = comm.get().get_communication_stats()
        query_layer = transpose_for_scores(self.query(hidden_states)) # (bs, num_heads, seq_length, head_dim)
        key_layer = transpose_for_scores(self.key(hidden_states)) # (bs, num_heads, seq_length, head_dim)
        value_layer = transpose_for_scores(self.value(hidden_states)) # (bs, num_heads, seq_length, head_dim)
        t1_lin = time()
        comm1_lin = comm.get().get_communication_stats()

        query_layer = query_layer / math.sqrt(self.attention_head_size)
        attention_scores = query_layer.matmul(key_layer.transpose(-1, -2)) # (bs, num_heads, seq_length, seq_length)
        t1_lin1 = time()
        comm1_lin1 = comm.get().get_communication_stats()



        if attention_mask is not None:
            attention_scores = masked_fill(attention_scores, attention_mask)
            #attention_scores = attention_scores + attention_mask.view(mask_reshape).expand(attention_scores.size())

        if self.logits_cap is not None:
            t0_capping = time()
            comm0_capping = comm.get().get_communication_stats()
            attention_scores = logits_soft_capping(attention_scores, max_cap=self.logits_cap)
            t1_capping = time()
            comm1_capping = comm.get().get_communication_stats()

        comm0_smax = comm.get().get_communication_stats()
        t0_smax = time()
        if self.softmax_str == "softmax_max":   
            attention_probs = self.softmax(attention_scores, self.logits_cap) # (bs, num_heads, seq_length, seq_length)
        else:
            attention_probs = self.softmax(attention_scores) # (bs, num_heads, seq_length, seq_length)

        t1_smax = time()
        comm1_smax = comm.get().get_communication_stats()
        
        attention_probs = self.dropout(attention_probs)

        comm0_smax_v = comm.get().get_communication_stats()
        t0_smax_v = time()
        context_layer = attention_probs.matmul(value_layer) # (bs, num_heads, seq_length, head_dim)
        t1_smax_v = time()
        comm1_smax_v = comm.get().get_communication_stats()

        context_layer = un_transpose_for_scores(context_layer) # (bs, seq_length, hidden_size)
        t1_sa = time()
        comm1_sa = comm.get().get_communication_stats()

        self.timing["Layer"]["Linear"] += (t1_lin - t0_lin)
        self.timing["Layer"]["LinearCommunication"] += (comm1_lin["time"] - comm0_lin["time"])
        self.communication["Layer"]["LinearRounds"] += (comm1_lin["rounds"] - comm0_lin["rounds"])
        self.communication["Layer"]["LinearBytes"] += (comm1_lin["bytes"] - comm0_lin["bytes"])


        if self.logits_cap is not None:
            self.timing["Layer"]["AttentionSoftCapping"] += (t1_capping - t0_capping)
            self.timing["Layer"]["AttentionSoftCappingComm"] += (comm1_capping["time"] - comm0_capping["time"])
            self.communication["Layer"]["AttentionSoftCappingRounds"] += (comm1_capping["rounds"] - comm0_capping["rounds"])
            self.communication["Layer"]["AttentionSoftCappingBytes"] += (comm1_capping["bytes"] - comm0_capping["bytes"])

            self.timing["Layer"]["Softcapping"] += (t1_capping - t0_capping)
            self.timing["Layer"]["SoftcappingCommunication"] += (comm1_capping["time"] - comm0_capping["time"])
            self.communication["Layer"]["SoftcappingRounds"] += (comm1_capping["rounds"] - comm0_capping["rounds"])
            self.communication["Layer"]["SoftcappingBytes"] += (comm1_capping["bytes"] - comm0_capping["bytes"])

        self.timing["Layer"]["Softmax"] += (t1_smax - t0_smax)
        self.timing["Layer"]["SoftmaxCommunication"] += (comm1_smax["time"] - comm0_smax["time"])
        self.communication["Layer"]["SoftmaxRounds"] += (comm1_smax["rounds"] - comm0_smax["rounds"])
        self.communication["Layer"]["SoftmaxBytes"] += (comm1_smax["bytes"] - comm0_smax["bytes"])


        self.timing["Block"]["SelfAttention"] += (t1_sa - t0_sa)
        self.timing["Block"]["SelfAttentionComm"] += (comm1_sa["time"] - comm0_sa["time"])
        self.communication["Block"]["SelfAttentionRounds"] += (comm1_sa["rounds"] - comm0_sa["rounds"])
        self.communication["Block"]["SelfAttentionBytes"] += (comm1_sa["bytes"] - comm0_sa["bytes"])

        self.communication["Encoder"]["Lin1Rounds"] += (comm1_lin1["rounds"] - comm0_lin["rounds"])
        self.communication["Encoder"]["Lin1Bytes"] += (comm1_lin1["bytes"] - comm0_lin["bytes"])
        # Add timing
        self.timing["Encoder"]["Lin1"] += t1_lin1 - t0_lin

        self.communication["Encoder"]["SoftmaxValue"] += (comm1_smax_v["rounds"] - comm0_smax_v["rounds"])
        self.communication["Encoder"]["SoftmaxBytes"] += (comm1_smax_v["bytes"] - comm0_smax_v["bytes"])
        self.timing["Encoder"]["SoftmaxValue"] += t1_smax_v - t0_smax_v

        return context_layer
    

class BertLikeSelfOutput(cnn.Module):
    def __init__(self, config, timing, communication):
        super(BertLikeSelfOutput, self).__init__()
        self.dense = cnn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = cnn.Dropout(config.hidden_dropout_prob)
        #self.apply_LoRA = config.apply_LoRA

        self.timing = timing
        self.communication = communication


    def forward(self, hidden_states, input_tensor):
        comm0_so = comm.get().get_communication_stats()
        t0_so = time()
        hidden_states = self.dense(hidden_states)
        t1_lin = time()
        comm1_lin = comm.get().get_communication_stats()
        
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor

        comm0_ln = comm.get().get_communication_stats()
        t0_ln = time()
        hidden_states = self.LayerNorm(hidden_states)
        t1 = time()
        comm1 = comm.get().get_communication_stats()

        self.timing["Layer"]["Linear"] += (t1_lin - t0_so)
        self.timing["Layer"]["LinearComm"] += (comm1_lin["time"] - comm0_so["time"])
        self.communication["Layer"]["LinearRounds"] += (comm1_lin["rounds"] - comm0_so["rounds"])
        self.communication["Layer"]["LinearBytes"] += (comm1_lin["bytes"] - comm0_so["bytes"])

        self.timing["Layer"]["LayerNorm"] += (t1 - t0_ln)
        self.timing["Layer"]["LayerNormComm"] += (comm1["time"] - comm0_ln["time"])
        self.communication["Layer"]["LayerNormRounds"] += (comm1["rounds"] - comm0_ln["rounds"])
        self.communication["Layer"]["LayerNormBytes"] += (comm1["bytes"] - comm0_ln["bytes"])

        self.timing["Block"]["SelfOutput"] += (t1 - t0_so)
        self.timing["Block"]["SelfOutputComm"] += (comm1["time"] - comm0_so["time"])
        self.communication["Block"]["SelfOutputRounds"] += (comm1["rounds"] - comm0_so["rounds"])
        self.communication["Block"]["SelfOutputBytes"] += (comm1["bytes"] - comm0_so["bytes"])


        self.communication["Encoder"]["Lin2Rounds"] += (comm1_lin["rounds"] - comm0_so["rounds"])
        self.communication["Encoder"]["Lin2Bytes"] += (comm1_lin["bytes"] - comm0_so["bytes"])
        self.timing["Encoder"]["Lin2"] += t1_lin - t0_so

        return hidden_states


class BertLikeAttention(cnn.Module):
    def __init__(self, config, timing, communication):
        super(BertLikeAttention, self).__init__()
        self.self = BertLikeSelfAttention(config, timing, communication)
        self.output = BertLikeSelfOutput(config, timing, communication)
        self.timing = timing
        self.communication = communication
    
    def forward(
        self,
        hidden_states : torch.Tensor,
        attention_mask : Optional[torch.Tensor] = None
    ):
        comm0_att = comm.get().get_communication_stats()
        t0 = time()
        self_output = self.self(
            hidden_states = hidden_states, 
            attention_mask = attention_mask
        )

        attention_output = self.output(
            hidden_states = self_output, 
            input_tensor = hidden_states
        )
        t1 = time()
        comm1_att = comm.get().get_communication_stats()

        self.timing["Block"]["Attention"] += (t1 - t0)
        self.timing["Block"]["AttentionComm"] += (comm1_att["time"] - comm0_att["time"])
        self.communication["Block"]["AttentionRounds"] += (comm1_att["rounds"] - comm0_att["rounds"])
        self.communication["Block"]["AttentionBytes"] += (comm1_att["bytes"] - comm0_att["bytes"])

        return attention_output
    

class BertLikeIntermediate(cnn.Module):
    def __init__(self, config, timing, communication):
        super(BertLikeIntermediate, self).__init__()
        self.dense = cnn.Linear(config.hidden_size, config.intermediate_size)
        self.hidden_act = config.hidden_act
        self.intermediate_act_fn = ACT2FN[config.hidden_act]
        self.timing = timing
        self.communication = communication
    
    def forward(self, hidden_states):
        comm0 = comm.get().get_communication_stats()
        t0 = time()
        hidden_states = self.dense(hidden_states) # (bs, seq_len, dim)
        t1_lin = time()
        comm1_lin = comm.get().get_communication_stats()

        comm0_act = comm.get().get_communication_stats()
        t0_act = time()
        hidden_states = self.intermediate_act_fn(hidden_states) # (bs, q, dim)
        t1_act = time()
        comm1_act = comm.get().get_communication_stats()

        t1 = time()
        comm1 = comm.get().get_communication_stats()

        self.timing["Layer"]["Linear"] += (t1_lin - t0)
        self.timing["Layer"]["LinearComm"] += (comm1_lin["time"] - comm0["time"])
        self.communication["Layer"]["LinearRounds"] += (comm1_lin["rounds"] - comm0["rounds"])
        self.communication["Layer"]["LinearBytes"] += (comm1_lin["bytes"] - comm0["bytes"])

        self.timing["Layer"][f"{self.hidden_act}"] += (t1_act - t0_act)
        self.timing["Layer"][f"{self.hidden_act}Comm"] += (comm1_act["time"] - comm0_act["time"])
        self.communication["Layer"][f"{self.hidden_act}Rounds"] += (comm1_act["rounds"] - comm0_act["rounds"])
        self.communication["Layer"][f"{self.hidden_act}Bytes"] += (comm1_act["bytes"] - comm0_act["bytes"])

        self.timing["Block"]["Intermediate"] += (t1 - t0)
        self.timing["Block"]["Intermediate"] += (comm1["time"] - comm0["time"])
        self.communication["Block"]["IntermediateRounds"] += (comm1["rounds"] - comm0["rounds"])
        self.communication["Block"]["IntermediateBytes"] += (comm1["bytes"] - comm0["bytes"])
        
        self.communication["Encoder"]["Lin3Rounds"] += (comm1_lin["rounds"] - comm0["rounds"])
        self.communication["Encoder"]["Lin3Bytes"] += (comm1_lin["bytes"] - comm0["bytes"])
        self.timing["Encoder"]["Lin3"] += t1_lin - t0

        self.communication["Encoder"][f"{self.hidden_act}Rounds"] += (comm1_act["rounds"] - comm0_act["rounds"])
        self.communication["Encoder"][f"{self.hidden_act}Bytes"] += (comm1_act["bytes"] - comm0_act["bytes"])
        self.timing["Encoder"][f"{self.hidden_act}"] += t1_act - t0_act
        return hidden_states
    

class BertLikeOutput(cnn.Module):
    def __init__(self, config, timing, communication):
        super(BertLikeOutput, self).__init__()
        self.dense = cnn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = cnn.Dropout(config.hidden_dropout_prob)
        self.timing = timing
        self.communication = communication

    def forward(self, hidden_states, input_tensor):
        comm0 = comm.get().get_communication_stats()
        t0 = time()
        hidden_states = self.dense(hidden_states)
        t1_lin = time()
        comm1_lin = comm.get().get_communication_stats()

        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor
        hidden_mid = hidden_states.get_plain_text().detach()

        comm0_ln = comm.get().get_communication_stats()
        t0_ln = time()
        hidden_states = self.LayerNorm(hidden_states)
        #if ((hidden_states.get_plain_text()> 1e5).any() or (hidden_states.get_plain_text() < -1e5).any()):
        #    print("LayerNorm output is out of bounds")
            #print(hidden_mid)
            #print(hidden_states.get_plain_text())
            #print(hidden_states.get_plain_text() - hidden_mid)
        t1 = time()
        comm1 = comm.get().get_communication_stats()
        
        self.timing["Layer"]["Linear"] += (t1_lin - t0)
        self.timing["Layer"]["LinearComm"] += (comm1_lin["time"] - comm0["time"])
        self.communication["Layer"]["LinearRounds"] += (comm1_lin["rounds"] - comm0["rounds"])
        self.communication["Layer"]["LinearBytes"] += (comm1_lin["bytes"] - comm0["bytes"])

        self.timing["Layer"]["LayerNorm"] += (t1 - t0_ln)
        self.timing["Layer"]["LayerNormComm"] += (comm1["time"] - comm0_ln["time"])
        self.communication["Layer"]["LayerNormRounds"] += (comm1["rounds"] - comm0_ln["rounds"])
        self.communication["Layer"]["LayerNormBytes"] += (comm1["bytes"] - comm0_ln["bytes"])

        self.timing["Block"]["LinearTime"] += (t1 - t0)
        self.timing["Block"]["LinearCommTime"] += (comm1["time"] - comm0["time"])
        self.communication["Block"]["LinearRounds"] += (comm1["rounds"] - comm0["rounds"])
        self.communication["Block"]["LinearBytes"] += (comm1["bytes"] - comm0["bytes"])


        self.communication["Encoder"]["Lin4Rounds"] += (comm1_lin["rounds"] - comm0["rounds"])
        self.communication["Encoder"]["Lin4Bytes"] += (comm1_lin["bytes"] - comm0["bytes"])
        self.timing["Encoder"]["Lin4"] += t1_lin - t0
        return hidden_states
    

class BertLikeLayer(cnn.Module):
    def __init__(self, config, timing, communication):
        super(BertLikeLayer, self).__init__()
        self.attention = BertLikeAttention(config, timing, communication)
        self.intermediate = BertLikeIntermediate(config, timing, communication)
        self.output = BertLikeOutput(config, timing, communication)
        self.timing = timing
        self.communication = communication
    
    def forward(
        self,
        hidden_states : torch.Tensor,
        attention_mask : Optional[torch.Tensor] = None
    ):
        attention_output = self.attention(
            hidden_states = hidden_states, 
            attention_mask = attention_mask
        )

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(
            hidden_states = intermediate_output, 
            input_tensor = attention_output
        )

        return layer_output
    

class BertLikeEncoder(cnn.Module):
    def __init__(self, config, timing, communication):
        super(BertLikeEncoder, self).__init__()
        self.layer = cnn.ModuleList([BertLikeLayer(config, timing, communication) 
                                    for _ in range(config.num_hidden_layers)])
        self.timing = timing
        self.communication = communication

        if not communication.keys().__contains__("Encoder"):
            communication["Encoder"] = defaultdict(float)
        if not timing.keys().__contains__("Encoder"):
            timing["Encoder"] = defaultdict(float)
    
    def forward(
        self,
        hidden_states : torch.Tensor,
        attention_mask : Optional[torch.Tensor] = None
    ):
        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states = hidden_states, 
                attention_mask = attention_mask
            )
        
        return hidden_states
    

class BertLikePooler(cnn.Module):
    def __init__(self, config, timing, communication):
        super(BertLikePooler, self).__init__()
        self.dense = cnn.Linear(config.hidden_size, config.hidden_size)
        self.timing = timing
        self.communication = communication

    def forward(self, hidden_states):
        comm0 = comm.get().get_communication_stats()
        t0 = time()
        # first_token_tensor = hidden_states[:, 0]
        # Replace indexing with narrow to avoid errors in CrypTen
        first_token_tensor = hidden_states.narrow(1,0,1).squeeze(1) # (bs, dim)

        comm0_lin = comm.get().get_communication_stats()
        t0_lin = time()
        pooled_output = self.dense(first_token_tensor) # (bs, dim)
        t1_lin = time()
        comm1_lin = comm.get().get_communication_stats()

        comm0_tanh = comm.get().get_communication_stats()
        t0_tanh = time()
        pooled_output = pooled_output.tanh() #self.activation(pooled_output) # (bs, dim)        
        t1 = time()
        comm1 = comm.get().get_communication_stats()


        self.timing["Layer"]["Linear"] += (t1_lin - t0_lin)
        self.timing["Layer"]["LinearComm"] += (comm1_lin["time"] - comm0_lin["time"])
        self.communication["Layer"]["LinearRounds"] += (comm1_lin["rounds"] - comm0_lin["rounds"])
        self.communication["Layer"]["LinearBytes"] += (comm1_lin["bytes"] - comm0_lin["bytes"])

        self.timing["Layer"]["Tanh"] += (t1 - t0_tanh)
        self.timing["Layer"]["TanhComm"] += (comm1["time"] - comm0_tanh["time"])
        self.communication["Layer"]["TanhRounds"] += (comm1["rounds"] - comm0_tanh["rounds"])
        self.communication["Layer"]["TanhBytes"] += (comm1["bytes"] - comm0_tanh["bytes"])


        self.timing["Block"]["Pooler"] += (t1 - t0)
        self.timing["Block"]["PoolerComm"] += (comm1["time"] - comm0["time"])
        self.communication["Block"]["PoolerRounds"] += (comm1["rounds"] - comm0["rounds"])
        self.communication["Block"]["PoolerBytes"] += (comm1["bytes"] - comm0["bytes"])

        return pooled_output

    