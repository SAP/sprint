# Copyright (c) 2025 SAP SE or an SAP affiliate company and sprint contributors
# SPDX-License-Identifier: Apache-2.0

import math
import torch
from time import time
import torch.nn as nn
from typing import Optional

from transformers.activations import GELUActivation

from modeling.activations.activations_clear import (
    Softmax_NN,
    hardGELU,
    boltGELU,
    SoftmaxMax,
    scaled_attention,
    relurs_attention,
    softmax_2RELU,
    activation_quad,
    softmax_2QUAD,
)





def logits_soft_capping(logits, max_cap=1):
    # logits soft capping for numerical stability
    # according to https://huggingface.co/blog/gemma2#soft-capping-and-attention-implementations
    return (logits / max_cap).tanh() * max_cap 

ACT2FN = {
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
    "gelu": GELUActivation(), # Already approximated with tanh
    "quad": activation_quad(),
    "hard_gelu": hardGELU(),
    "erf_gelu": nn.GELU(),
    "bolt_gelu": boltGELU(),
}

SOFTMAX2FN = {
    "softmax": nn.Softmax(dim=-1),
    "softmax_2RELU": softmax_2RELU(dim=-1),
    "softmax_2QUAD": softmax_2QUAD(dim=-1),
    "softmax_nn" : Softmax_NN(input_size=128, hidden_size=128),
    "softmax_max": SoftmaxMax(dim=-1),
    "scaled_attention": scaled_attention,
    "relurs": relurs_attention
} 


class BertLikeSelfAttention(nn.Module):
    def __init__(self, config, timing):
        super(BertLikeSelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.timing = timing

        self.query = nn.Linear(config.hidden_size, self.all_head_size) # for roberta-base 768 -> 768 (12 heads)
        self.key = nn.Linear(config.hidden_size, self.all_head_size) # for roberta-base 768 -> 768 (12 heads)
        self.value = nn.Linear(config.hidden_size, self.all_head_size) # for roberta-base 768 -> 768 (12 heads)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.softmax_str = config.softmax_act
        self.softmax = SOFTMAX2FN[config.softmax_act]

        self.logits_cap = config.logits_cap_attention if hasattr(config, "logits_cap_attention") else None

        self.half_precision = False
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states : torch.Tensor,
        attention_mask : Optional[torch.Tensor] = None
    ):  
        
        
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states)) # (bs, num_heads, seq_len, head_dim)
        value_layer = self.transpose_for_scores(self.value(hidden_states)) # (bs, num_heads, seq_len, head_dim)
        query_layer = self.transpose_for_scores(mixed_query_layer) # (bs, num_heads, seq_len, head_dim)


        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size) if (self.softmax != scaled_attention and self.softmax!= relurs_attention) else attention_scores

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertLikeModel forward() function)
            attention_scores = attention_scores + attention_mask

        attention_scores = logits_soft_capping(attention_scores, self.logits_cap) if self.logits_cap is not None else attention_scores
        
        # Normalize the attention scores to probabilities.
        if self.softmax_str == "softmax_max" and attention_mask is not None:
            attention_probs = self.softmax(attention_scores, max_cap=self.logits_cap)
        else:
            attention_probs = self.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer
    
    def half(self):
        self.half_precision = True
        return super().half()


class BertLikeSelfOutput(nn.Module):
    def __init__(self, config, timing):
        super(BertLikeSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.timing = timing
    
    def forward(self, hidden_states, input_tensor):
        t0_so = time()
        hidden_states = self.dense(hidden_states)
        t1_lin = time()
        
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor

        t0_ln = time()
        hidden_states = self.LayerNorm(hidden_states)
        t1 = time()

        self.timing["Layer"]["Linear"] += (t1_lin - t0_so)
        self.timing["Layer"]["LayerNorm"] += (t1 - t0_ln)
        self.timing["Block"]["SelfOutput"] += (t1 - t0_so)
        self.timing["Encoder"]["Lin2"] += t1_lin - t0_so

        return hidden_states
    
    def half(self):
        self.dense.half()
        self.LayerNorm.half()
    
class BertLikeAttention(nn.Module):
    def __init__(self, config, timing):
        super(BertLikeAttention, self).__init__()
        self.self = BertLikeSelfAttention(config, timing)
        self.output = BertLikeSelfOutput(config, timing)
        self.timing = timing
    
    def forward(
        self,
        hidden_states : torch.Tensor,
        attention_mask : Optional[torch.Tensor] = None
    ):
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

        self.timing["Block"]["Attention"] += (t1 - t0)

        return attention_output
    
    def half(self):
        self.self.half()
        self.output.half()
    

class BertLikeIntermediate(nn.Module):
    def __init__(self, config, timing):
        super(BertLikeIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.hidden_act = config.hidden_act
        self.intermediate_act_fn = ACT2FN[config.hidden_act]
        self.timing = timing
    
    def forward(self, hidden_states):
        t0 = time()
        hidden_states = self.dense(hidden_states) # (bs, seq_len, dim)
        t1_lin = time()

        #if torch.isnan(hidden_states).any():
        #    print("NAN in BertLikeIntermediate")
        
        t0_act = time()
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float32)
            hidden_states = self.intermediate_act_fn(hidden_states) # (bs, q, dim)
            hidden_states = hidden_states.to(torch.float16)
        else:
            hidden_states = self.intermediate_act_fn(hidden_states)
        t1_act = time()

        #if torch.isnan(hidden_states).any():
        #    print("NAN in BertLikeIntermediate")

        t1 = time()
        self.timing["Layer"]["Linear"] += (t1_lin - t0)
        self.timing["Layer"][f"{self.hidden_act}"] += (t1_act - t0_act)
        self.timing["Block"]["Intermediate"] += (t1 - t0)
        self.timing["Encoder"]["Lin3"] += t1_lin - t0

        self.timing["Encoder"][f"{self.hidden_act}"] += t1_act - t0_act
        return hidden_states
    
    def half(self):
        super().half()
    

class BertLikeOutput(nn.Module):
    def __init__(self, config, timing):
        super(BertLikeOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.timing = timing

    def forward(self, hidden_states, input_tensor):
        t0 = time()
        hidden_states = self.dense(hidden_states)
        t1_lin = time()

        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        t0_ln = time()
        hidden_states = self.LayerNorm(hidden_states)

        t1 = time()
        self.timing["Layer"]["Linear"] += (t1_lin - t0)
        self.timing["Layer"]["LayerNorm"] += (t1 - t0_ln)
        self.timing["Block"]["LinearTime"] += (t1 - t0)
        self.timing["Encoder"]["Lin4"] += t1_lin - t0
        return hidden_states
    
    def half(self):
        self.dense.half()
        self.LayerNorm.half()
    

class BertLikeLayer(nn.Module):
    def __init__(self, config, timing):
        super(BertLikeLayer, self).__init__()
        self.attention = BertLikeAttention(config, timing)
        self.intermediate = BertLikeIntermediate(config, timing)
        self.output = BertLikeOutput(config, timing)
        self.timing = timing
    
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

        #DEBUG
        #if torch.isnan(layer_output).any() or torch.isnan(intermediate_output).any() or torch.isnan(attention_output).any():
        #    print("NAN in BertLikeLayer")

        return layer_output
    
    def half(self):
        self.attention.half()
        self.intermediate.half()
        self.output.half()
    

class BertLikeEncoder(nn.Module):
    def __init__(self, config, timing):
        super(BertLikeEncoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLikeLayer(config, timing) 
                                    for _ in range(config.num_hidden_layers)])
        self.timing = timing
    
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
    
    def half(self):
        for layer in self.layer:
            layer.half()
    

class BertLikePooler(nn.Module):
    def __init__(self, config, timing):
        super(BertLikePooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.timing = timing

    def forward(self, hidden_states):
        t0 = time()
        # first_token_tensor = hidden_states[:, 0]
        # Replace indexing with narrow to avoid errors in CrypTen
        first_token_tensor = hidden_states.narrow(1,0,1).squeeze(1) # (bs, dim)
    
        t0_lin = time()
        pooled_output = self.dense(first_token_tensor) # (bs, dim)
        t1_lin = time()

        t0_tanh = time()
        pooled_output = pooled_output.tanh() #self.activation(pooled_output) # (bs, dim)        
        t1 = time()


        self.timing["Layer"]["Linear"] += (t1_lin - t0_lin)
        self.timing["Layer"]["Tanh"] += (t1 - t0_tanh)
        self.timing["Block"]["Pooler"] += (t1 - t0)

        return pooled_output
