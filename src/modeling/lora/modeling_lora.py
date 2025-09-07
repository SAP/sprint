"""
    Code adapted from: https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
"""

import crypten 
import crypten.nn as cnn

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class LoraConfig():
    def __init__(
        self,
        r: int = 8,
        lora_alpha: int = 8,
        target_modules: list = ["query", "value"],
        modules_to_save: list = ["classifier"],
        lora_dropout: float = 0.1,
        bias: str = "none",
        fan_in_fan_out: bool = False,
        freeze_A: bool = False,
        init_method: str = 'normal'
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules
        self.modules_to_save = modules_to_save
        self.lora_dropout = lora_dropout
        self.bias = bias
        self.fan_in_fan_out = fan_in_fan_out
        self.freeze_A = freeze_A
        self.init_method = init_method
    
    def set_freeze_A(self, freeze_A: bool):
        self.freeze_A = freeze_A

    def set_rank(self, r: int):
        self.r = r

    def set_alpha(self, lora_alpha: int):
        self.lora_alpha = lora_alpha

    def set_target_modules(self, target_modules: list):
        self.target_modules = target_modules

    def set_modules_to_save(self, modules_to_save: list):
        self.modules_to_save = modules_to_save

    def set_lora_dropout(self, lora_dropout: float):
        self.lora_dropout = lora_dropout

    def set_bias(self, bias: str):
        self.bias = bias

    def set_fan_in_fan_out(self, fan_in_fan_out: bool):
        self.fan_in_fan_out = fan_in_fan_out

    def set_init_method(self, init_method: str):
        self.init_method = init_method

    def get_config(self):
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "modules_to_save": self.modules_to_save,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "fan_in_fan_out": self.fan_in_fan_out,
            "freeze_A": self.freeze_A
        }
    
    def to_dict(self):
        return self.get_config()

    def from_dict(self, lora_dict):
        for key, value in lora_dict.items():
            setattr(self, key, value)
        return self


class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout layer
        if lora_dropout > 0.:
            self.lora_dropout = cnn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class LoRALinearCrypten(cnn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        fan_in_fan_out: bool = False,
        bias: str = 'none',
        state_dict: dict = None,
        freeze_A: bool = False,
        non_linearity: str = 'gelu',
        encrypted: bool = False,
        init_method: str = 'kaiming_uniform',
        kaiming_a: float = 1,
        linear_layer: nn.Module = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.merge_weights = merge_weights
        self.fan_in_fan_out = fan_in_fan_out
        self.update_bias = bias
        self.non_linearity = non_linearity
        self.freeze_A = freeze_A

        if lora_dropout > 0.:
            self.lora_dropout = cnn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        self.linear = linear_layer
        #self.linear = cnn.Linear(in_features, out_features, bias=bias)
#
        #if state_dict is not None:
        #    self.linear.weight.data = state_dict["weight"]
        #    self.linear.bias.data = state_dict["bias"]

        device = self.linear.weight.device
        if self.r > 0:
            device = self.linear.weight.device
            self.lora_A = cnn.Linear(in_features=in_features, out_features=r, bias=False).to(device)
            self.lora_B = cnn.Linear(in_features=r, out_features=out_features, bias=False).to(device)
            self.scaling = self.lora_alpha / self.r

            self.linear.weight.requires_grad = False

            if self.linear.bias is not None:
                if self.update_bias == 'none':
                    self.linear.bias.requires_grad = False
                else:
                    self.linear.bias.requires_grad = True
            if freeze_A:
                self.lora_A.weight.requires_grad = False

        self.init_method = init_method
        self.kaiming_a = kaiming_a
        self.reset_parameters()


        if encrypted:
            self.encrypt()

        self.merged = False

        #self.train()

    def reset_parameters(self):
        #self.reset_linear_parameters()
        # No need to reset the parameters of the linear layer from a pre-trained model
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            if self.init_method == 'normal':
                torch.nn.init.normal_(self.lora_A.weight)
            elif self.init_method == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=self.kaiming_a)
            elif self.init_method == 'kaiming_normal':
                torch.nn.init.kaiming_normal_(self.lora_A.weight, a=self.kaiming_a)
            elif self.init_method == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(self.lora_A.weight)
            elif self.init_method == 'xavier_normal':
                torch.nn.init.xavier_normal_(self.lora_A.weight)
            elif self.init_method == 'orthogonal':
                torch.nn.init.orthogonal_(self.lora_A.weight)
            else:
                raise NotImplementedError("Initialization method not implemented")

            
            #else:
                #torch.nn.init.kaiming_normal_(self.lora_A.weight, nonlinearity=self.non_linearity)
                #torch.nn.init.kaiming_normal_(self.lora_A.weight, nonlinearity=self.non_linearity)
                #torch.nn.init.kaiming_uniform_(self.lora_A, nonlinearity=self.non_linearity)

            self.lora_B.weight.data.zero_()
            #torch.nn.init.zeros_(self.lora_B)

    def encrypt(self, mode=True, src=0):
        super().encrypt(mode, src)
        if self.r > 0:
            if mode:
                if isinstance(self.linear.weight, crypten.CrypTensor):
                    self.linear.weight = self.linear.weight.get_plain_text()
                    if self.update_bias == 'none':
                        self.linear.bias = self.linear.bias.get_plain_text()
                    if self.freeze_A:
                        self.lora_A.weight = self.lora_A.weight.get_plain_text()

    def reset_linear_parameters(self):
        pass
#        self.linear.reset_parameters()

    def forward(self, input):
        result = self.linear(input)
        if self.r > 0 and not self.merged:
            after_dropout = self.lora_dropout(input)
            after_A = self.lora_A(after_dropout)
            after_B = self.lora_B(after_A)
            result = result + after_B * self.scaling
        return result
    
    def merge_weights_to_unload(self):
        if self.r > 0:
            self.linear.weight.add(self.lora_B.weight.matmul(self.lora_A.weight) * self.scaling)
            self.merged = True
    
    #def train(self, mode=True):
    #    return super().train(mode)

class LoRALinearClear(nn.Module):
    """
    Linear layer with LoRA

    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        r (int): rank of the LoRA injected matrices
        lora_alpha (int): scaling parameter for LoRA
        lora_dropout (float): dropout probability for LoRA
        merge_weights (bool): merge the weights during training
        state_dict (dict): state dict to load for linear layer
        bias (string): bias for the linear layer (set to not trainable if 'none'
        freeze_A (bool): freeze the A matrix in LoRA following the FFALoRA paper
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        fan_in_fan_out: bool = False,
        bias: str = 'none',
        #state_dict: dict = None,
        freeze_A: bool = False,
        non_linearity: str = 'gelu',
        init_method: str = 'normal',
        kaiming_a: float = 1,
        linear_layer: nn.Module = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.merge_weights = merge_weights
        self.freeze_A = freeze_A
        self.fan_in_fan_out = fan_in_fan_out
        self.update_bias = bias
        self.non_linearity = non_linearity

        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        # initialize model parameters:
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

        self.linear = linear_layer

        self.half_precision = True if self.linear.weight.dtype == torch.float16 else False


        if self.r > 0:
            device = self.linear.weight.device
            self.lora_A = nn.Linear(in_features=in_features, out_features=r, bias=False, device=device)
            self.lora_B = nn.Linear(in_features=r, out_features=out_features, bias=False, device=device)
            self.scaling = self.lora_alpha / self.r
            
            # Freezing the pre-trained weight matrix
            self.linear.weight.requires_grad = False
            #self.weight.requires_grad = False

            #if self.bias is not None:
            if self.linear.bias is not None:
                if self.update_bias == 'none':
                    self.linear.bias.requires_grad = False
                    #self.bias.requires_grad = False
                else:
                    self.linear.bias.requires_grad = True
                    #self.bias.requires_grad = True
            if freeze_A:
                self.lora_A.weight.requires_grad = False
                if self.half_precision:
                    self.lora_A.half()

        self.init_method = init_method
        self.kaiming_a = kaiming_a
        self.reset_parameters()

        self.merged = False
        #if self.fan_in_fan_out:
        #    self.weight = self.weight.transpose(0, 1)
        
        self.train()

    def half(self):
        super().half()
        if self.r > 0:
            if not self.freeze_A:
                self.lora_A.float()
            self.lora_B.float()
        self.half_precision = True


    def reset_parameters(self):
        #self.reset_linear_parameters()
        # No need to reset the parameters of the linear layer from a pre-trained model
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            if self.init_method == 'normal':
                self.lora_A.reset_parameters()
            elif self.init_method == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=self.kaiming_a)
            elif self.init_method == 'kaiming_normal':
                torch.nn.init.kaiming_normal_(self.lora_A.weight, a=self.kaiming_a)
            elif self.init_method == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(self.lora_A.weight)
            elif self.init_method == 'xavier_normal':
                torch.nn.init.xavier_normal_(self.lora_A.weight)
            elif self.init_method == 'orthogonal':
                torch.nn.init.orthogonal_(self.lora_A.weight)
            else:
                raise NotImplementedError("Initialization method not implemented")

            self.lora_B.weight.data.zero_()

    
    def reset_linear_parameters(self):
        """
        From torch.nn.Linear.reset_parameters
        Not implemented in crypten.nn.Linear since the init functions are 
        imported from torch.nn.init at compile time
        """
        #torch.nn.init.kaiming_uniform_(self.weight, a=pow(5,0.5))
        self.linear.reset_parameters()
           

    def forward(self, input):
        result = self.linear(input)
        if self.r > 0 and not self.merged:
            after_dropout = self.lora_dropout(input)

            if self.half_precision and not self.freeze_A:
                after_A = self.lora_A(after_dropout.to(torch.float32)).to(torch.float16)
            else:
                after_A = self.lora_A(after_dropout)
            
            if self.half_precision:
                after_B = self.lora_B(after_A.to(torch.float32)).to(torch.float16)
            else:
                after_B = self.lora_B(after_A)
            result = result + after_B * self.scaling
        return result
        

    def merge_weights_to_unload(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        if self.r > 0:
            self.linear.weight.add(T((self.lora_B.weight).matmul(self.lora_A.weight)) * self.scaling)
            self.merged = True
