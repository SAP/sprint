# Copyright (c) 2025 SAP SE or an SAP affiliate company and sprint contributors
# SPDX-License-Identifier: Apache-2.0

"""
Code adapted from: https://github.com/microsoft/LoRA/blob/main/loralib/utils.py
"""


import torch
import torch.nn as nn

import crypten
import crypten.nn as cnn

from typing import Dict

from modeling.lora.modeling_lora import LoRALayer, LoraConfig, LoRALinearClear, LoRALinearCrypten



def mark_only_lora_as_trainable(model: cnn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    mark_bias_as_trainable(model, bias)


def mark_bias_as_trainable(model: cnn.Module, bias: str = 'none') -> None:
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
    


def get_peft_model(
    model, 
    peft_config: LoraConfig,
    non_linearity: str = 'relu',
) -> cnn.Module:
    
    named_modules = list(model.named_modules())
    if isinstance(model, cnn.Module):
        encrypted = model.encrypted
    for name, module in named_modules:
        if any([module_to_save in name for module_to_save in peft_config.modules_to_save]):
            #print(name)
            # This modules will be retrained, so set requires_grad to True
            for n, p in module.named_parameters():
                p.requires_grad = True
        elif any([target_module in name for target_module in peft_config.target_modules]):
            #print(name)
            weight = module.weight
            if isinstance(module, cnn.Module):
                new_module = LoRALinearCrypten(
                    in_features = weight.shape[1],
                    out_features = weight.shape[0],
                    r = peft_config.r,
                    lora_dropout=peft_config.lora_dropout,
                    lora_alpha = peft_config.lora_alpha,
                    #state_dict = module.state_dict(),
                    encrypted = encrypted,
                    fan_in_fan_out=peft_config.fan_in_fan_out,
                    freeze_A=peft_config.freeze_A,
                    bias=peft_config.bias,
                    non_linearity=non_linearity,
                    init_method=peft_config.init_method,
                    linear_layer=module,
                )
            else:
                new_module = LoRALinearClear(
                    in_features = weight.shape[1],
                    out_features = weight.shape[0],
                    r = peft_config.r,
                    lora_dropout=peft_config.lora_dropout,
                    lora_alpha = peft_config.lora_alpha,
                    #state_dict = module.state_dict(),
                    fan_in_fan_out=peft_config.fan_in_fan_out,
                    freeze_A=peft_config.freeze_A,
                    bias=peft_config.bias,
                    non_linearity=non_linearity,
                    init_method=peft_config.init_method,
                    linear_layer=module,
                )
            
            name_splits = name.split('.')
            parent_module_name = '.'.join(name_splits[:-1])
            child_module_name = name_splits[-1]
            
            parent_module = model
            for attr in parent_module_name.split('.'):
                parent_module = getattr(parent_module, attr)
            
            setattr(parent_module, child_module_name, new_module)
        elif 'lora_' in name:
            if "A" in name and peft_config.freeze_A:
                # This modules will not be retrained, so set requires_grad to False
                for n, p in module.named_parameters():
                    p.requires_grad = False
            else:
                for n, p in module.named_parameters():
                    p.requires_grad = True
        else:
            # This modules will not be retrained, so set requires_grad to False
            for n, p in module.named_parameters():
                p.requires_grad = False

    mark_bias_as_trainable(model, peft_config.bias)


#def get_peft_model_return_new_model(
#    model: cnn.Module,
#    peft_config: LoraConfig,
#) -> cnn.Module:
    

def merge_weights_and_unloads(
    model: cnn.Module, 
) -> None:
    encrypted = model.encrypted
    named_modules = list(model.named_modules())
    for name, module in named_modules:
        if isinstance(module, LoRALinearCrypten): 
            module.merge_weights_to_unload()
            weight = module.weight
            if isinstance(module, LoRALinearCrypten):
                new_module = cnn.Linear(
                    in_features = weight.shape[1],
                    out_features = weight.shape[0],
                )
            if encrypted:
                new_module.encrypt()
            new_module.load_state_dict(module.state_dict())
            name_splits = name.split('.')
            parent_module_name = '.'.join(name_splits[:-1])
            child_module_name = name_splits[-1]

            parent_module = model
            for attr in parent_module_name.split('.'):
                parent_module = getattr(parent_module, attr)

            setattr(parent_module, child_module_name, new_module)
            

    


def decrypt_non_lora_parameters(
    model: cnn.Module,
    peft_config: LoraConfig,
) -> None:
    # if the parameter is not a lora parameter, or one of the modules to save, then decrypt it
    for name, module in model.named_modules():

        #just for testing
        if "classifier" in name and not "lora" in name:
            for n,p in module._parameters.items():
                if isinstance(p, torch.Tensor):
                    module.set_parameter(n, crypten.cryptensor(p))


        if isinstance(module, LoRALinearCrypten) or "lora" in name:
            continue
        if any([lora_module in name for lora_module in peft_config.target_modules]):
            continue
        if any([module_to_save in name for module_to_save in peft_config.modules_to_save]):
            continue
        
        for n,p in module._parameters.items():
            if 'bias' in n and peft_config.bias == 'all':
                continue
            if isinstance(p, crypten.CrypTensor):
                module.set_parameter(n, p.get_plain_text())
                #module.load_state_dict({n: p.get_plain_text()})


def get_lora_config_from_model(model, model_type):
    if "roberta" in model_type:
        model = model.roberta
    elif "bert" in model_type:
        model = model.bert
    else:
        raise ValueError("Model type must be either bert or roberta")    
        
    lora_alpha = model.encoder.layer[0].attention.self.query.lora_alpha
    lora_rank = model.encoder.layer[0].attention.self.query.r
    fa_lora = model.encoder.layer[0].attention.self.query.freeze_A
    fan_in_fan_out = model.encoder.layer[0].attention.self.query.fan_in_fan_out
    lora_dropout = model.encoder.layer[0].attention.self.query.lora_dropout

    # retrieve target modules from the model
    target_modules = []
    if isinstance(model.encoder.layer[0].attention.self.query, LoRALinearClear):
        target_modules.append("query")
    if isinstance(model.encoder.layer[0].attention.self.key, LoRALinearClear):
        target_modules.append("key")
    if isinstance(model.encoder.layer[0].attention.self.value, LoRALinearClear):
        target_modules.append("value")
    if isinstance(model.encoder.layer[0].attention.output.dense, LoRALinearClear):
        target_modules.append("dense")

   
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        freeze_A=fa_lora,
        modules_to_save=[
            "classifier",
        ],
        bias = "none",
        target_modules=target_modules,
        fan_in_fan_out=fan_in_fan_out
    )

    return lora_config