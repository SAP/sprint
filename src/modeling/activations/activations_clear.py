import torch
import math

import torch.nn as nn
import os

import os

"""
Introduced activations:
    - GELU (from OpenAI GPT repo)
    - erf_GELU (from python implementation in Transformers library)
    - hard_GELU (from Google BERT repo:)
    - quick_GELU (from Transformers)
"""
 
class Softmax_NN(nn.Module):
    def __init__(self, input_size = 128, hidden_size = 128):
        super(Softmax_NN, self).__init__()
        #Neural Network with 1 hidden layer (ReLU) and 1 output layer
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size, input_size)
        abs_path = f"{os.path.expanduser('~')}/sprint/data/models/ma_bert_softmax_weights.pt"

        try:
            self.load_state_dict(torch.load("ma_bert_softmax_weights.pt"), strict=False)
        except FileNotFoundError:
            try:
                self.load_state_dict(torch.load(abs_path), strict=False)
            except FileNotFoundError:
                print("Random initialization of the weights for softmax NN")
                self.lin1.weight = torch.nn.Parameter(torch.randn(hidden_size, input_size))
                self.lin1.bias = torch.nn.Parameter(torch.randn(hidden_size))
                
    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        output = self.lin2(x)
        return output

class hardGELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.half = torch.tensor([0.5]).item()
        self.one = torch.tensor([1.0]).item()
        self.three = torch.tensor([3.0]).item()
        self.constant = torch.tensor([0.044715]).item()
        self.pi_const = torch.tensor([math.sqrt(2/math.pi)]).item()
        self.tanh = nn.Hardtanh()

    def forward(self, x):
        return self.half * x * (self.one + self.tanh(self.pi_const * (x + self.constant * pow(x, self.three))))
    
class boltGELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.sqrt_const = torch.tensor([math.sqrt(2)]).item()
        self.one = torch.tensor([1.0]).item()
        self.half = torch.tensor([0.5]).item()
        self.g0 = torch.tensor([0.14439048359960427]).item()
        self.g1 = torch.tensor([-0.7077117131613893]).item()
        self.g2 = torch.tensor([4.5702822654246535]).item()
        self.g3 = torch.tensor([-8.15444702051307]).item()
        self.g4 = torch.tensor([16.382265425072532]).item()


    def optim_relu_abs(self, abs_x, x):
        return (abs_x + x) * 0.5

    def optim_gelu_p0(self, abs_x):
        return (self.g0 * abs_x + self.g1) * abs_x + self.g2
    
    def forward(self, x):
        abs_x = abs(x)
        #abs_x = x.abs()

        cond = abs_x > 2.7
        y = self.optim_gelu_p0(abs_x)
        approx = (y + self.g0*abs_x + self.g3 ) * y + self.g4 + 0.5*x

        #return (1-cond) * approx + cond * self.optim_relu_abs(abs_x, x)

        return cond * (self.optim_relu_abs(abs_x, x) - approx) + approx

def softmax_max(x, dim=-1, max_cap=10.0):
    logits = x - max_cap
    logits_exp = logits.exp()
    return logits_exp / logits_exp.sum(dim=dim, keepdim=True)


class SoftmaxMax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x, max_cap):
        return softmax_max(x, self.dim, max_cap)


def scaled_attention(attention_scores):
    sequence_length = attention_scores.size(-1) # it should be 128
    return attention_scores / sequence_length

def relurs_attention(attention_scores):
    return torch.relu(attention_scores)


"""
MPCFormer activations (or approximations):
    - softmax_2RELU
    - softmax_2QUAD
    - activation_quad
"""
class softmax_2RELU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.dim = dim

    def forward(self, x):
        relu_x = self.relu(x)
        denominator = torch.sum(relu_x, dim=self.dim, keepdim=True)
        return torch.div(relu_x, denominator)


class activation_quad(nn.Module):
    # Used for GELU
    def __init__(self):
        super().__init__()
        self.first_coef = torch.tensor([0.125]).item()
        self.second_coef = torch.tensor([0.5]).item()
        self.third_coef = torch.tensor([0.25]).item()
        self.pow = torch.tensor([2]).item()
     
    def forward(self, x):
        return self.first_coef*x*x + self.second_coef*x + self.third_coef

class softmax_2QUAD(nn.Module):
    def __init__(self, dim, norm = None):
        # HERE modified function to avoid overflow
        super().__init__()
        self.dim = dim
        self.norm = norm
        self.five = torch.tensor([5]).item()
        self.one = torch.tensor([1.0]).item()

    
    def forward(self, x):
        intermediate_quad = x + self.five
        quad = torch.pow(intermediate_quad, 2)
        return torch.div(quad, torch.sum(quad, dim=self.dim, keepdim=True))