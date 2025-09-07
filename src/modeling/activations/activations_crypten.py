"""
Crypten activations:
    - relu cnn.ReLU()
    - relu6 cnn.ReLU6()
    - sigmoid cnn.Sigmoid()
    - softmax cnn.Softmax()
    - hard tanh cnn.Hardtanh()
    - logsoftmax cnn.LogSoftmax()
    From crypten approximations 
    - tanh  applied to x as x.tanh()
    - softmax applied to x as x.softmax()
    - sigmoid applied to x as x.sigmoid()
    - logsoftmax applied to x as x.log_softmax()
"""
"""
Hardtanh
The Hardtanh function is a piecewise linear function that takes input from the entire real number range and squashes it to the range [-1, 1]. For inputs less than -1, it outputs -1. For inputs greater than 1, it outputs 1. For inputs within the range [-1, 1], it outputs the input value itself. This means that the Hardtanh function is not differentiable at x = -1 and x = 1, but it is faster and more computationally efficient than the Tanh function due to its simplicity.

"""
import crypten.nn as cnn
import crypten
import torch
import math

import torch.nn as nn
from crypten.config import cfg
import os

import os

"""
Introduced activations:
    - GELU (from OpenAI GPT repo)
    - erf_GELU (from python implementation in Transformers library)
    - hard_GELU (from Google BERT repo:)
    - quick_GELU (from Transformers)
    - Softmax_NN (Neural Network with 1 hidden layer (ReLU) and 1 output layer)
    - SoftmaxMaxCap (Softmax with max cap)
"""



class Softmax_NN(cnn.Module):
    def __init__(self, input_size = 128, hidden_size = 128):
        super(Softmax_NN, self).__init__()
        #Neural Network with 1 hidden layer (ReLU) and 1 output layer
        self.lin1 = cnn.Linear(input_size, hidden_size)
        self.relu = cnn.ReLU()
        self.lin2 = cnn.Linear(hidden_size, input_size)


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


class ReLU_optim(cnn.Module):
    def __init__(self):
        super(ReLU_optim, self).__init__()
    
    def forward(self, x):
        abs_x = x.abs()
        return (abs_x + x) / 2


class GELU(cnn.Module):
    """
        f(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    """
    def __init__(self):
        super().__init__()
        self.pi_const = torch.tensor(math.sqrt(2) / math.pi).item()
        self.pow = cnn.Pow()
        self.half = torch.tensor([0.5]).item()
        self.one = torch.tensor([1.0]).item()
        self.constant = torch.tensor([0.044715]).item()
        self.three = torch.tensor([3.0]).item()

        self.tanh_method = "reciprocal"
        self.sigmoid_tanh_terms = 32

    def set_tanh_method(self, method, terms=32):
        if method in ["reciprocal", "chebyshev"]:
            self.tanh_method = method
        else:
            raise ValueError(f"Unrecognized method {method} for tanh")
        self.sigmoid_tanh_terms = terms

    def forward(self, x):
        with cfg.temp_override({"functions.sigmoid_tanh_method": self.tanh_method,
                                "functions.sigmoid_tanh_terms": self.sigmoid_tanh_terms}):
            res = self.half * x * (self.one + (
                self.pi_const * (x + self.constant * self.pow((x, self.three)))).tanh())
        return res


class erf_GELU(cnn.Module):
    def __init__(self):
        super().__init__()
        self.sqrt_const = torch.tensor([math.sqrt(2)]).item()
        self.one = torch.tensor([1.0]).item()
        self.half = torch.tensor([0.5]).item()

    def forward(self, x):
        # Probably overflow
        return x * self.half * (self.one + (x * self.sqrt_const).erf())


class bolt_GELU(cnn.Module):
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
        abs_x = x.abs()
        cond = abs_x > 2.7
        y = self.optim_gelu_p0(abs_x)
        approx = (y + self.g0*abs_x + self.g3 ) * y + self.g4 + 0.5*x

        #return (1-cond) * approx + cond * self.optim_relu_abs(abs_x, x)
        return cond * (self.optim_relu_abs(abs_x, x) - approx) + approx
    


class swish(cnn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = cnn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)
    

class hard_GELU(cnn.Module):
    """
        Uses the hardTanh instead of the tanh function
    """
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()
        self.half = torch.tensor([0.5]).item()
        self.one = torch.tensor([1.0]).item()
        self.three = torch.tensor([3.0]).item()
        self.constant = torch.tensor([0.044715]).item()
        self.pi_const = torch.tensor([math.sqrt(2/math.pi)]).item()
        self.pow = cnn.Pow()
        self.tanh = cnn.Hardtanh()

    def forward(self, x):
        return self.half * x * (self.one + self.tanh(self.pi_const * (x + self.constant * self.pow((x, self.three)))))



def softmax_max(self, dim, max_value=10):
	r"""Compute the softmax of a tensor's elements along a given dimension"""
	# 0-d case
	if self.dim() == 0:
		assert dim == 0, "Improper dim argument"
		return self.new(torch.ones_like((self.data)))

	if self.size(dim) == 1:
		return self.new(torch.ones_like(self.data))

	logits = self - max_value
	numerator = logits.exp()
	with cfg.temp_override({"functions.reciprocal_all_pos": True}):
		inv_denominator = numerator.sum(dim, keepdim=True).reciprocal()
	return numerator * inv_denominator


class SoftmaxMaxCap(cnn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x, max_cap):
        return softmax_max(x, self.dim, max_cap)


"""
MPCFormer activations (or approximations):
    - softmax_2RELU
    - softmax_2QUAD
    - activation_quad
    - activation_newGeLU (from Google BERT repo)
"""
class softmax_2RELU(cnn.Module):
    def __init__(self, dim):
        super().__init__()
        self.relu = cnn.ReLU()
        self.div = cnn.Div()
        self.sum = cnn.Sum(dim=dim, keepdim=True)
        self.dim = dim

    def forward(self, x):
        relu_x = self.relu(x)
        #return self.div((func_x, self.sum(func_x)))
        with cfg.temp_override({"functions.reciprocal_all_pos": True}):
            inv_denominator = relu_x.sum(self.dim, keepdim=True).reciprocal()
        return relu_x * inv_denominator
        #sum_func_x = func_x.sum(keepdim=True, dim=self.dim)
        #dec_func_x = func_x.get_plain_text()
        #dec_sum_func_x = sum_func_x.get_plain_text()
        #return func_x / func_x.sum(keepdim=True, dim=self.dim)

class softmax_2QUAD(cnn.Module):
    def __init__(self, dim, norm = None):
        # HERE modified function to avoid overflow
        super().__init__()
        self.dim = dim
        self.norm = norm
        self.five = torch.tensor([5]).item()
        self.pow = cnn.Pow()
        self.div = cnn.Div()
        self.sum = cnn.Sum(dim=self.dim, keepdim=True)
        self.mul = cnn.Mul()
        self.one = torch.tensor([1.0]).item()

    
    def forward(self, x):
        #a, b, c, d = x.size()
        #quad = x#self.norm(x)
        intermediate_quad = x + self.five
        #dec_int = intermediate_quad.get_plain_text()
        quad = self.pow((intermediate_quad, 2))
        #return self.div((quad, self.sum(quad)))
        # Temporary solution to avoid overflow in the reciprocal should be divide the denominator by 100 or 1000 and then divide the final result by 100 or 1000
        return self.div((quad, self.sum(quad)))
        #quad / quad.sum(dim=self.dim, keepdims=True)


class activation_quad(cnn.Module):
    # Used for GELU
    def __init__(self):
        super().__init__()
        self.first_coef = torch.tensor([0.125]).item()
        self.second_coef = torch.tensor([0.5]).item()
        self.third_coef = torch.tensor([0.25]).item()
        self.pow = torch.tensor([2]).item()
     
    def forward(self, x):
        return self.first_coef*x*x + self.second_coef*x + self.third_coef

