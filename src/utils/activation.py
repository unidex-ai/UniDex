import torch.nn as nn

ACTIVATION_MAPS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "softmax": nn.Softmax,
    "softplus": nn.Softplus,
    "swish": nn.SiLU,
    "mish": nn.Mish,
}