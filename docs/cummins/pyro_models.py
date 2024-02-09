"""
Author: Maxwell Bolt

Bayesian Linear Regression Model
From example: https://pyro.ai/examples/bayesian_regression.html

Bayesian Neural Network Model
From example: 
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Bayesian_Neural_Networks/dl2_bnn_tut1_students_with_answers.html

"""

__all__ = ["BLR", "PyroBNN"]

import os
import time

import numpy as np
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from torch import nn


class BLR(PyroModule):
    """
    Performs simple Bayesian linear regression with a standard Normal prior on
    ``weight``.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean
    
class PyroBNN(PyroModule):
    def __init__(self, dim, out_dim, hidden_dim, num_hidden_layer, prior_scale=1.):
            super().__init__()

            self.activation = nn.ReLU()  # could also be ReLU or LeakyReLU
            assert dim > 0 and out_dim > 0 and hidden_dim > 0 and num_hidden_layer > 0  # make sure the dimensions are valid

            # Define the layer sizes and the PyroModule layer list
            self.layer_sizes = [dim] + num_hidden_layer * [hidden_dim] + [out_dim]
            layer_list = [PyroModule[nn.Linear](self.layer_sizes[idx - 1], self.layer_sizes[idx]) for idx in
                        range(1, len(self.layer_sizes))]
            self.layers = PyroModule[nn.ModuleList](layer_list)

            for layer_idx, layer in enumerate(self.layers):
                layer.weight = PyroSample(dist.Normal(0., prior_scale * np.sqrt(2 / self.layer_sizes[layer_idx])).expand(
                    [self.layer_sizes[layer_idx + 1], self.layer_sizes[layer_idx]]).to_event(2))
                layer.bias = PyroSample(dist.Normal(0., prior_scale).expand([self.layer_sizes[layer_idx + 1]]).to_event(1))

    def forward(self, x, y):
        
        x = self.activation(self.layers[0](x))  # input --> hidden
        for layer in self.layers[1:-1]:
            x = self.activation(layer(x))  # hidden --> hidden
        mu = self.layers[-1](x).squeeze()  # hidden --> output
        sigma = pyro.sample("sigma", dist.Gamma(.5, 1))  # infer the response noise

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma * sigma), obs=y)
        return mu

