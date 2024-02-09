"""
Author: Maxwell Bolt

Numpyro Bayesian Neural Network Model
From example:
https://num.pyro.ai/en/stable/examples/bnn.html
"""

__all__ = ["create_sliding_window", "jnp_model1", "jnp_model2", "jnp_run_inference", "jnp_predict"]

import os
import time

import jax
from jax import vmap, jit
import jax.numpy as jnp
import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# Function to create sliding window
def create_sliding_window(input_data, output_data, window_size):
    inputs = []
    outputs = []
    for i in range(len(input_data) - window_size):
        inputs.append(input_data[i:i+window_size])
        outputs.append(output_data[i+window_size])
    return jnp.array(inputs), jnp.array(outputs)

# the non-linearity we use in our neural network
def nonlin(x):
    return jnp.tanh(x)
    # return jax.nn.relu(x)

def jnp_model1(X, Y, D_H, D_Y=1):
    """
    BNN using numpyro
    """
    N, D_X = X.shape

    # sample first layer (we put unit normal priors on all weights)
    w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((D_X, D_H)), jnp.ones((D_X, D_H))))
    assert w1.shape == (D_X, D_H)
    z1 = nonlin(jnp.matmul(X, w1))  # <= first layer of activations
    assert z1.shape == (N, D_H)

    # sample second layer
    w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((D_H, D_H)), jnp.ones((D_H, D_H))))
    assert w2.shape == (D_H, D_H)
    z2 = nonlin(jnp.matmul(z1, w2))  # <= second layer of activations
    assert z2.shape == (N, D_H)

    # sample third layer
    w3 = numpyro.sample("w3", dist.Normal(jnp.zeros((D_H, D_H)), jnp.ones((D_H, D_H))))
    assert w3.shape == (D_H, D_H)
    z3 = nonlin(jnp.matmul(z2, w3))  # <= second layer of activations
    assert z3.shape == (N, D_H)

    # sample fourth layer
    w4 = numpyro.sample("w4", dist.Normal(jnp.zeros((D_H, D_H)), jnp.ones((D_H, D_H))))
    assert w4.shape == (D_H, D_H)
    z4 = nonlin(jnp.matmul(z3, w4))  # <= second layer of activations
    assert z4.shape == (N, D_H)

    # sample fifth layer
    w5 = numpyro.sample("w5", dist.Normal(jnp.zeros((D_H, D_H)), jnp.ones((D_H, D_H))))
    assert w5.shape == (D_H, D_H)
    z5 = nonlin(jnp.matmul(z4, w5))  # <= second layer of activations
    assert z5.shape == (N, D_H)

    # sample final layer of weights and neural network output
    wf = numpyro.sample("wf", dist.Normal(jnp.zeros((D_H, D_Y)), jnp.ones((D_H, D_Y))))
    assert wf.shape == (D_H, D_Y)
    zf = jnp.matmul(z5, wf)  # <= output of the neural network
    assert zf.shape == (N, D_Y)

    if Y is not None:
        assert zf.shape == Y.shape

    # we put a prior on the observation noise
    # prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    # prec_obs = numpyro.sample("prec_obs", dist.InverseGamma(3.0, 1.0))
    raw_prec_obs = numpyro.sample("prec_obs", dist.Normal(0.0, 1.0))
    prec_obs = raw_prec_obs ** 2
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # observe data
    with numpyro.plate("data", N):
        # note we use to_event(1) because each observation has shape (1,)
        numpyro.sample("Y", dist.Normal(zf, sigma_obs).to_event(1), obs=Y)

def layers(X, Y, D_H, D_Y=1, prefix=""):
    
    N, D_X = X.shape

    # sample first layer (we put unit normal priors on all weights)
    w1 = numpyro.sample(prefix + "w1", dist.Normal(jnp.zeros((D_X, D_H)), jnp.ones((D_X, D_H))))
    assert w1.shape == (D_X, D_H)
    z1 = nonlin(jnp.matmul(X, w1))  # <= first layer of activations
    assert z1.shape == (N, D_H)

    # sample second layer
    w2 = numpyro.sample(prefix + "w2", dist.Normal(jnp.zeros((D_H, D_H)), jnp.ones((D_H, D_H))))
    assert w2.shape == (D_H, D_H)
    z2 = nonlin(jnp.matmul(z1, w2))  # <= second layer of activations
    assert z2.shape == (N, D_H)

    # # sample third layer
    # w3 = numpyro.sample(prefix + "w3", dist.Normal(jnp.zeros((D_H, D_H)), jnp.ones((D_H, D_H))))
    # assert w3.shape == (D_H, D_H)
    # z3 = nonlin(jnp.matmul(z2, w3))  # <= second layer of activations
    # assert z3.shape == (N, D_H)

    # # sample fourth layer
    # w4 = numpyro.sample(prefix + "w4", dist.Normal(jnp.zeros((D_H, D_H)), jnp.ones((D_H, D_H))))
    # assert w4.shape == (D_H, D_H)
    # z4 = nonlin(jnp.matmul(z3, w4))  # <= second layer of activations
    # assert z4.shape == (N, D_H)

    # # sample fifth layer
    # w5 = numpyro.sample(prefix + "w5", dist.Normal(jnp.zeros((D_H, D_H)), jnp.ones((D_H, D_H))))
    # assert w5.shape == (D_H, D_H)
    # z5 = nonlin(jnp.matmul(z4, w5))  # <= second layer of activations
    # assert z5.shape == (N, D_H)

    # sample final layer of weights and neural network output
    wf = numpyro.sample(prefix + "wf", dist.Normal(jnp.zeros((D_H, D_Y)), jnp.ones((D_H, D_Y))))
    assert wf.shape == (D_H, D_Y)
    zf = jnp.matmul(z2, wf)  # <= output of the neural network
    assert zf.shape == (N, D_Y)

    if Y is not None:
        assert zf.shape == Y.shape

    return zf


def jnp_model2(X, Y, D_H, D_Y=1):
    """BNN using numpyro, outputs 2 values to form the mean and covariance 
    of a normal for the likelihood.
    """

    N, D_X = X.shape

    mean = layers(X, Y, D_H, D_Y=1, prefix="mean_")

    raw_cov = layers(X, Y, D_H, D_Y=1, prefix="cov_")
    cov = jnp.exp(raw_cov)

    # observe data
    with numpyro.plate("data", N):
        numpyro.sample("Y", dist.Normal(mean, cov).to_event(1), obs=Y)

# helper function for HMC inference
def jnp_run_inference(model, rng_key, X, Y, D_H, num_chains=1, num_warmup=1000, num_samples=1000):
    """Runs NUTS on the numpyro model."""
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, X, Y, D_H)
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()

# helper function for prediction
def jnp_predict(model, rng_key, samples, X, D_H):
    """Predicts the output of the model given samples from the posterior"""

    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None, D_H=D_H)
    return model_trace["Y"]["value"]
