#!/usr/bin/env python3
import matplotlib.pyplot as plt
import math
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import wandb
import torch
import torch.distributions as dist
import torch.nn as nn
import numpy as np
from copy import deepcopy
import time
from typing import List

# params to generate ys
ys = torch.tensor([[ 0.1198],
                   [-0.1084],
                   [ 0.2107],
                   [ 0.2983],
                   [-0.5857],
                   [ 0.2646],
                   [-1.0533],
                   [-0.9299],
                   [ 0.2199],
                   [ 1.1221]])
gen_A = torch.tensor(0.8).reshape(1, 1)
gen_Q = torch.tensor(0.2).reshape(1, 1)
gen_C = torch.tensor(0.8).reshape(1, 1)
gen_R = torch.tensor(0.2).reshape(1, 1)
gen_mu_0 = torch.tensor(0.).reshape(1, 1)
gen_Q_0 = gen_Q

# # test params
# A = torch.tensor(1.).reshape(1, 1)
# Q = torch.tensor(1.).reshape(1, 1)
# C = torch.tensor(1).reshape(1, 1)
# R = torch.tensor(1.).reshape(1, 1)
# mu_0 = torch.tensor(0.).reshape(1, 1)
# Q_0 = Q

# # verification params
# ys = torch.rand(10, 1)
# A = torch.tensor(2.).reshape(1, 1)
# Q = torch.tensor(3.).reshape(1, 1)
# C = torch.tensor(0.5).reshape(1, 1)
# R = torch.tensor(2.).reshape(1, 1)
# mu_0 = torch.zeros_like(ys[0])
# Q_0 = Q

def state_transition_from_dist(distribution):
    """ Computes a state transition from x_t using distribution """
    return distribution.rsample()

def state_transition(x_t, A, Q):
    """
    Computes state transition from x_t defined by x_{t+1} = Ax_t + w
    where $w \sim N(0, Q)$
    """
    return state_transition_from_dist(dist.MultivariateNormal(A*x_t, Q))

def score_state_transition(xt, prev_xt, A, Q):
    """ Scores xt against the prior N(A*prev_xt, Q) """
    return dist.MultivariateNormal(A*prev_xt, Q).log_prob(xt)

def generate_y_from_dist(distribution):
    """ Computes a state transition from x_t using distribution """
    return distribution.rsample()

def generate_y(x_t, C, R):
    """ Computes a state transition from x_t using distribution """
    return generate_y_from_dist(dist.MultivariateNormal(C*x_t, R))

def score_y_from_dist(y_test, distribution):
    """ Scores y_test against a distribution object """
    return distribution.log_prob(y_test)

def score_y(y_test, x_t, C, R):
    """
    Scores y_test under the likelihood defined by y_t = Ax_t + v
    where $v \sim N(0, R)$
    """
    return score_y_from_dist(y_test, dist.MultivariateNormal(C*x_t, R))

def neg_two_log_prob(xs, ys, A, Q, C, R, mu0, Q0):
    assert len(xs) == len(ys)

    tau = len(xs)
    p, k = C.shape

    y_term = ys - C*xs
    r_term = torch.matmul(y_term.t() * torch.inverse(R), y_term) + torch.logdet(R)*tau

    q_term = torch.zeros_like(r_term)
    for i in range(len(xs)-1):
        xt = xs[i]
        xt_plus_1 = xs[i+1]
        x_term = xt_plus_1 - A*xt
        q_term += torch.matmul(x_term.t() * torch.inverse(Q), x_term) + torch.logdet(Q)

    x0_term = xs[0] - mu0
    zero_term = x0_term.t() * torch.inverse(Q0) * x0_term + tau * (p + k) * torch.log(2*torch.tensor(math.pi))
    return r_term + q_term + zero_term

def log_joint(xs, ys, A, Q, C, R, mu0, Q0):
    return -neg_two_log_prob(xs, ys, A, Q, C, R, mu0, Q0) / 2

def compute_joint(xs, ys, A, Q, C, R, mu0, Q0):
    return torch.exp(-neg_two_log_prob(xs, ys, A, Q, C, R, mu0, Q0) / 2)

def get_start_state(mu_0, Q_0):
    return dist.MultivariateNormal(mu_0, Q_0).rsample()

def score_initial_state(x0, mu_0, Q_0):
    """ Scores xt against the prior N(mu_0, Q_0) """
    return dist.MultivariateNormal(mu_0, Q_0).log_prob(x0)

def y_dist(num_steps, A=gen_A,
           Q=gen_Q, C=gen_C, R=gen_R,
           mu_0=gen_mu_0, Q_0=gen_Q_0):
    var_x = A**(2*num_steps)*Q_0 + Q*(A**(2*num_steps)-1)/(A**2-1)
    return dist.Normal(C*A**num_steps*mu_0, C**2*var_x + R)

def sample_y(num_steps, A=gen_A,
             Q=gen_Q, C=gen_C, R=gen_R,
             mu_0=gen_mu_0, Q_0=gen_Q_0):
    d = y_dist(num_steps)
    y_sample = d.rsample()
    score = d.log_prob(y_sample)
    return y_sample, score, d

def generate_trajectory(num_steps, A=gen_A,
                        Q=gen_Q, C=gen_C, R=gen_R,
                        mu_0=gen_mu_0, Q_0=gen_Q_0):
    ys = []
    xs = []
    xt = get_start_state(mu_0, Q_0)
    xs.append(xt)
    priors = []
    liks = []
    for i in range(num_steps):
        yt = generate_y(xt, C, R)
        ys.append(yt)
        liks.append(score_y(yt, xt, C, R))
        prev_xt = xt.clone()
        xt = state_transition(xt, A, Q)
        xs.append(xt)
        priors.append(score_state_transition(xt, prev_xt, A, Q))
    return torch.cat(ys), torch.cat(xs), torch.cat(priors), torch.cat(liks)
