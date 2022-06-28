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


def gen_covariance_matrix(d):
    sigma_d = torch.rand(d, d)
    return torch.mm(sigma_d, sigma_d.t()) + torch.eye(d)

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

# single dim gen params
single_gen_A = torch.tensor(0.8).reshape(1, 1)
single_gen_Q = torch.tensor(0.2).reshape(1, 1)
single_gen_C = torch.tensor(0.8).reshape(1, 1)
single_gen_R = torch.tensor(0.2).reshape(1, 1)
single_gen_mu_0 = torch.tensor(0.).reshape(1, 1)
single_gen_Q_0 = single_gen_Q

# multidim test params
gen_A = torch.rand(2, 2)
gen_Q = gen_covariance_matrix(2)
gen_C = torch.rand(1, 2)
gen_R = torch.rand(1, 1)
gen_mu_0 = torch.zeros(2)
gen_Q_0 = gen_Q

# test params
test_A = torch.tensor(0.5).reshape(1, 1)
test_Q = torch.tensor(0.4).reshape(1, 1)
test_C = torch.tensor(1.2).reshape(1, 1)
test_R = torch.tensor(0.9).reshape(1, 1)
test_mu_0 = torch.tensor(0.).reshape(1, 1)
test_Q_0 = test_Q

# multidim test params
A = torch.rand(2, 2)
Q = gen_covariance_matrix(2)
C = torch.rand(1, 2)
R = torch.rand(1, 1)
mu_0 = torch.zeros(2)
Q_0 = Q

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

def get_state_transition_dist(x_t, A, Q):
    x_shape = (A.shape[1], -1)
    return dist.MultivariateNormal(torch.mm(A, x_t.reshape(x_shape)).squeeze(1), Q)

def state_transition(x_t, A, Q):
    """
    Computes state transition from x_t defined by x_{t+1} = Ax_t + w
    where $w \sim N(0, Q)$
    """
    x_shape = (A.shape[1], -1)
    return state_transition_from_dist(get_state_transition_dist(x_t, A, Q))

def score_state_transition(xt, prev_xt, A, Q):
    """ Scores xt against the prior N(A*prev_xt, Q) """
    prev_xt_shape = (A.shape[1], -1)
    d = get_state_transition_dist(prev_xt, A, Q)
    return d.log_prob(xt)

def generate_y_from_dist(distribution):
    """ Computes a state transition from x_t using distribution """
    return distribution.rsample()

def generate_y(x_t, C, R):
    """ Computes a state transition from x_t using distribution """
    x_shape = (C.shape[1], -1)
    return generate_y_from_dist(dist.MultivariateNormal(torch.mm(C, x_t.reshape(x_shape)).reshape(1), R))

def score_y_from_dist(y_test, distribution):
    """ Scores y_test against a distribution object """
    return distribution.log_prob(y_test.reshape(distribution.mean.shape))

def score_y(y_test, x_t, C, R):
    """
    Scores y_test under the likelihood defined by y_t = Ax_t + v
    where $v \sim N(0, R)$
    """
    # assume that y will always be
    x_shape = (C.shape[1], -1)
    return score_y_from_dist(y_test, dist.MultivariateNormal(torch.mm(C, x_t.reshape(x_shape)).reshape(1), R))

def get_start_state(mu_0, Q_0):
    return dist.MultivariateNormal(mu_0, Q_0).rsample()

def score_initial_state(x0, mu_0, Q_0):
    """ Scores xt against the prior N(mu_0, Q_0) """
    return dist.MultivariateNormal(mu_0, Q_0).log_prob(x0)

def mat_sum(A, num_steps, coef=torch.tensor(1.)):
    output = torch.zeros_like(A)
    for i in range(num_steps):
        output += torch.matrix_power(A, coef*i)
    return output

def y_dist(num_steps, A=gen_A,
           Q=gen_Q, C=gen_C, R=gen_R,
           mu_0=gen_mu_0, Q_0=gen_Q_0):
    first_term = torch.mm(torch.matrix_power(A, 2*num_steps), Q_0)

    eye = torch.eye(A.shape[0])
    sum_of_a_sq = torch.mm(torch.matrix_power(A, 2*num_steps)-eye, torch.inverse(torch.matrix_power(A, 2)-eye))
    second_term = torch.mm(Q, sum_of_a_sq)

    var_x = first_term + second_term

    var_y = torch.mm(torch.mm(C, var_x), C.t()) + R
    mean = torch.mm(C, torch.mm(torch.matrix_power(A, num_steps), mu_0.reshape(A.shape[1], 1)))

    # Note: I am transforming this distribution to a Normal (as opposed to MultivariateNormal)
    # so that I can compute the cdf. Moreover, the second parameter of a MultivariateNormal
    # is the covariance whereas for a Normal, it's the standard deviation
    return dist.Normal(mean, torch.sqrt(var_y))

def sample_y(num_steps, A=gen_A,
             Q=gen_Q, C=gen_C, R=gen_R,
             mu_0=gen_mu_0, Q_0=gen_Q_0):
    d = y_dist(num_steps, A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0)
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
        liks.append(score_y(yt, xt, C, R).reshape(-1))
        prev_xt = xt.clone()
        xt = state_transition(xt, A, Q)
        xs.append(xt)
        priors.append(score_state_transition(xt, prev_xt, A, Q).reshape(-1))
    return torch.cat(ys), torch.cat(xs), torch.cat(priors), torch.cat(liks)
