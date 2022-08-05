#!/usr/bin/env python3
import os
from datetime import date
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from copy import deepcopy
from generative_model import y_dist, sample_y, generate_trajectory, \
    gen_A, gen_Q, gen_C, gen_R, gen_mu_0, gen_Q_0, \
    test_A, test_Q, test_C, test_R, test_mu_0, test_Q_0, \
    state_transition, score_state_transition, gen_covariance_matrix, \
    score_initial_state, score_y
    # A, Q, C, R, mu_0, Q_0, \
import wandb
from linear_gaussian_env import LinearGaussianEnv, LinearGaussianSingleYEnv
from math_utils import logvarexp, importance_sampled_confidence_interval, log_effective_sample_size, log_max_weight_proportion, log_mean
from plot_utils import legend_without_duplicate_labels
from linear_gaussian_prob_prog import \
    MultiGaussianRandomVariable, GaussianRandomVariable, MultiLinearGaussian, \
    LinearGaussian, VecLinearGaussian, get_linear_gaussian_variables, JointVariables, \
    RandomVariable
from evaluation import EvaluationObject
from dimension_table import create_dimension_table


class FilteringPosterior:
    def __init__(self, numerator, left):
        self.numerator = numerator
        self.left = left

    def condition(self, y_values, x_value=None):
        if self.numerator.right is not None:
            joint = self.numerator * self.numerator.right.marginal()
        else:
            joint = self.numerator

        names = [y.name for y in self.numerator.left if 'y' in y.name]
        sorted_idx = np.argsort(names)
        r_vars = np.array([y for y in self.numerator.left if 'y' in y.name])
        r_vars = r_vars[sorted_idx]
        r_vars = list(r_vars) + [self.numerator.right] if self.numerator.right is not None else list(r_vars)

        if x_value is not None:
            values = torch.cat([y_values, x_value])
            # values = torch.cat([y_values.reshape(x_value.shape[0], -1), x_value.reshape(-1, 1)], dim=1)
        else:
            values = y_values
        values = values.squeeze()

        rvs = []
        # for r_var, val in zip(r_vars, values):
        current_idx = 0
        for r_var in self.left:
            dim = r_var.mu.nelement()
            rv = RandomVariable(r_var=r_var, value=values[current_idx:current_idx+dim])
            rvs.append(rv)
            current_idx += dim

        # # the next part is to ensure that the random variables
        # # in joint correspond to the y_values and x_values that
        # # are passed for conditioning
        # names = [y.name for y in self.numerator.left if 'y' in y.name]
        # r_vars = [y for y in self.numerator.left if 'y' in y.name]
        # sorted_idx = np.argsort(names)
        # vals = y_vals[sorted_idx]
        # if x_value is not None:
        #     # get index of x
        #     x_idx = joint.left.index(self.numerator.right)
        #     if x_idx == 0:
        #         vals = torch.cat([x_value, vals], dim=1)
        #     elif x_idx == len(joint.left)-1:
        #         vals = torch.cat([vals, x_value], dim=1)
        #     else:
        #         vals = torch.cat([vals[0:x_idx], x_value, vals[x_idx:]], dim=1)

        # return joint.condition(r_vars, vals)
        try:
            return joint.condition(rvs)
        except:
            import pdb; pdb.set_trace()
            return joint.condition(rvs)


def compute_filtering_data_structures(dim, num_obs):
    lgv = get_linear_gaussian_variables(dim=dim, num_obs=num_obs)
    xs = lgv.xs
    ys = lgv.ys

    # compute denominator
    p_y_next_given_x = [None] * (num_obs-1)

    p_y_T_x_T_given_x_T_minus_1 = ys[-1].likelihood() * xs[-1].likelihood()
    p_y_next_given_x[-1] = p_y_T_x_T_given_x_T_minus_1.marginalize_out(xs[-1])

    current_index = -2

    for i in range(len(p_y_next_given_x)-2, -1, -1):
        x_dist = xs[current_index].likelihood()
        lik = ys[current_index].likelihood()
        p_ys_given_x_t = lik * p_y_next_given_x[current_index+1]
        p_ys_x_t_given_x_t_minus_1 = p_ys_given_x_t * x_dist
        p_y_next_given_x[current_index] = p_ys_x_t_given_x_t_minus_1.marginalize_out(xs[current_index])

        current_index -= 1
    return xs, ys, p_y_next_given_x

def old_compute_filtering_posterior(t, num_obs, xs, ys, p_y_next_given_x, A, C):
    if t == 0:
        lik = ys[0].likelihood()
        cond_ys = lik * p_y_next_given_x[0]
        prior = xs[0].prior()
        try:
            numerator = cond_ys * prior
        except:
            import pdb; pdb.set_trace()
            numerator = cond_ys * prior

        denominator = JointVariables(ys, A=A, C=C).dist
    else:
        lik = xs[t].likelihood()
        numerator = ys[t].likelihood()
        if t < num_obs - 1:
            numerator *= p_y_next_given_x[t]
        numerator *= lik
        denominator = p_y_next_given_x[t-1]

    return FilteringPosterior(numerator, denominator.left + [x for x in [denominator.right] if x is not None])

def compute_filtering_posterior(t, num_obs, xs, ys, A, C):
    rest_of_ys = ys[t+1:]
    rvars = [ys[t], xs[t]] + rest_of_ys
    rvars += [xs[t-1]] if t > 0 else []
    jvs = JointVariables(rvars, A=A, C=C)
    condition_vars = [ys[t]] + rest_of_ys
    condition_vars = condition_vars + [xs[t-1]] if t > 0 else condition_vars
    return FilteringPosterior(jvs.dist, condition_vars)

def old_compute_filtering_posteriors(table, num_obs, dim, ys=None):
    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    mu_0 = table[dim]['mu_0']
    Q_0 = table[dim]['Q_0']

    if ys is None:
        ys = generate_trajectory(num_obs, A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0)[0]
    env = LinearGaussianEnv(A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0, ys=ys, sample=False)

    assert(num_obs == len(ys))

    fp_xs, fp_ys, p_y_next_given_x = compute_filtering_data_structures(dim=dim, num_obs=num_obs)

    # true evidence
    jvs = JointVariables(fp_ys, A=A, C=C)
    print('true evidence: ', jvs.dist.log_prob(ys).exp())

    fps = []
    for t in range(num_obs):
        filtering_posterior = old_compute_filtering_posterior(t, num_obs, fp_xs, fp_ys, p_y_next_given_x, A, C)
        fps.append(filtering_posterior)
    return fps, ys

def compute_filtering_posteriors(table, num_obs, dim, ys=None):
    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    mu_0 = table[dim]['mu_0']
    Q_0 = table[dim]['Q_0']

    if ys is None:
        ys = generate_trajectory(num_obs, A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0)[0]
    env = LinearGaussianEnv(A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0, ys=ys, sample=False)

    assert(num_obs == len(ys))

    lgv = get_linear_gaussian_variables(dim=dim, num_obs=num_obs)

    # true evidence
    jvs = JointVariables(lgv.ys, A=A, C=C)
    print('true evidence: ', jvs.dist.log_prob(ys).exp())

    fps = []
    for t in range(num_obs):
        filtering_posterior = compute_filtering_posterior(t, num_obs, lgv.xs, lgv.ys, A, C)
        fps.append(filtering_posterior)
    return fps, ys

def evaluate_filtering_posterior(ys, N, tds, epsilon, env=None):
    run = wandb.init(project='linear_gaussian_model evaluation', save_code=True, entity='iai')
    if env is None:
        # create env
        if env is None:
            env = LinearGaussianEnv(A=gen_A, Q=gen_Q,
                                    C=gen_C, R=gen_R,
                                    mu_0=gen_mu_0,
                                    Q_0=gen_Q_0, ys=ys,
                                    sample=False)

    # evidence estimate
    evidence_est = torch.tensor(0.).reshape(1, -1)
    log_evidence_est = torch.tensor(0.).reshape(1, -1)
    total_rewards = []
    # collect log( p(x,y)/q(x) )
    log_p_y_over_qs = torch.zeros(N)
    # keep track of log evidence estimates up to N sample trajectories
    running_log_evidence_estimates = []
    # keep track of (log) weights p(x) / q(x)
    log_weights = []
    # get trajectory length
    n = len(ys)
    # get dimensionality
    _td = tds[0].condition(y_values=ys)
    td = dist.MultivariateNormal(_td.mean(), _td.covariance() + epsilon * torch.eye(_td.mean().shape[0]))
    d = int(td.mean.nelement())
    for i in range(N):
        # get first obs
        obs = env.reset()
        # keep track of xs
        xs = []
        # keep track of prior over actions p(x)
        log_p_x = torch.tensor(0.).reshape(1, -1)
        log_p_y_given_x = torch.tensor(0.).reshape(1, -1)
        # collect actions, likelihoods
        states = [obs]
        xts = [env.prev_xt]
        xt = td.sample()
        actions = [xt]
        prior_reward = score_initial_state(x0=xt.reshape(-1), mu_0=env.mu_0, Q_0=env.Q_0)  # the reward for the initial state
        lik_reward = score_y(y_test=ys[0], x_t=xt, C=env.C, R=env.R)  # the first likelihood reward
        priors = [prior_reward]
        log_p_x += prior_reward
        liks = [lik_reward]
        log_p_y_given_x += lik_reward
        total_reward = prior_reward + lik_reward
        log_q = td.log_prob(xt)
        for j in range(1, len(tds)):
            td_fps = tds[j]
            y = ys[j:]
            _dst = td_fps.condition(y_values=y, x_value=xt)
            dst = dist.MultivariateNormal(_dst.mean(), _dst.covariance() + epsilon * torch.eye(_dst.mean().shape[0]))
            prev_xt = xt
            xt = dst.sample()
            log_q += dst.log_prob(xt)
            xts.append(xt)
            prior_reward = score_state_transition(xt=xt.reshape(-1), prev_xt=prev_xt, A=env.A, Q=env.Q)
            lik_reward = score_y(y_test=y[0], x_t=xt, C=env.C, R=env.R)  # the first likelihood reward
            total_reward += prior_reward + lik_reward
            states.append(xt)
            priors.append(prior_reward)
            liks.append(lik_reward)
            log_p_x += prior_reward
            log_p_y_given_x += lik_reward
            xs.append(xt)
            actions.append(xt)
        # log p(x,y)
        log_p_y_x = log_p_y_given_x + log_p_x
        if N == 1:
            log_p_y_over_qs[i] = (log_p_y_x - log_q).item()
            running_log_evidence_estimates.append(log_p_y_over_qs[0])
        else:
            log_p_y_over_qs[i] = (log_p_y_x - log_q).item()
            running_log_evidence_estimates.append(torch.logsumexp(log_p_y_over_qs[0:i+1], -1) - torch.log(torch.tensor(i+1.)))
        log_weights.append(log_p_x - log_q)  # ignore these since we consider the weights to be p(y|x)p(x)/q(x)
        total_rewards.append(total_reward)
        wandb.log({'total_reward': total_reward})

    # print('filtering score:' , log_p_y_over_qs)
    # print('filtering evidence: ', log_p_y_over_qs.exp())
    # calculate variance estmate as
    # $\hat{\sigma}_{int}^2=(n(n-1))^{-1}\sum_{i=1}^n(f_iW_i-\overline{fW})^2$
    sigma_est = torch.sqrt( (logvarexp(log_p_y_over_qs) - torch.log(torch.tensor(len(log_p_y_over_qs) - 1, dtype=torch.float32))).exp() )
    return EvaluationObject(running_log_estimates=torch.tensor(running_log_evidence_estimates), sigma_est=sigma_est,
                            xts=xts, states=states, actions=actions, priors=priors, liks=liks,
                            log_weights=log_p_y_over_qs)

def test_filtering_posterior():
    dim = 1
    t = 1
    num_obs = 5
    table = create_dimension_table(dimensions=[dim], random=False)
    A = table[dim]['A']
    C = table[dim]['C']

    xs, ys, p_y_next_given_x = compute_filtering_data_structures(dim=dim, num_obs=num_obs, table=table)
    fp1 = old_compute_filtering_posterior(t=0, num_obs=num_obs, xs=xs, ys=ys, p_y_next_given_x=p_y_next_given_x, A=A, C=C)
    fp2 = old_compute_filtering_posterior(t=1, num_obs=num_obs, xs=xs, ys=ys, p_y_next_given_x=p_y_next_given_x, A=A, C=C)
    fp3 = old_compute_filtering_posterior(t=2, num_obs=num_obs, xs=xs, ys=ys, p_y_next_given_x=p_y_next_given_x, A=A, C=C)
    fp4 = old_compute_filtering_posterior(t=3, num_obs=num_obs, xs=xs, ys=ys, p_y_next_given_x=p_y_next_given_x, A=A, C=C)

    mu = fp2.denominator.mean(value=torch.ones(1))
    fp_dist = fp2.condition(mu)
    fp_dist.mean(value=torch.tensor(1.))


if __name__ == "__main__":
    dim = 1
    num_obs = 2
    table = create_dimension_table(torch.tensor([dim]), random=False)
    fps, ys = compute_filtering_posteriors(table=table, num_obs=num_obs, dim=dim)

    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    mu_0 = table[dim]['mu_0']
    Q_0 = table[dim]['Q_0']
    env = LinearGaussianEnv(A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0, ys=ys, sample=True)

    evaluate_filtering_posterior(ys=ys, N=2, tds=fps, env=env)
