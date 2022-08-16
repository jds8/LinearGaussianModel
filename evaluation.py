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
from generative_model import score_state_transition, score_initial_state, score_y
import wandb
from linear_gaussian_env import LinearGaussianEnv
from all_obs_linear_gaussian_env import AllObservationsAbstractLinearGaussianEnv
from math_utils import logvarexp, importance_sampled_confidence_interval, log_effective_sample_size, log_max_weight_proportion, log_mean
from plot_utils import legend_without_duplicate_labels
from linear_gaussian_prob_prog import MultiGaussianRandomVariable, GaussianRandomVariable, MultiLinearGaussian, LinearGaussian, VecLinearGaussian
from rl_models import load_rl_model
from dimension_table import create_dimension_table


class EvaluationObject:
    def __init__(self, running_log_estimates, sigma_est, xts, states, actions, priors, liks, log_weights):
        self.running_log_estimates = running_log_estimates
        self.sigma_est = sigma_est
        self.xts = xts
        self.states = states
        self.actions = actions
        self.priors = priors
        self.liks = liks
        self.log_weights = log_weights

        self._compute_statistics()

    def _compute_statistics(self):
        self._compute_importance_weight_diagnostics()
        self._compute_confidence_intervals()

    def _compute_confidence_intervals(self):
        self.running_ci = importance_sampled_confidence_interval(self.running_log_estimates.exp(), self.sigma_est,
                                                                 len(self.running_log_estimates), epsilon=torch.tensor(0.05))
        self.running_ess_ci = importance_sampled_confidence_interval(self.running_log_estimates.exp(), self.ess_sigma_est,
                                                                     self.log_effective_sample_size.exp(), epsilon=torch.tensor(0.05))
        self.ci = (round(self.running_ci[0][-1].item(), 4), round(self.running_ci[1][-1].item(), 4))
        self.ess_ci = (round(self.running_ess_ci[0][-1].item(), 4), round(self.running_ess_ci[1][-1].item(), 4))

    def _compute_importance_weight_diagnostics(self):
        self.log_weight_mean = log_mean(self.log_weights)
        self.log_max_weight_prop = log_max_weight_proportion(self.log_weights)
        self.log_effective_sample_size = log_effective_sample_size(self.log_weights)
        self.ess_sigma_est = self.sigma_est / self.log_weight_mean.exp()


def evaluate(ys, d, N, env):
    run = wandb.init(project='linear_gaussian_model evaluation', save_code=True, entity='iai')
    print('\nevaluating...')
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
    # evaluate N times
    for i in range(N):
        done = False
        # get first obs
        obs = env.reset()
        # keep track of xs
        xs = []
        # keep track of prior over actions p(x)
        log_p_x = torch.tensor(0.).reshape(1, -1)
        log_p_y_given_x = torch.tensor(0.).reshape(1, -1)
        # collect actions, likelihoods
        states = [obs]
        actions = []
        priors = []
        liks = []
        xts = []
        total_reward = 0.
        while not done:
            xt = d.predict(obs, deterministic=False)[0]
            xts.append(env.prev_xt)
            obs, reward, done, info = env.step(xt)
            total_reward += reward
            states.append(obs)
            priors.append(info['prior_reward'])
            liks.append(info['lik_reward'])
            log_p_x += info['prior_reward']
            log_p_y_given_x += info['lik_reward']
            actions.append(info['action'])
            xs.append(xt)
        try:
            if isinstance(xs[0], torch.Tensor):
                xs = torch.cat(xs).reshape(-1, env.traj_length)
            else:
                xs = torch.tensor(np.array(xs)).reshape(-1, env.traj_length)
        except:
            pass

        # log p(x,y)
        log_p_y_x = log_p_y_given_x + log_p_x
        log_qrobs = torch.zeros(len(env.states))
        for j in range(len(env.states)):
            state = env.states[j]
            action = actions[j]
            log_qrobs[j] = d.evaluate_actions(obs=state.t(), actions=action)[1].sum().item()

        log_q = torch.sum(log_qrobs)
        log_p_y_over_qs[i] = (log_p_y_x - log_q).item()
        running_log_evidence_estimates.append(torch.logsumexp(log_p_y_over_qs[0:i+1], -1) - torch.log(torch.tensor(i+1.)))
        log_weights.append(log_p_x - log_q)  # ignore these since we consider the weights to be p(y|x)p(x)/q(x)
        total_rewards.append(total_reward)
        wandb.log({'total_reward': total_reward})

    # calculate variance estmate as
    # $\hat{\sigma}_{int}^2=(n(n-1))^{-1}\sum_{i=1}^n(f_iW_i-\overline{fW})^2$
    sigma_est = torch.sqrt( (logvarexp(log_p_y_over_qs) - torch.log(torch.tensor(len(log_p_y_over_qs) - 1, dtype=torch.float32))).exp() )
    return EvaluationObject(running_log_estimates=torch.tensor(running_log_evidence_estimates), sigma_est=sigma_est,
                            xts=xts, states=states, actions=actions, priors=priors, liks=liks,
                            log_weights=log_p_y_over_qs)

def evaluate_until(d, truth, env, epsilon, max_samples=100000):
    run = wandb.init(project='linear_gaussian_model evaluation', save_code=True, entity='iai')
    print('\nevaluating...')

    # evidence estimate
    evidence_est = torch.tensor(0.).reshape(1, -1)
    log_evidence_est = torch.tensor(0.).reshape(1, -1)
    total_rewards = []
    # collect log( p(x,y)/q(x) )
    log_p_y_over_qs = torch.tensor([0.])
    # keep track of log evidence estimates up to N sample trajectories
    running_log_evidence_estimates = []
    # keep track of (log) weights p(x) / q(x)
    log_weights = []
    # flag when to finish
    outside_epsilon = True
    # evaluate at most max_samples times
    i = 0
    while outside_epsilon or i > max_samples:
        i += 1
        done = False
        # get first obs
        obs = env.reset()
        # keep track of xs
        xs = []
        # keep track of prior over actions p(x)
        log_p_x = torch.tensor(0.).reshape(1, -1)
        log_p_y_given_x = torch.tensor(0.).reshape(1, -1)
        # collect actions, likelihoods
        states = [obs]
        actions = []
        priors = []
        liks = []
        xts = []
        total_reward = 0.
        while not done:
            xt = d.predict(obs, deterministic=False)[0]
            xts.append(env.prev_xt)
            obs, reward, done, info = env.step(xt)
            total_reward += reward
            states.append(obs)
            priors.append(info['prior_reward'])
            liks.append(info['lik_reward'])
            log_p_x += info['prior_reward']
            log_p_y_given_x += info['lik_reward']
            actions.append(info['action'])
            xs.append(xt)
        try:
            if isinstance(xs[0], torch.Tensor):
                xs = torch.cat(xs).reshape(-1, env.traj_length)
            else:
                xs = torch.tensor(np.array(xs)).reshape(-1, env.traj_length)
        except:
            pass

        # log p(x,y)
        log_p_y_x = log_p_y_given_x + log_p_x
        log_qrobs = torch.zeros(len(env.states))
        for j in range(len(env.states)):
            state = env.states[j]
            action = actions[j]
            log_qrobs[j] = d.evaluate_actions(obs=state.t(), actions=action)[1].sum().item()

        log_q = torch.sum(log_qrobs)
        log_p_y_over_qs = torch.cat([log_p_y_over_qs, (log_p_y_x - log_q).reshape(1)])
        running_log_evidence_estimates.append(torch.logsumexp(log_p_y_over_qs[0:i+1], -1) - torch.log(torch.tensor(i+1.)))
        log_weights.append(log_p_x - log_q)  # ignore these since we consider the weights to be p(y|x)p(x)/q(x)
        total_rewards.append(total_reward)
        wandb.log({'total_reward_in_evaluate': total_reward})

        log_ratio = running_log_evidence_estimates[-1] - truth
        ratio = min(log_ratio, -log_ratio).exp()
        outside_epsilon = (ratio - 1) > epsilon

    # calculate variance estmate as
    # $\hat{\sigma}_{int}^2=(n(n-1))^{-1}\sum_{i=1}^n(f_iW_i-\overline{fW})^2$
    sigma_est = torch.sqrt( (logvarexp(log_p_y_over_qs) - torch.log(torch.tensor(len(log_p_y_over_qs) - 1, dtype=torch.float32))).exp() )
    return EvaluationObject(running_log_estimates=torch.tensor(running_log_evidence_estimates), sigma_est=sigma_est,
                            xts=xts, states=states, actions=actions, priors=priors, liks=liks,
                            log_weights=log_p_y_over_qs)

def evaluate_filtering_posterior(ys, N, tds, epsilon, env):
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

    # print('filtering score:' , log_p_y_over_qs)
    # print('filtering evidence: ', log_p_y_over_qs.exp())
    # calculate variance estmate as
    # $\hat{\sigma}_{int}^2=(n(n-1))^{-1}\sum_{i=1}^n(f_iW_i-\overline{fW})^2$
    sigma_est = torch.sqrt( (logvarexp(log_p_y_over_qs) - torch.log(torch.tensor(len(log_p_y_over_qs) - 1, dtype=torch.float32))).exp() )
    return EvaluationObject(running_log_estimates=torch.tensor(running_log_evidence_estimates), sigma_est=sigma_est,
                            xts=xts, states=states, actions=actions, priors=priors, liks=liks,
                            log_weights=log_p_y_over_qs)

def evaluate_agent_until(posterior_evidence, linear_gaussian_env_type, using_entropy_loss, traj_length, dim, model_name, epsilon=0.5):
    _, policy = load_rl_model(model_name, device='cpu')

    table = create_dimension_table(torch.tensor([dim]), random=False)
    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    mu_0 = table[dim]['mu_0']
    Q_0 = table[dim]['Q_0']

    env = linear_gaussian_env_type(A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0, ys=posterior_evidence.ys,
                                   using_entropy_loss=using_entropy_loss, sample=False)

    return evaluate_until(d=policy, truth=posterior_evidence.evidence, env=env, epsilon=epsilon)
