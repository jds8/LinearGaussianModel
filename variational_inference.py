#!/usr/bin/env python3
import os
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
from math_utils import band_matrix
from dimension_table import create_dimension_table
from generative_model import y_dist, sample_y, generate_trajectory, score_state_transition, score_y, get_state_transition_dist
from get_args import get_args
from all_obs_linear_gaussian_env import ConditionalObservationsLinearGaussianEnv
from evaluation import evaluate


# https://jaketae.github.io/study/pytorch-rnn/
class RNN(nn.Module):
# model = RNN(args.input_size, args.hidden_size, 2)
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2output = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state), 1)
        hidden = torch.relu(self.in2hidden(combined))
        output = self.in2output(combined)
        return output, hidden

    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))


class Distribution:
    def __init__(self, distribution):
        self.distribution = distribution
        self.mean = self.distribution.mean
        self.scale = self.distribution.scale**2

    def rsample(self):
        return self.distribution.rsample()

    def log_prob(self, xt):
        return self.distribution.log_prob(xt)


class Policy:
    def __init__(self, get_obs_output, model, x_dim):
        self.get_obs_output = get_obs_output
        self.model = model
        self.x_dim = x_dim

    def get_distribution(self, obs):
        obs = obs.squeeze()
        prev_xt = obs[:self.x_dim]
        future_ys = obs[self.x_dim:]
        obs_output = self.get_obs_output(future_ys)
        inpt = torch.cat([prev_xt.reshape(1, -1), obs_output.reshape(1, -1)], dim=1)
        mean_output, log_std_output = self.model(inpt).squeeze()
        std_output = log_std_output.exp().reshape(1, 1).clamp(min=1e-10, max=1e10)
        return Distribution(Distribution(dist.Normal(mean_output, std_output)))

    def predict(self, obs, deterministic=False):
        d = self.get_distribution(obs).distribution
        xt = d.mean if deterministic else d.rsample()
        return xt, d.log_prob(xt)

    def evaluate_actions(self, obs, actions):
        d = self.get_distribution(obs).distribution
        return actions, d.log_prob(actions)


def rollout_policy(policy, ys):
    prev_xt = torch.zeros(1)
    xs = [prev_xt]
    scores = []
    for i in range(len(ys)):
        future_ys = ys[i:]
        obs = torch.cat([prev_xt.reshape(1, -1), future_ys.reshape(1, -1)], dim=1)
        prev_xt, score = policy.predict(obs)
        xs.append(prev_xt)
        scores.append(score)
        dd = policy.get_distribution(obs).distribution.distribution
    return xs, scores


class VariationalLGM:
    def __init__(self, args):
        self.args = args

        # takes each observation as input
        self.model = nn.Linear(args.dim + args.obs_size, 2)

        self.var_dir = 'variational_inference'
        self.model_state_dict_path = '{}/{}/model_state_dict_traj_length_{}'.format(self.var_dir, type(self).__name__, self.args.traj_length)
        os.makedirs(self.var_dir, exist_ok=True)

        self.params = list(self.model.parameters())

    def initialize_optimizer(self):
        self.optimizer = torch.optim.Adam(self.params, lr=args.learning_rate)

    def get_obs_output(self, ys, state_idx):
        raise NotImplementedError

    def infer(self):
        num_epochs = self.args.num_epochs
        traj_length = self.args.traj_length

        print_interval = num_epochs / 50
        for epoch in range(num_epochs):
            loss = torch.tensor([0.])
            for _ in range(self.args.num_samples):
                ys, _, _, _ = generate_trajectory(traj_length, A=self.args.A, Q=self.args.Q, C=self.args.C,
                                                  R=self.args.R, mu_0=self.args.mu_0, Q_0=self.args.Q_0)
                prev_xt = torch.zeros(self.args.dim)
                policy = self.extract_policy()
                for state_idx in range(traj_length):
                    obs_output = self.get_obs_output(ys, state_idx)
                    inpt = torch.cat([prev_xt.reshape(1, -1), obs_output.reshape(1, -1)], dim=1)
                    q_dist = policy.get_distribution(inpt).distribution.distribution
                    mean_output, log_std_output = self.model(inpt).squeeze()
                    # std_output = self.std_model(torch.tensor([[state_idx/traj_length]], dtype=torch.float))
                    # hidden_state = model.init_hidden()
                    # for y in future_ys:
                    #     output, hidden_state = model(y, hidden_state)
                    mvn_p_dist = get_state_transition_dist(prev_xt, A, Q)
                    p_dist = dist.Normal(mvn_p_dist.mean.squeeze(), torch.sqrt(mvn_p_dist.covariance_matrix).squeeze())
                    loss += dist.kl_divergence(q_dist, p_dist).squeeze()
                    state_loss, prev_xt = self.compute_elbo(mean_output, log_std_output, prev_xt, ys[state_idx])
                    loss -= state_loss
            loss /= self.args.num_samples
            self.optimizer.zero_grad()
            loss.backward()
            self.clip_gradients()
            self.optimizer.step()

            if (epoch + 1) % print_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Loss: {loss.item():.4f}"
                )
        try:
            self.save_models()
        except:
            pass

    def compute_elbo(self, mean, log_std, prev_xt, y_test):
        mean = mean.reshape(1, 1)
        std = log_std.exp().reshape(1, 1)
        xt = dist.Normal(mean, std).rsample().clamp(min=1e-10, max=1e10)
        lik = score_y(y_test, xt, self.args.C, self.args.R)
        return lik, xt

    def clip_gradients(self):
        nn.utils.clip_grad_norm_(self.model.parameters(), 1)

    def save_models(self):
        torch.save(self.model.state_dict(), self.model_state_dict_path)

    def load_models(self):
        self.model.load_state_dict(torch.load(self.model_state_dict_path))

    def extract_policy(self):
        return Policy(lambda output: self.get_obs_output(output, 0), self.model, self.args.dim)


class RecurrentVariationalLGM(VariationalLGM):
    def __init__(self, args):
        super().__init__(args)

        self.rnn = nn.RNN(1, args.obs_size, args.num_rnn_layers)
        self.params += list(self.rnn.parameters())
        self.initialize_optimizer()
        self.rnn_state_dict_path = '{}/{}/rnn_state_dict_traj_length_{}'.format(self.var_dir, type(self).__name__, self.args.traj_length)

    def get_obs_output(self, ys, state_idx):
        future_ys = ys[state_idx:].reshape(-1, 1, 1)
        obs_output, _ = self.rnn(future_ys)
        return obs_output[-1, :, :]

    def clip_gradients(self):
        super().clip_gradients()
        nn.utils.clip_grad_norm_(self.rnn.parameters(), 1)

    def save_models(self):
        super().save_models()
        torch.save(self.rnn.state_dict(), self.rnn_state_dict_path)

    def load_models(self):
        super().load_models()
        self.rnn.load_state_dict(torch.load(self.rnn_state_dict_path))


class LinearVariationalLGM(VariationalLGM):
    """ Passes the observations directly to the model without encoding """
    def __init__(self, args):
        args.obs_size = args.traj_length
        super().__init__(args)
        self.initialize_optimizer()

        # self.softplus = nn.Softplus()
        # self.params += list(self.softplus.parameters())
        # self.softplus_state_dict_path = '{}/{}/soft_plus_state_dict_traj_length_{}'.format(self.var_dir, type(self).__name__, self.args.traj_length)

    def get_obs_output(self, ys, state_idx):
        mask = torch.ones_like(ys)
        mask[:state_idx] = 0.
        future_ys = mask.mul(ys)
        return future_ys.reshape(-1, 1, 1)

    # def clip_gradients(self):
    #     super().clip_gradients()
    #     nn.utils.clip_grad_norm_(self.softplus.parameters(), 1)

    # def save_models(self):
    #     super().save_models()
    #     torch.save(self.softplus.state_dict(), self.softplus_state_dict_path)

    # def load_models(self):
    #     super().load_models()
    #     self.softplus.load_state_dict(torch.load(self.softplus_state_dict_path))


def get_lgm_class_type(type_str):
    if type_str.lower() == 'recurrent':
        return RecurrentVariationalLGM
    elif type_str.lower() == 'linear':
        return LinearVariationalLGM
    else:
        raise NotImplementedError


if __name__ == "__main__":
    args, _ = get_args()

    dim = args.dim

    table = create_dimension_table([dim], random=False)
    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    mu_0 = table[dim]['mu_0']
    Q_0 = table[dim]['Q_0']

    args.table = table
    args.A = A
    args.Q = Q
    args.C = C
    args.R = R
    args.mu_0 = mu_0
    args.Q_0 = Q_0

    lgm_class = get_lgm_class_type(args.lgm_type)

    vlgm = lgm_class(args)
    # vlgm.load_models()
    vlgm.infer()
    # vlgm.load_models()

    # policy = vlgm.extract_policy()
    # ys, xs = generate_trajectory(args.traj_length, A=args.A, Q=args.Q, C=args.C, R=args.R, mu_0=args.mu_0, Q_0=args.Q_0)[0:2]
    # obs = torch.cat([torch.zeros(args.dim), ys])
    # policy.get_distribution(obs)

    # rollout_policy(policy, ys)
