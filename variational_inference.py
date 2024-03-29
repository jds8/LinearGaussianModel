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
        self.scale = self.distribution.scale

    def rsample(self):
        return self.distribution.rsample()

    def log_prob(self, xt):
        return self.distribution.log_prob(xt)


class Policy:
    def __init__(self, y_embedding, x_embedding, model, args):
        self.y_embedding = y_embedding
        self.x_embedding = x_embedding
        self.model = model
        self.args = args

    def get_model_input_from_obs(self, obs):
        raise NotImplementedError

    def get_distribution(self, obs):
        inpt = self.get_model_input_from_obs(obs)
        mean_output, log_std_output = self.model(inpt).squeeze()
        std_output = log_std_output.exp().reshape(1, 1) + 1e-8
        try:
            return Distribution(Distribution(dist.Normal(mean_output, std_output)))
        except:
            import pdb; pdb.set_trace()
            inpt = self.get_model_input_from_obs(obs)


    def predict(self, obs, deterministic=False):
        d = self.get_distribution(obs).distribution
        xt = d.mean if deterministic else d.rsample()
        return xt, d.log_prob(xt)

    def evaluate_actions(self, obs, actions):
        d = self.get_distribution(obs).distribution
        return actions, d.log_prob(actions)


class RNNPolicy(Policy):
    def __init__(self, rnn, y_embedding, x_embedding, model, args):
        super().__init__(y_embedding, x_embedding, model, args)
        self.rnn = rnn

    def get_model_input_from_obs(self, obs):
        """
        This assumes that obs contains [prev_xt, future_ys], i.e.
        that *only* future ys are passed in obs, *not all* obs.
        This is the case for the EnsembleLinearGaussianEnv
        """
        _obs = obs.squeeze()
        prev_xt = _obs[:self.args.dim]
        future_ys = _obs[self.args.dim:].reshape(-1, 1)

        # normalize and embed ys
        normalized_ys = normalize(future_ys, mean=future_ys.mean(), std=self.args.R)
        embedded_ys = self.y_embedding(normalized_ys).unsqueeze(0)

        # invoke rnn on ys
        obs_output, _ = self.rnn(embedded_ys)
        b = obs_output[0, -1, :]

        # normalize and embed prev_xt
        normalized_xt = normalize(prev_xt, mean=torch.zeros_like(prev_xt), std=self.args.Q)
        embedded_prev_xt = self.x_embedding(normalized_xt).squeeze()

        # return concatenation of embedded_x and rnn output
        return torch.cat([embedded_prev_xt, b], dim=0)


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

def kl_divergence(p, q):
    return q.scale.log() - p.scale.log() + (p.scale / q.scale).abs().pow(2)/2 + ((p.loc - q.loc) / q.scale).abs().pow(2)/2 - 0.5

def normalize(ys, mean, std):
    return (ys - mean) / std


class VariationalLGM:
    def __init__(self, args):
        torch.autograd.set_detect_anomaly(True)
        self.args = args
        self.name = type(self).__name__

        # embeds ys into larger space
        self.y_embedding = nn.Linear(1, args.embedding_dim)
        self.x_embedding = nn.Linear(args.dim, args.embedding_dim)

        # takes each observation as input
        self.model = nn.Sequential(
            nn.Linear(args.model_dim, args.model_dim),
            nn.LeakyReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(args.hidden_dim, 2),
        )
        self.params = list(self.model.parameters()) + list(self.y_embedding.parameters()) + list(self.x_embedding.parameters())

        var_dir = 'variational_inference'
        # self.run_dir = '{}/{}/m={}'.format(var_dir, self.name, self.args.m)
        self.run_dir = '/home/jsefas/linear-gaussian-model/from_borg/Sep-17-2022/m=5'
        # self.run_dir = '{}/m=0'.format(var_dir)
        self.model_state_dict_path = '{}/model_state_dict_traj_length_{}'.format(self.run_dir, self.args.traj_length)
        os.makedirs(self.run_dir, exist_ok=True)

    def initialize_optimizer(self):
        self.optimizer = torch.optim.Adam(self.params, lr=self.args.learning_rate)

    def get_model_input(self, prev_xt, obs_output, state_idx):
        raise NotImplementedError

    def get_model_input_from_obs(self, obs):
        raise NotImplementedError

    def infer(self):
        num_epochs = self.args.num_epochs
        traj_length = self.args.traj_length

        print_interval = num_epochs / 50
        for epoch in range(num_epochs):
            loss = torch.tensor([0.])
            for _ in range(self.args.num_samples):
                # sample new set of observations
                ys, xs, _, _ = generate_trajectory(traj_length, A=self.args.A, Q=self.args.Q, C=self.args.C,
                                                   R=self.args.R, mu_0=self.args.mu_0, Q_0=self.args.Q_0)

                # normalized_ys = torch.stack([normalize(y, mean=self.args.C*x, std=self.args.R) for x, y in zip(xs, ys)], dim=0)
                # embedded_ys = self.y_embedding(normalized_ys)

                prev_xt = torch.zeros(self.args.dim)

                # get current policy
                policy = self.extract_policy()

                obses = []
                prev_xts = [prev_xt]
                kls = []
                for state_idx in range(traj_length):
                    double_prev_xt = prev_xts[-2] if len(prev_xts) > 1 else 0.
                    # normalized_xt = normalize(prev_xt, mean=self.args.A*double_prev_xt, std=self.args.Q)

                    # embedded_prev_xt = self.x_embedding(prev_xt)

                    # obs = torch.cat([embedded_prev_xt.reshape(-1, self.args.embedding_dim), embedded_ys[state_idx:state_idx+self.args.condition_length].reshape(-1, self.args.embedding_dim)], dim=0)
                    obs = torch.cat([prev_xt.reshape(-1, 1), ys[state_idx:state_idx+self.args.condition_length].reshape(-1, 1)], dim=0)

                    obses.append(obs)

                    # compute kl divergence with prior
                    q_dist = policy.get_distribution(obs).distribution.distribution
                    mvn_p_dist = get_state_transition_dist(prev_xt, self.args.A, self.args.Q)
                    p_dist = dist.Normal(mvn_p_dist.mean.squeeze(), torch.sqrt(mvn_p_dist.covariance_matrix).squeeze())
                    kl = kl_divergence(q_dist, p_dist).squeeze()
                    kls.append(kl)
                    loss += kl
                    # loss += dist.kl_divergence(q_dist, p_dist).squeeze()

                    # sample new xt
                    prev_xt = q_dist.rsample()
                    prev_xts.append(prev_xt)

                    # score likelihood
                    lik = score_y(ys[state_idx], prev_xt, self.args.C, self.args.R)
                    # compute total loss
                    loss -= lik

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

    def clip_gradients(self):
        nn.utils.clip_grad_norm_(self.model.parameters(), 1)

    def save_models(self):
        torch.save(self.model.state_dict(), self.model_state_dict_path)

    def load_models(self):
        try:
            self.model.load_state_dict(torch.load(self.model_state_dict_path))
        except:
            import pdb; pdb.set_trace()
            self.model.load_state_dict(torch.load(self.model_state_dict_path))


    def extract_policy(self):
        return RNNPolicy(self.rnn, self.y_embedding, self.x_embedding, self.model, self.args)


class RNNWrapper(nn.RNN):
    def forward(self, inpt, hx=None):
        return super().forward(reversed(inpt), hx)


class RecurrentVariationalLGM(VariationalLGM):
    def __init__(self, args):
        args.model_dim = args.dim*args.embedding_dim + args.obs_size

        super().__init__(args)

        self.rnn = RNNWrapper(args.embedding_dim, args.obs_size, args.num_rnn_layers)
        self.params += list(self.rnn.parameters())
        self.initialize_optimizer()
        self.rnn_state_dict_path = '{}/rnn_state_dict_traj_length_{}'.format(self.run_dir, self.args.traj_length)
        os.makedirs(self.run_dir, exist_ok=True)

    def get_obs_output(self, ys, state_idx):
        future_ys = ys[state_idx:state_idx+self.args.condition_length].reshape(-1, 1, 1)
        obs_output, _ = self.rnn(future_ys)
        return obs_output[-1, :, :]

    def get_model_input(self, prev_xt, ys, state_idx):
        obs_output = self.get_obs_output(ys, state_idx)
        return torch.cat([prev_xt.reshape(1, -1), obs_output.reshape(1, -1)], dim=1)

    def get_model_input_from_obs(self, obs):
        """
        This assumes that obs contains [prev_xt, future_ys], i.e.
        that *only* future ys are passed in obs, *not all* obs.
        This is the case for the EnsembleLinearGaussianEnv
        """
        _obs = obs.squeeze()
        prev_xt = _obs[:self.args.dim]
        future_ys = _obs[self.args.dim:].reshape(-1, 1, 1)
        # we can ignore the ys before state_idx since they will be zeroed out anyway
        ys = torch.zeros(self.args.traj_length)
        ys[state_idx:] = future_ys
        return self.get_model_input(prev_xt, ys, state_idx)

    # def get_model_input_from_obs(self, obs):
    #     """
    #     This assumes that obs contains [prev_xt, future_ys], i.e.
    #     that *only* future ys are passed in obs, *not all* obs.
    #     This is the case for the EnsembleLinearGaussianEnv
    #     """
    #     _obs = obs.squeeze()
    #     try:
    #         embedded_prev_xt = _obs[:self.args.dim, :].squeeze()
    #     except:
    #         import pdb; pdb.set_trace()
    #         embedded_prev_xt = _obs[:self.args.dim, :].squeeze()

    #     embedded_future_ys = _obs[self.args.dim:, :]
    #     embedded_future_ys = embedded_future_ys.reshape(1, *embedded_future_ys.shape)
    #     obs_output, _ = self.rnn(embedded_future_ys)
    #     b = obs_output[0, -1, :]
    #     return torch.cat([embedded_prev_xt, b])

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
        assert args.condition_length == args.traj_length
        args.model_dim = args.traj_length * (args.dim + 1)

        super().__init__(args)

        self.initialize_optimizer()

        # self.softplus = nn.Softplus()
        # self.params += list(self.softplus.parameters())
        # self.softplus_state_dict_path = '{}/{}/soft_plus_state_dict_traj_length_{}'.format(self.var_dir, type(self).__name__, self.args.traj_length)

    def get_obs_output(self, ys, state_idx):
        mask = torch.zeros(self.args.traj_length)
        future_ys = mask.mul(ys)
        return future_ys.reshape(-1, 1, 1)

    def get_model_input(self, prev_xt, ys, state_idx):
        obs_output = self.get_obs_output(ys, state_idx)
        latents = torch.zeros(self.args.traj_length * self.args.dim)

        if state_idx > 0:
            prev_state_idx = state_idx-1
            latents[prev_state_idx:prev_state_idx + self.args.dim] = prev_xt

        return torch.cat([latents.reshape(1, -1), obs_output.reshape(1, -1)], dim=1)

    def get_model_input_from_obs(self, obs):
        """
        This assumes that obs contains [prev_xt, future_ys], i.e.
        that *only* future ys are passed in obs, *not all* obs.
        This is the case for the EnsembleLinearGaussianEnv
        """
        _obs = obs.squeeze()
        prev_xt = _obs[:self.args.dim]
        future_ys = _obs[self.args.dim:].reshape(-1, 1, 1)
        state_idx = self.args.traj_length - len(future_ys)
        # we can ignore the ys before state_idx since they will be zeroed out anyway
        ys = torch.zeros(self.args.traj_length)
        ys[state_idx:] = future_ys
        return self.get_model_input(prev_xt, ys, state_idx)

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

def get_vlgm(args):
    dim = args.dim

    table = create_dimension_table([dim], random=False)
    if args.Q > 0 or args.C > 0:
        table[dim]['A'] = torch.tensor(args.A).reshape(table[dim]['A'].shape)
        table[dim]['Q'] = torch.tensor(args.Q).reshape(table[dim]['Q'].shape)
        table[dim]['C'] = torch.tensor(args.C).reshape(table[dim]['C'].shape)
        table[dim]['R'] = torch.tensor(args.R).reshape(table[dim]['R'].shape)

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

    args.m = args.condition_length
    args.condition_length = args.condition_length if args.condition_length > 0 else args.traj_length

    lgm_class = get_lgm_class_type(args.lgm_type)

    return lgm_class(args)


if __name__ == "__main__":
    # print(torch.manual_seed(10908954672825061901))
    args, _ = get_args()

    vlgm = get_vlgm(args)

    # vlgm.load_models()
    vlgm.infer()
    # vlgm.load_models()

    # policy = vlgm.extract_policy()
    # ys, xs = generate_trajectory(args.traj_length, A=args.A, Q=args.Q, C=args.C, R=args.R, mu_0=args.mu_0, Q_0=args.Q_0)[0:2]
    # obs = torch.cat([torch.zeros(args.dim), ys])
    # policy.get_distribution(obs)

    # rollout_policy(policy, ys)
