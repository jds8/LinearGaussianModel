#!/usr/bin/env python3
import os
import time
import glob
from datetime import date
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from stable_baselines3 import PPO, SAC, FKL
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.fkl.policies import SACPolicy as FKLPolicy
from stable_baselines3.common.callbacks import BaseCallback
from copy import deepcopy
from generative_model import y_dist, sample_y, generate_trajectory, \
    get_state_transition_dist, \
    gen_A, gen_Q, gen_C, gen_R, gen_mu_0, gen_Q_0, \
    test_A, test_Q, test_C, test_R, test_mu_0, test_Q_0, \
    state_transition, score_state_transition, gen_covariance_matrix, \
    score_initial_state, score_y, get_stacked_state_transition_dist
    # A, Q, C, R, mu_0, Q_0, \
import wandb
from linear_gaussian_env import LinearGaussianEnv, LinearGaussianSingleYEnv
from all_obs_linear_gaussian_env import AllObservationsLinearGaussianEnv, ConditionalObservationsLinearGaussianEnv, EnsembleLinearGaussianEnv
from math_utils import logvarexp, importance_sampled_confidence_interval, log_effective_sample_size, log_max_weight_proportion, log_mean
from plot_utils import legend_without_duplicate_labels, compare_normals, generate_gif
from linear_gaussian_prob_prog import MultiGaussianRandomVariable, GaussianRandomVariable, MultiLinearGaussian, LinearGaussian, VecLinearGaussian, JointVariables, get_linear_gaussian_variables
from evaluation import EvaluationObject, evaluate, evaluate_filtering_posterior, \
    evaluate_agent_until, evaluate_rl_posterior_ensemble, \
    evaluate_smoothing_posterior_until, evaluate_pure_rl_ensemble
from dimension_table import create_dimension_table
from filtering_posterior import compute_conditional_filtering_posteriors
import pandas as pd
from get_args import get_args
from pathlib import Path
from data_loader import load_ess_data
from plot import plot_ess_data, plot_state_occupancy, plot_3d_state_occupancy
from linear_policy import LinearActorCriticPolicy
from rl_models import load_rl_model, get_rl_type_from_str, evaluate_actions
from variational_inference import VariationalLGM, get_vlgm

# model name
# MODEL = 'trial_linear_gaussian_model_(traj_{}_dim_{})'
# MODEL = 'linear_gaussian_model_(traj_{}_dim_{})'
MODEL = '/opt/agents/{}_{}_linear_gaussian_model_(traj_{}_dim_{})'
MODEL_W_CONDITION = '/opt/agents/{}_{}_linear_gaussian_model_(traj_{}_dim_{}_condition_length_{}_use_mlp_policy_{})'
MODEL_W_CONDITION_AND_RL_TYPE = '/opt/agents/{}_{}_linear_gaussian_model_(traj_{}_dim_{}_condition_length_{}_use_mlp_policy_{}_rl_type_{})'
# MODEL = 'from_borg/rl_agents/linear_gaussian_model_(traj_{}_dim_{})'
# MODEL = 'new_linear_gaussian_model_(traj_{}_dim_{})'

TODAY = date.today().strftime("%b-%d-%Y")

SAVE_DIR = '/opt/linear_gaussian_data'
RL_TIMESTEPS = 1000000
NUM_SAMPLES = 1000
NUM_VARIANCE_SAMPLES = 10
NUM_REPEATS = 20

FILTERING_POSTERIOR_DISTRIBUTION = 'filtering_posterior'
FILTERING_POSTERIOR_CONDITIONAL_DISTRIBUTION = 'filtering_posterior_conditional'
POSTERIOR_DISTRIBUTION = 'posterior'
PRIOR_DISTRIBUTION = 'prior'
RL_DISTRIBUTION = 'RL'
VI_DISTRIBUTION = 'variational_inference'

ENTROPY_LOSS = 'entropy'
FORWARD_KL = 'forward_kl'
REVERSE_KL = 'reverse_kl'

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, env, model_name, verbose=1, eval_freq=1, log_freq=100,
                    gradient_save_freq = 0, run_id=''):
        super(CustomCallback, self).__init__(verbose)
        self.env = env
        self.model_name_base = self._get_model_name_parts(model_name)
        self.eval_freq = eval_freq
        self.log_freq = log_freq
        self.eval_incr = 1
        self.best_mean_reward = -np.Inf
        self.verbose = verbose
        self.gradient_save_freq = gradient_save_freq
        self.current_mod = 1

    @staticmethod
    def _get_model_name_parts(model_name):
        parts = model_name.split('.zip')
        return parts[0]

    def _init_callback(self) -> None:
        d = {}
        if "algo" not in d:
            d["algo"] = type(self.model).__name__
        for key in self.model.__dict__:
            if key in wandb.config:
                continue
            if type(self.model.__dict__[key]) in [float, int, str]:
                d[key] = self.model.__dict__[key]
            else:
                d[key] = str(self.model.__dict__[key])
        if self.gradient_save_freq > 0:
            pass
            wandb.watch(self.model.policy, log_freq=self.gradient_save_freq, log="all")
        wandb.config.setdefaults(d)

    def eval_policy(self, policy, env):
        #
        total_reward, horizon, avg_wp = 0, 0, 0.
        #
        def eval_policy_helper(policy,env):
            obs = env.reset()
            done = False
            total_reward, horizon = 0, 0
            prior_kls = []
            posterior_kls = []
            while not done:
                obs_env = env.envs[0].env
                prev_xt = torch.tensor(obs.squeeze()[0:env.action_space.shape[0]]).to(dtype=obs_env.A.dtype)
                prior_kls.append(dispatch_kl_with_prior(policy=self.model.policy,
                                                        prev_xt=prev_xt, ys=obs_env.get_condition_ys(),
                                                        condition_length=obs_env.condition_length,
                                                        A=obs_env.A, Q=obs_env.Q, sac_num_samples=1000))
                # td = obs_env.tds[obs_env.index]
                # posterior_kls.append(compute_conditional_kl(td_fps=td, policy=self.model.policy,
                #                             prev_xt=prev_xt, ys=obs_env.get_condition_ys(),
                #                             condition_length=obs_env.condition_length))

                action = policy(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                horizon += 1
            if isinstance(info, list):
                info = info[0]
            return total_reward, horizon, info, torch.tensor(prior_kls), torch.tensor(posterior_kls)

        next_total_reward, next_horizon, wp, prior_kls, posterior_kls = eval_policy_helper(policy, env)
        total_reward += next_total_reward
        horizon += next_horizon
        # avg_wp += wp

        avg_prior_kl = prior_kls.mean()
        avg_posterior_kl = posterior_kls.mean()
        return total_reward, horizon, avg_prior_kl, avg_posterior_kl#, avg_wp

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        # Evaluate the model
        policy = lambda obs_: self.model.predict(obs_, deterministic=True)[0]
        avg_return, avg_horizon, avg_prior_kl, avg_posterior_kl = self.eval_policy(policy, self.training_env)
        self.training_env.reset()
        # log to wandb
        wandb.log({'det_avg_return':avg_return,
                   'det_avg_horizon':avg_horizon,
                   'time_steps': self.num_timesteps,
                   'updates': self.model._n_updates,
                   'avg kl divergence with prior': avg_prior_kl,
                   'avg kl divergence with posterior': avg_posterior_kl,
                   })
        return None

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        # if we have hit conditions for full eval
        if int(np.floor(self.eval_incr / self.eval_freq)) >= self.current_mod:
            # if we passed increment threshold then we eval
            self.current_mod += 1
            # Evaluate the model
            policy = lambda obs_: self.model.predict(obs_, deterministic=True)[0]
            avg_return, avg_horizon, avg_prior_kl, avg_posterior_kl = self.eval_policy(policy, self.training_env)
            self.training_env.reset()
            #
            wandb.log({'det_avg_return':avg_return,
                       'det_avg_horizon':avg_horizon,
                       'avg kl divergence with prior':avg_prior_kl,
                       'avg kl divergence with posterior':avg_posterior_kl,
                       'stoch_avg_return': np.mean([val['l'] for val in self.model.ep_info_buffer]),
                       'stoch_avg_horizon': np.mean([val['r'] for val in self.model.ep_info_buffer]),
                       'time_steps': self.num_timesteps,
                       'updates': self.model._n_updates})

            # New best model, you could save the agent here
            if avg_return > self.best_mean_reward:
                self.best_mean_reward = avg_return
                self.model.save('{}_best.zip'+self.model_name_base)

        # otherwise just log stochastic info
        else:
            wandb.log({'stoch_avg_return': np.mean([val['l'] for val in self.model.ep_info_buffer]),
                       'stoch_avg_horizon': np.mean([val['r'] for val in self.model.ep_info_buffer]),
                       'time_steps': self.num_timesteps,
                       'updates': self.model._n_updates})

        policy_alignment_str = 'PolicyAlignment'
        # create_distribution_alignment_of_state_0_png(policy=self.model.policy, condition_length=self.env.condition_length,
        #                                              dim=self.env.mu_0.nelement(), ys=self.env.ys,
        #                                              save_dir=TODAY, filename=policy_alignment_str+'{}.png'.format(self.eval_incr))
        # create_distribution_alignment_gif(file_regex='{}/{}*.png'.format(TODAY, policy_alignment_str),
        #                                   save_dir=TODAY, gif_name=policy_alignment_str)

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # it looks like this is only invoked if there are more than 1000 timesteps in training
        self.eval_incr += 1

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        if self.env.states:
            policy = self.model.policy.to(self.env.states[-1].device)
            q_log_prob = evaluate_actions(policy=policy, obs=self.env.states[-1].t(), actions=self.env.actions[-1].reshape(-1, 1))
            # note that in this case the weights are log[ p(y|x)p(x)/q(x|y) ]
            wandb.log({'log weights': self.env.liks[-1] + self.env.p_log_probs[-1] - q_log_prob})
            wandb.log({'likelihood reward': self.env.liks[-1]})
            wandb.log({'prior reward': self.env.p_log_probs[-1]})
            wandb.log({'total reward': self.env.rewards[-1]})

            # compute KL between filtering posterior and policy at the last state (env.index - 1)
            td = self.env.tds[self.env.index-1]
            # kl = compute_conditional_kl(td_fps=td, policy=self.model.policy,
            #                             prev_xt=self.env.prev_xts[-2], ys=self.env.previous_condition_ys,
            #                             condition_length=self.env.condition_length)
            # wandb.log({'kl divergence with smoothing posterior at state {}'.format(self.env.index-1): kl})
            wandb.log({'reward at state {}'.format(self.env.index-1): self.env.rewards[-1]})

        return True

def create_distribution_alignment_gif(file_regex, save_dir, gif_name):
    filenames = glob.glob(file_regex)
    generate_gif(filenames, save_dir, gif_name)

def create_distribution_alignment_of_state_0_png(policy, condition_length, dim, ys, save_dir, filename):
    traj_length = len(ys)
    table = create_dimension_table(torch.tensor([dim]), random=False)

    tds = compute_conditional_filtering_posteriors(table=table, num_obs=traj_length, dim=dim, ys=ys)
    td = tds[0].condition(y_values=ys).get_dist()

    obs = torch.cat([torch.zeros(dim), ys[:condition_length]]).reshape(1, -1)
    pd = policy.get_distribution(obs).distribution

    compare_normals(td.mean.item(), td.covariance_matrix.item(), 'filtering posterior',
                    pd.mean.detach().item(), (pd.scale.detach()**2).item(), 'rl policy',
                    save_dir, filename)

def get_model_name(traj_length, dim, ent_coef, loss_type, condition_length, use_mlp_policy=False, rl_type='PPO'):
    if condition_length == 1:
        return MODEL.format(ent_coef, loss_type, traj_length, dim)+'.zip'
    elif rl_type == 'PPO':
        return MODEL_W_CONDITION.format(ent_coef, loss_type, traj_length, dim, condition_length, use_mlp_policy)+'.zip'
    else:
        return MODEL_W_CONDITION_AND_RL_TYPE.format(ent_coef, loss_type, traj_length, dim, condition_length, use_mlp_policy, rl_type)+'.zip'

def model_without_directory(model):
    return Path(model).parts[-1]

def get_loss_type(model_name):
    model = model_name.split('/')[1]
    return model.split('_')[0]

def train(traj_length, env, dim, condition_length, ent_coef=1.0,
          loss_type='forward_kl', learning_rate=3e-4, clip_range=0.2,
          continue_training=False, ignore_reward=False, use_mlp_policy=False,
          rl_type='PPO'):
    params = {}
    run = wandb.init(project='linear_gaussian_model training', save_code=True, config=params, entity='iai')

    # create policy

    # batch_obs is of shape BATCH_SIZE x (HIDDEN_DIMENSION + 1) x 1 where the middle dimension includes (all of the) ys
    # we only want the x part of the hidden_dimension, so we exclude the y part
    print('assuming that the observation ys have dimensionality 1')
    # prior = lambda batch_obs: get_stacked_state_transition_dist(batch_obs[:, 0:-traj_length-1, :], A=env.A, Q=env.Q)
    prior = lambda batch_obs: get_stacked_state_transition_dist(batch_obs[:, 0:dim, :], A=env.A, Q=env.Q)
    # prior = lambda batch_obs: get_stacked_state_transition_dist(batch_obs[:, 0:-1, :], A=env.A, Q=env.Q)

    model_name = get_model_name(traj_length=traj_length, dim=dim,
                                ent_coef=ent_coef, loss_type=loss_type,
                                condition_length=condition_length,
                                use_mlp_policy=use_mlp_policy,
                                rl_type=rl_type)

    print('training agent: {}'.format(model_name))

    if continue_training:
        model, _ = load_rl_model(model_name=model_name, device='cpu', env=env)
    elif use_mlp_policy:
        arch = [1024 for _ in range(3)]
        if rl_type == PPO:
            model = rl_type('MlpPolicy', env, ent_coef=ent_coef, device='cpu',
                            policy_kwargs=dict(net_arch=[dict(pi=arch, vf=arch)]),
                            verbose=1, loss_type=loss_type, prior=prior, ignore_reward=ignore_reward,
                            learning_rate=learning_rate, clip_range=clip_range)
        elif rl_type == FKL:
            policy_kwargs = dict(net_arch=dict(pi=arch, qf=arch))
            model = rl_type('MlpPolicy', env, device='cpu', verbose=1, policy_kwargs=policy_kwargs,
                            learning_rate=learning_rate, learning_starts=0,
                            num_particles=20)
        elif rl_type == SAC:
            # set the action space to [-1, 1] and rescale later
            env.action_space.low[0] = -1.
            env.action_space.high[0] = 1.
            model = rl_type('MlpPolicy', env, device='cpu',
                            policy_kwargs=dict(net_arch=dict(pi=arch, qf=arch)),
                            verbose=1, learning_rate=learning_rate)
    else:
        model = rl_type(LinearActorCriticPolicy, env, ent_coef=ent_coef, device='cpu',
                        verbose=1, loss_type=loss_type, prior=prior, ignore_reward=ignore_reward,
                        learning_rate=learning_rate, clip_range=clip_range)

    # train policy
    model.learn(total_timesteps=RL_TIMESTEPS, callback=CustomCallback(env, model_name, verbose=1))

    # save model
    model.save(model_name)
    wandb.save(model_name)


class ProposalDist:
    def __init__(self, A, Q):
        if isinstance(A, torch.Tensor):
            self.A = A
        else:
            self.A = torch.tensor(A).reshape(1, -1)
        self.prev_xt_shape = self.A.shape[0]
        if isinstance(Q, torch.Tensor):
            self.Q = Q
        else:
            self.Q = torch.tensor(Q).reshape(1, -1)

    def evaluate_actions(self, obs, actions):
        """
        Return None and score as tuple.
        The only reason we return a tuple is to match what
        the PPO policy returns on `evaluate_actions`
        """

        # We can ignore the "y" part of obs and make
        # prev_xt's shape match that of self.A
        prev_xt = obs.reshape(-1, 1)[0:self.prev_xt_shape, :]

        return (None, score_state_transition(actions, prev_xt, self.A, self.Q))

    def predict(self, obs, deterministic=False):
        # We can ignore the "y" part of obs
        prev_xt = obs.reshape(-1, 1)[0:self.prev_xt_shape, :]
        return (state_transition(prev_xt, self.A, self.Q), None)


def importance_estimate(ys, A, Q, C, R, mu_0, Q_0, N, env=None, sample=False, traj_length=0):
    if ys is not None:
        traj_length = len(ys)
    print('\nimportance estimate\n')
    pd = ProposalDist(A=A, Q=Q)

    # create env
    if env is None:
        env = LinearGaussianEnv(A=A, Q=Q,
                                C=C, R=R,
                                mu_0=mu_0,
                                Q_0=Q_0, ys=ys,
                                traj_length=traj_length,
                                sample=sample)

    return evaluate(pd, N, env)

def test_importance_sampler(traj_length, A, Q):
    """
    Generate trajectories with respect to the gen_A, gen_Q params
    and estimate the evidence using IS with input params A, Q
    """
    ys, xs, priors, liks = generate_trajectory(traj_length, gen_A, gen_Q, gen_C, gen_R, gen_mu_0, gen_Q_0)
    states = torch.cat([torch.zeros(1, 1), xs[0:-1]])
    print('\ntesting importance sampler...\n')
    for s, x, y, prior, lik in zip(states, xs, ys, liks, priors):
        print('s_t = {} a_t = {} y_t = {} where p(a_t|s_t) = N({}, {}) = {} and p(y_t|a_t) = N({}, {}) = {}'.format(s.item(), x.item(), y.item(), (A*s).item(), Q.item(), prior, (C*x).item(), R.item(), lik))
    print('ys: {}'.format(ys))

    return importance_estimate(ys, A=A, Q=Q, C=gen_C, R=gen_R, mu_0=gen_mu_0, Q_0=gen_Q_0, N=NUM_SAMPLES)

def rl_estimate(ys, dim, N, model_name, env=None, traj_length=0, device='cpu'):
    if ys is not None:
        traj_length = len(ys)
        device = ys.device
    print('\nrl_estimate\n')
    _, policy = load_rl_model(model_name, device)
    return evaluate(d=policy, N=N, env=env)


class Estimator:
    def __init__(self, running_log_estimates, ci,
                 weight_mean, max_weight_prop, ess, ess_ci, label='RL'):
        self.running_log_estimate_repeats = [running_log_estimates]
        self.ci = ci
        self.weight_mean = weight_mean
        self.max_weight_prop = max_weight_prop
        self.ess = [ess]
        self.ess_ci = ess_ci
        self.label = self._correct_label(label)
        # self.distribution_type = self._get_distribution_type()
        self.save_dir = self.create_save_dir()

    @staticmethod
    def _correct_label(label):
        return Path(label).stem

    def create_save_dir(self):
        # save_dir = '{}/{}'.format(TODAY, self.distribution_type)
        save_dir = '{}/{}'.format(TODAY, self.label)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def add_repeat(self, running_log_estimates, ess):
        self.running_log_estimate_repeats.append(running_log_estimates)
        self.ess.append(ess)

    def plot(self):
        # plot prob estimates
        x_vals = torch.arange(1, self.running_log_estimate_repeats[0].squeeze().nelement()+1)
        lower_ci, med, upper_ci = torch.quantile(torch.stack(self.running_log_estimate_repeats), torch.tensor([0.05, 0.5, 0.95]), dim=0)
        plt.plot(x_vals, med.squeeze(), label=self.label)
        plt.fill_between(x_vals, y1=lower_ci, y2=upper_ci, alpha=0.3)

    def plot_running_log_estimates(self):
        self._plot_nice(xlen=self.running_log_estimate_repeats[0].squeeze().nelement(),
                        data=torch.stack(self.running_log_estimate_repeats),
                        quantiles=torch.tensor([0.05, 0.5, 0.95]))
    def plot_ess(self):
        self._plot_nice(xlen=self.ess[0].squeeze().nelement(),
                        data=torch.stack(self.ess),
                        quantiles=torch.tensor([0.05, 0.5, 0.95]))

    def _plot_nice(self, xlen, data, quantiles):
        assert quantiles.nelement() == 3

        # plot prob estimates
        x_vals = torch.arange(1, xlen+1)
        lower_ci, med, upper_ci = torch.quantile(data, quantiles, dim=0)
        plt.plot(x_vals, med.squeeze(), label=self.label)
        plt.fill_between(x_vals, y1=lower_ci, y2=upper_ci, alpha=0.3)

    def compute_evidence_estimate(self, quantile=torch.tensor([0.5])):
        data = torch.stack(self.running_log_estimate_repeats)
        estimate = torch.quantile(data, quantile, dim=0).squeeze()
        return estimate[-1]

    def get_error_in_estimate(self, true, quantile=torch.tensor([0.5])):
        estimate = self.compute_evidence_estimate(quantile)
        return torch.abs((estimate.log() - true.log()).exp() - 1)

    # def _get_distribution_type(self):
    #     if RL_DISTRIBUTION in self.label:
    #         return RL_DISTRIBUTION
    #     elif FILTERING_POSTERIOR_DISTRIBUTION in self.label:
    #         return FILTERING_POSTERIOR_DISTRIBUTION
    #     elif POSTERIOR_DISTRIBUTION in self.label:
    #         return POSTERIOR_DISTRIBUTION
    #     elif PRIOR_DISTRIBUTION in self.label:
    #         return PRIOR_DISTRIBUTION
    #     else:
    #         return self.label

    # def stack_with_padding(self, array_of_tensors):
    #     largest_len = max(len(a) for a in array_of_tensors)
    #     torch.nn.functional.pad(input=a, pad=(0,1), mode='constant', value=0)

    def save_data(self):
        """
        Saves the running log estimate and ess into their own csv files
        where there are NUM_SAMPLES number of columns and NUM_REPEATS
        number of rows
        """
        try:
            estimates_df = pd.DataFrame(torch.stack(self.running_log_estimate_repeats).numpy())
            estimates_df.to_csv('{}/{}_LogEstimates.csv'.format(SAVE_DIR, self.label))

            ess_df = pd.DataFrame(torch.stack(self.ess).numpy())
            ess_df.to_csv('{}/{}_ESS.csv'.format(SAVE_DIR, self.label))
        except:
            pass


class ISEstimator(Estimator):
    def __init__(self, A, Q, running_log_estimates, ci,
                 weight_mean, max_weight_prop, ess, ess_ci, label=''):
        label = label if label else "_A {}\n_Q {}".format(A, Q)
        super().__init__(running_log_estimates, ci, weight_mean, max_weight_prop, ess, ess_ci, label=label)
        self.A = A
        self.Q = Q
        self.weight_mean = weight_mean
        self.max_weight_prop = max_weight_prop
        self.ess = [ess]


class ImportanceOutput(dict):
    def __init__(self, traj_length, ys, dim):
        self.traj_length = traj_length
        self.ys = ys
        self.is_estimators = {}
        self.rl_estimators = {}
        self.dimension = dim

    def add_is_estimator(self, A, Q, running_log_estimates, ci,
                         weight_mean, max_weight_prop, ess,
                         ess_ci, idstr):
        if idstr in self.is_estimators:
            self.is_estimators[idstr].add_repeat(running_log_estimates, ess)
        else:
            self.is_estimators[idstr] = ISEstimator(A, Q, running_log_estimates,
                                                    ci, weight_mean, max_weight_prop, ess, ess_ci, label=idstr)
            self[idstr] = self.is_estimators[idstr]
        return self.is_estimators[idstr]

    def add_rl_estimator(self, running_log_estimates, ci, weight_mean, max_weight_prop, ess, ess_ci, idstr):
        if idstr in self.rl_estimators:
            self.rl_estimators[idstr].add_repeat(running_log_estimates, ess)
        else:
            self.rl_estimators[idstr] = Estimator(running_log_estimates, ci, weight_mean, max_weight_prop, ess, ess_ci, label=idstr)
            self[idstr] = self.rl_estimators[idstr]
        return self.rl_estimators[idstr]

    def set_truth(self, truth):
        self.truth = truth

    def display_self(self):
        for key, val in self.is_estimators.items():
            print('A {}, Q {}: weight_mean: {}'.format(val.A, val.Q, val.weight_mean))
            print('A {}, Q {}: max_weight_prop: {}'.format(val.A, val.Q, val.max_weight_prop))
            print('A {}, Q {}: ess: {}\n'.format(val.A, val.Q, val.ess))
        for key, val in self.rl_estimators.items():
            print('RL: weight_mean: {}'.format(val.weight_mean))
            print('RL: max_weight_prop: {}'.format(val.max_weight_prop))
            print('RL: ess: {}\n'.format(val.ess))

    def plot(self, idstr=None):
        if idstr is None:
            idstr = list(self.is_estimators.keys()) + list(self.rl_estimators.keys())
        elif not isinstance(idstr, list):
            idstr = [idstr]
        for key in idstr:
            x_vals = torch.arange(1, len(self[key].running_log_estimates.squeeze())+1)
            plt.plot(x_vals, self[key].running_log_estimates.exp().squeeze(), label=key)
            plt.fill_between(x_vals, y1=self[key].running_ess_ci[0], y2=self[key].running_ci[1], alpha=0.3)

def make_ess_versus_dimension_plot(outputs, num_samples):
    # plt.gca().set_yscale('log')
    traj_length = None
    dims = []
    ess = {}
    for output in outputs:
        # Ensure all outputs have same trajectory length
        if traj_length is None:
            traj_length = output.traj_length
        else:
            assert(traj_length == output.traj_length)

        dims.append(output.dimension)

        for key, val in output.is_estimators.items():
            ess[output.dimension] = val.ess
        for key, val in output.rl_estimators.items():
            ess[output.dimension] = val.ess

    plt.scatter(dims, [torch.median(torch.tensor(e)) for k, e in ess.items()])
    for dim, (k, e) in zip(dims, ess.items()):
        lower, upper = torch.quantile(torch.tensor(e), torch.tensor([0.05, 0.95]), dim=0)
        plt.vlines(x=dim, ymin=lower.item(), ymax=upper.item())

    plt.xlabel('Dimension of X')
    plt.ylabel('Effective Sample Size')
    plt.title('Effective Sample Size Versus Hidden Dimension\n(num samples: {} trajectory length: {})'.format(num_samples, traj_length))
    legend_without_duplicate_labels(plt.gca())
    plt.savefig('{}/ess_versus_dimension_(traj_len: {}).pdf'.format(TODAY, traj_length))
    wandb.save('{}/ess_versus_dimension_(traj_len: {}).pdf'.format(TODAY, traj_length))
    plt.close()

def make_ess_plot_nice(outputs_with_names, fixed_feature_string,
                       fixed_feature, num_samples, num_repeats,
                       traj_lengths, xlabel, distribution_type, name=''):
    plot_ess_estimators_traj(outputs_with_names, traj_lengths)

    plt.xlabel(xlabel)
    plt.ylabel('Effective Sample Size')
    plt.title('Effective Sample Size Versus {}\n (num samples: {}, num repeats: {})'.format(xlabel, num_samples, num_repeats))
    legend_without_duplicate_labels(plt.gca())
    plt.savefig('{}/{}/{}_ess_plot_{}_{}.pdf'.format(TODAY, distribution_type, name, fixed_feature_string, fixed_feature))
    wandb.save('{}/{}/{}_ess_plot_{}_{}.pdf'.format(TODAY, distribution_type, name, fixed_feature_string, fixed_feature))

def make_ess_plot_nice_dim(outputs_with_names, fixed_feature_string,
                           fixed_feature, num_samples, num_repeats,
                           dims, xlabel, distribution_type, name=''):
    plot_ess_estimators_dim(outputs_with_names, dims)

    plt.xlabel(xlabel)
    plt.ylabel('Effective Sample Size')
    plt.title('Effective Sample Size Versus {}\n (num samples: {}, num repeats: {})'.format(xlabel, num_samples, num_repeats))
    legend_without_duplicate_labels(plt.gca())
    plt.savefig('{}/{}/{}_ess_plot_{}_{}.pdf'.format(TODAY, distribution_type, name, fixed_feature_string, fixed_feature))
    wandb.save('{}/{}/{}_ess_plot_{}_{}.pdf'.format(TODAY, distribution_type, name, fixed_feature_string, fixed_feature))

def make_ess_plot_nice_condition_length(outputs_with_names, fixed_feature_string,
                                        fixed_feature, num_samples, num_repeats,
                                        condition_lengths, xlabel, distribution_type, name=''):
    plot_ess_estimators_condition_length(outputs_with_names, condition_lengths)

    plt.xlabel(xlabel)
    plt.ylabel('Effective Sample Size')
    plt.title('Effective Sample Size Versus {}\n (num samples: {}, num repeats: {})'.format(xlabel, num_samples, num_repeats))
    legend_without_duplicate_labels(plt.gca())
    plt.savefig('{}/{}/{}_ess_plot_{}_{}.pdf'.format(TODAY, distribution_type, name, fixed_feature_string, fixed_feature))
    wandb.save('{}/{}/{}_ess_plot_{}_{}.pdf'.format(TODAY, distribution_type, name, fixed_feature_string, fixed_feature))


class Plotter:
    def __init__(self, name, A=None, Q=None, C=None,
                 R=None, mu_0=None, Q_0=None, dim=torch.tensor(1),
                 sample=True):
        self.name = name
        if A is None:
            try:
                self.dimension = dim.item()
            except:
                self.dimension = dim
            self.A = torch.rand(dim, dim)
            self.Q = gen_covariance_matrix(dim)
            self.C = torch.rand(1, dim)
            self.R = torch.rand(1, 1)
            self.mu_0 = torch.zeros(dim)
            self.Q_0 = self.Q
        else:
            self.A = A
            self.Q = Q
            self.C = C
            self.R = R
            self.mu_0 = mu_0
            self.Q_0 = Q_0
            self.dimension = self.A.shape[0]

        self.sample = sample

    def reset(self, table, dimension):
        try:
            dim = dimension.item()
        except:
            dim = dimension
        self.dimension = dim
        self.A = table[dim]['A']
        self.Q = table[dim]['Q']
        self.C = table[dim]['C']
        self.R = table[dim]['R']
        self.mu_0 = table[dim]['mu_0']
        self.Q_0 = table[dim]['Q_0']

    def generate_env(self, ys, traj_length):
        raise NotImplementedError

    def sample_trajectory(self, traj_length):
        raise NotImplementedError

    def get_true_score(self, ys):
        raise NotImplementedError

    def _plot_log_diffs(log_values, log_true, label):
        # compute log ratio of estimate to true value
        diffs = torch.tensor(log_values) - log_true
        plt.plot(torch.arange(1, len(diffs.squeeze())+1), diffs.squeeze(), label=label)

    def plot_IS(self, traj_lengths,
                As=torch.arange(0.2, 0.6, 0.2), Qs=torch.arange(0.2, 0.6, 0.2),
                num_samples=10000, num_repeats=10, name=''):
        if len(As) > 0:
            assert self.dimension == As[0].shape[0]
        outputs = []
        for traj_length in traj_lengths:
            plt.figure(plt.gcf().number+1)

            for _ in range(num_repeats):
                # generate ys from gen_A, gen_Q params
                ys = self.sample_trajectory(traj_length)

                # collect output
                output = ImportanceOutput(traj_length, ys, dim=self.dimension)

                # get evidence estimate using true params, add it to output, and plot
                log_true = self.get_true_score(ys, traj_length)
                # true = round(log_true.exp().item(), 8)
                true = log_true.exp().item()
                output.set_truth(true)
                print('true prob.: {}'.format(true))
                plt.axhline(y=true, color='b')

                # generate environment
                env = self.generate_env(ys, traj_length)

                # get evidence estimates using IS with other params
                for i, _A in enumerate(As):
                    for j, _Q in enumerate(Qs):
                        # get is estimate of evidence using _A and _Q params
                        eval_obj = importance_estimate(ys, A=_A, Q=_Q, C=self.C, R=self.R, mu_0=self.mu_0, Q_0=self.Q_0, env=env, N=num_samples)
                        print('IS A: {}, Q: {} estimate: {}'.format(_A, _Q, eval_obj.running_log_estimates[-1].exp()))

                        # add importance confidence interval
                        is_estimator = output.add_is_estimator(A=_A, Q=_Q,
                                                            running_log_estimates=eval_obj.running_log_estimates,
                                                            ci=eval_obj.ci,
                                                            weight_mean=eval_obj.log_weight_mean.exp(),
                                                            max_weight_prop=eval_obj.log_max_weight_prop.exp(),
                                                            ess=eval_obj.log_effective_sample_size.exp(), ess_ci=eval_obj.ess_ci,
                                                            idstr='A: {}, Q: {}'.format(_A, _Q))

                    # plot mean and empirical confidence interval
                    is_estimator.plot_running_log_estimates()

                # add RL plot
                try:
                    model_name = MODEL.format(1, 'entropy', traj_length, self.dimension)
                    eval_obj = rl_estimate(ys, dim=self.dimension, N=num_samples, model_name=model_name, env=env)
                    print('rl estimate: {}'.format(eval_obj.running_log_estimates[-1].exp()))
                    x_vals = torch.arange(1, len(eval_obj.running_log_estimates.squeeze())+1)

                    # plt.plot(x_vals, eval_obj.running_log_estimates.exp().squeeze(), label='RL')
                    # plt.fill_between(x_vals, y1=eval_obj.running_ess_ci[0], y2=eval_obj.running_ci[1], alpha=0.3)

                    # add rl confidence interval
                    rl_estimator = output.add_rl_estimator(running_log_estimates=eval_obj.running_log_estimates,
                                                        ci=eval_obj.ci, weight_mean=eval_obj.log_weight_mean.exp(),
                                                        max_weight_prop=eval_obj.log_max_weight_prop.exp(),
                                                        ess=eval_obj.log_effective_sample_size.exp(),
                                                        ess_ci=eval_obj.ess_ci, idstr='rl_{}'.format(traj_length))

                except:
                    print('error could not load: {}'.format(MODEL.format(1, 'entropy', traj_length, self.dimension)))

            try:
                # plot em
                rl_estimator.plot_running_log_estimates()
            except:
                pass

            outputs.append(output)

            plt.xlabel('Number of Samples')
            plt.ylabel('Prob. {} Estimate'.format(self.name))
            plt.title('Convergence of Prob. {} Estimate to True Prob. {} \n(trajectory length: {}, dimension: {})'.format(self.name, self.name, traj_length, self.dimension))
            plt.legend()
            plt.savefig('{}/{}traj_length_{}_dimension_{}_{}_convergence.pdf'.format(TODAY, name, traj_length, self.dimension, self.name))
            wandb.save('{}/{}traj_length_{}_dimension_{}_{}_convergence.pdf'.format(TODAY, name, traj_length, self.dimension, self.name))
            plt.close()
        return outputs


class EvidencePlotter(Plotter):
    def __init__(self, num_samples, A=None, Q=None, C=None,
                 R=None, mu_0=None, Q_0=None,
                 dim=torch.tensor(1), sample=True):
        self.num_samples = num_samples
        super().__init__('Evidence', A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0, dim=dim, sample=sample)

    def sample_trajectory(self, traj_length):
        # print('warning, outputting only zeros as trajectory')
        # return torch.zeros(traj_length)
        return generate_trajectory(traj_length, A=self.A, Q=self.Q, C=self.C, R=self.R, mu_0=self.mu_0, Q_0=self.Q_0)[0]

    def get_true_score(self, ys, traj_length):
        # ignore traj_length
        return importance_estimate(ys, A=self.A, Q=self.Q, C=self.C, R=self.R, mu_0=self.mu_0, Q_0=self.Q_0, N=self.num_samples).running_log_estimates[-1]

    def generate_env(self, ys, traj_length):
        # ignore traj_length
        return LinearGaussianEnv(A=self.A, Q=self.Q,
                                 C=self.C, R=self.R,
                                 mu_0=self.mu_0,
                                 Q_0=self.Q_0, ys=ys,
                                 traj_length=len(ys),
                                 sample=self.sample)


class EventPlotter(Plotter):
    def __init__(self, event_prob, fix_event=False, A=None, Q=None, C=None,
                 R=None, mu_0=None, Q_0=None, dim=torch.tensor(1), sample=True):
        super().__init__('Event', A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0, dim=dim, sample=sample)
        self.event_prob = event_prob
        self.fix_event = fix_event

    def sample_trajectory(self, num_steps):
        ys, _, d = sample_y(num_steps=num_steps)
        if self.fix_event:
            return torch.ones_like(ys)
        threshold = d.icdf(1-self.event_prob)
        return (ys > threshold).type(ys.dtype)

    def get_true_score(self, ys, traj_length):
        return dist.Bernoulli(self.event_prob).log_prob(ys)

    def generate_env(self, ys, traj_length):
        return LinearGaussianSingleYEnv(A=self.A, Q=self.Q,
                                        C=self.C, R=self.R,
                                        mu_0=self.mu_0,
                                        Q_0=self.Q_0, ys=ys,
                                        traj_length=traj_length,
                                        sample=self.sample,
                                        event_prob=self.event_prob)

    def plot_IS(self, traj_lengths=torch.arange(10, 121, 10), As=torch.arange(0.2, 0.6, 0.2), Qs=torch.arange(0.2, 0.6, 0.2),
                num_samples=10000, num_repeats=10):
        for t in traj_lengths:
            d = y_dist(t)
            print('true dist (for traj_length {}) (dim {}): N({}, {})'.format(t, self.dimension, d.mean.item(), d.variance.item()))
        return super().plot_IS(traj_lengths, As, Qs, num_samples=num_samples, num_repeats=num_repeats)


def train_dimensions(traj_length, dimensions, table, condition_length, ignore_reward):
    os.makedirs(TODAY, exist_ok=True)
    for dim in dimensions:
        dimension = dim.item()
        A = table[dimension]['A']
        Q = table[dimension]['Q']
        R = table[dimension]['R']
        C = table[dimension]['C']
        mu_0 = table[dimension]['mu_0']
        Q_0 = table[dimension]['Q_0']
        env = LinearGaussianEnv(A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0,
                                ys=None, traj_length=traj_length, sample=True)
        train(traj_length, env, dim=dimension, condition_length=condition_length, ignore_reward=ignore_reward)

def compute_posterior(A, Q, R, C, num_observations, dim):
    w = MultiGaussianRandomVariable(mu=0., sigma=torch.sqrt(Q), name="w")
    v = MultiGaussianRandomVariable(mu=0., sigma=torch.sqrt(R), name="v")
    xt = MultiGaussianRandomVariable(mu=torch.zeros(dim), sigma=torch.sqrt(Q), name="x")
    xs = [xt]
    ys = []
    for i in range(num_observations):
        yt = MultiLinearGaussian(a=C, x=xt, b=v, name="y")
        ys.append(yt)
        xt = MultiLinearGaussian(a=A, x=xt, b=w, name="x")
        xs.append(xt)

    prior = None
    if num_observations > 1:
        prior = xs[num_observations-1].likelihood()
        for i in range(num_observations-2, 0, -1):
            lik = xs[i].likelihood()
            prior *= lik
    p_x_0 = xs[0].prior()
    prior = p_x_0 if prior is None else prior * p_x_0

    noise = v.prior()**num_observations

    # find full likelihood
    ys = VecLinearGaussian(a=C.t(), x=prior, b=noise)

    # compute posterior
    return ys.posterior_vec()

def condition_posterior(td, obs):
    return td.dist(arg={'value': obs})

def evaluate_posterior(ys, N, td, env=None):
    run = wandb.init(project='linear_gaussian_model evaluation', save_code=True, entity='iai')
    print('\nevaluating posterior...')
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
    d = int(td.mean.nelement()/n)
    for i in range(N):
        done = False
        # reset env
        env.reset()
        # keep track of xs
        xs = td.sample()
        if d > 1:
            xs = xs.reshape(-1, d)
        # keep track of prior over actions p(x)
        log_p_x = torch.tensor(0.).reshape(1, -1)
        log_p_y_given_x = torch.tensor(0.).reshape(1, -1)
        # collect actions, likelihoods
        actions = xs
        if len(xs) > 1:
            xts = torch.cat([env.prev_xt.reshape(-1), xs[0:-1].reshape(-1)])
        else:
            xts = env.prev_xt.reshape(-1)
        xts = xts.reshape(-1, d)
        prior_reward = score_initial_state(x0=xs[0].reshape(-1), mu_0=env.mu_0, Q_0=env.Q_0)  # the reward for the initial state
        lik_reward = score_y(y_test=ys[0], x_t=xs[0], C=env.C, R=env.R)  # the first likelihood reward
        priors = [prior_reward]
        log_p_x += prior_reward
        liks = [lik_reward]
        log_p_y_given_x += lik_reward
        total_reward = prior_reward + lik_reward
        if n > 1:
            for xt, prev_xt, y in zip(xs[1:], xts[1:], ys[1:]):
                prior_reward = score_state_transition(xt=xt.reshape(-1), prev_xt=prev_xt, A=env.A, Q=env.Q)
                lik_reward = score_y(y_test=y, x_t=xt, C=env.C, R=env.R)  # the first likelihood reward
                total_reward += prior_reward + lik_reward
                priors.append(prior_reward)
                liks.append(lik_reward)
                log_p_x += prior_reward
                log_p_y_given_x += lik_reward
        # log p(x,y)
        log_p_y_x = log_p_y_given_x + log_p_x
        log_q = td.log_prob(xs.flatten())
        if N == 1:
            log_p_y_over_qs[i] = (log_p_y_x - log_q).item()
            running_log_evidence_estimates.append(log_p_y_over_qs[0])
        else:
            log_p_y_over_qs[i] = (log_p_y_x - log_q).item()
            running_log_evidence_estimates.append(torch.logsumexp(log_p_y_over_qs[0:i+1], -1) - torch.log(torch.tensor(i+1.)))
        log_weights.append(log_p_x - log_q)  # ignore these since we consider the weights to be p(y|x)p(x)/q(x)
        total_rewards.append(total_reward)
        wandb.log({'total_reward': total_reward})

    print('\nevaluated posterior')
    # calculate variance estmate as
    # $\hat{\sigma}_{int}^2=(n(n-1))^{-1}\sum_{i=1}^n(f_iW_i-\overline{fW})^2$
    sigma_est = torch.sqrt( (logvarexp(log_p_y_over_qs) - torch.log(torch.tensor(len(log_p_y_over_qs) - 1, dtype=torch.float32))).exp() )
    return EvaluationObject(running_log_estimates=torch.tensor(running_log_evidence_estimates), sigma_est=sigma_est,
                            xts=xts, states=xts, actions=actions, priors=priors, liks=liks,
                            log_weights=log_p_y_over_qs)


class PosteriorEvidence:
    def __init__(self, td, ys, evidence, env):
        self.td = td
        self.ys = ys
        self.evidence = evidence
        self.env = env


def compute_evidence(table, traj_length, dim):
    os.makedirs(TODAY, exist_ok=True)

    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    mu_0 = table[dim]['mu_0']
    Q_0 = table[dim]['Q_0']

    ys = generate_trajectory(traj_length, A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0)[0]
    env = LinearGaussianEnv(A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0, using_entropy_loss=True, ys=ys, sample=True)

    posterior = compute_posterior(A=A, Q=Q, C=C, R=R, num_observations=len(ys), dim=dim)
    td = condition_posterior(posterior, ys)

    eval_obj = evaluate_posterior(ys=ys, N=1, td=td, env=env)
    true = eval_obj.running_log_estimates[0]

    # # true evidence
    # lgv = get_linear_gaussian_variables(dim=dim, num_obs=traj_length)
    # jvs = JointVariables(lgv.ys[0:2], A=A, C=C)
    # print('true evidence: ', jvs.dist.log_prob(end_ys))
    # print('True evidence: {}'.format(true))

    return PosteriorEvidence(td, ys, true, env)

def get_prior_output(table, ys, dim, sample, traj_length=0):
    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    mu_0 = table[dim]['mu_0']
    Q_0 = table[dim]['Q_0']

    if ys is not None:
        traj_length = len(ys)
    prior_output = ImportanceOutput(traj_length=traj_length, ys=ys, dim=dim)
    name = '{}(A: {}, Q: {})'.format(PRIOR_DISTRIBUTION, A, Q)
    for _ in range(NUM_REPEATS):
        eval_obj = importance_estimate(ys, A=A, Q=Q, C=C, R=R, mu_0=torch.zeros(dim), Q_0=Q, N=NUM_SAMPLES, sample=sample, traj_length=traj_length)
        is_estimator = prior_output.add_is_estimator(A=A, Q=Q,
                                                     running_log_estimates=eval_obj.running_log_estimates,
                                                     ci=eval_obj.ci,
                                                     weight_mean=eval_obj.log_weight_mean.exp(),
                                                     max_weight_prop=eval_obj.log_max_weight_prop.exp(),
                                                     ess=eval_obj.log_effective_sample_size.exp(), ess_ci=eval_obj.ess_ci,
                                                     idstr=name)
    return OutputWithName(prior_output, name)

def get_rl_output(linear_gaussian_env_type, table, ys, dim, model_name, traj_length=0):
    if ys is not None:
        traj_length = len(ys)
    rl_output = ImportanceOutput(traj_length=traj_length, ys=ys, dim=dim)

    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    mu_0 = table[dim]['mu_0']
    Q_0 = table[dim]['Q_0']

    loss_type = get_loss_type(model_name)

    env = linear_gaussian_env_type(A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q,
                                    using_entropy_loss=(loss_type==ENTROPY_LOSS),
                                    ys=ys, traj_length=traj_length, sample=False)

    sample_ys = ys is None
    for _ in range(NUM_REPEATS):
        if sample_ys:
            # create a new environment
            ys = generate_trajectory(traj_length, A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0)[0]
            env = linear_gaussian_env_type(A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q,
                                           using_entropy_loss=(loss_type==ENTROPY_LOSS),
                                           ys=ys, traj_length=traj_length, sample=False)
            env.reset()

        eval_obj = rl_estimate(ys, dim=dim, N=NUM_SAMPLES*traj_length**0, model_name=model_name,
                               env=env, traj_length=traj_length)
        # add rl confidence interval
        rl_estimator = rl_output.add_rl_estimator(running_log_estimates=eval_obj.running_log_estimates,
                                                  ci=eval_obj.ci, weight_mean=eval_obj.log_weight_mean.exp(),
                                                  max_weight_prop=eval_obj.log_max_weight_prop.exp(),
                                                  ess=eval_obj.log_effective_sample_size.exp(),
                                                  ess_ci=eval_obj.ess_ci, idstr=model_name)
    rl_estimator.save_data()
    return OutputWithName(rl_output, model_name)

def get_perturbed_posterior_output(traj_length, dim, epsilon, name):
    posterior_output = ImportanceOutput(traj_length=traj_length, ys=None, dim=dim)
    for _ in range(NUM_REPEATS):
        posterior_evidence = compute_evidence(table=table, traj_length=traj_length, dim=dim)
        ys = posterior_evidence.ys
        true_posterior = posterior_evidence.td
        env = posterior_evidence.env

        # get importance weighted score for comparison
        td = dist.MultivariateNormal(true_posterior.mean, true_posterior.covariance_matrix + epsilon * torch.eye(true_posterior.mean.shape[0]))

        eval_obj = evaluate_posterior(ys=ys, N=NUM_SAMPLES, td=td, env=env)
        posterior_estimator = posterior_output.add_rl_estimator(running_log_estimates=eval_obj.running_log_estimates,
                                                                ci=eval_obj.ci, weight_mean=eval_obj.log_weight_mean.exp(),
                                                                max_weight_prop=eval_obj.log_max_weight_prop.exp(),
                                                                ess=eval_obj.log_effective_sample_size.exp(),
                                                                ess_ci=eval_obj.ess_ci, idstr=name)
    posterior_estimator.save_data()
    return OutputWithName(posterior_output, name)

def get_perturbed_posterior_fixed_obs_output(posterior_evidence, dim, epsilon, name):
    ys = posterior_evidence.ys
    true_posterior = posterior_evidence.td
    env = posterior_evidence.env
    posterior_output = ImportanceOutput(traj_length=len(ys), ys=None, dim=dim)

    for _ in range(NUM_REPEATS):
        # get importance weighted score for comparison
        td = dist.MultivariateNormal(true_posterior.mean, true_posterior.covariance_matrix + epsilon * torch.eye(true_posterior.mean.shape[0]))

        eval_obj = evaluate_posterior(ys=ys, N=NUM_SAMPLES, td=td, env=env)
        posterior_estimator = posterior_output.add_rl_estimator(running_log_estimates=eval_obj.running_log_estimates,
                                                                ci=eval_obj.ci, weight_mean=eval_obj.log_weight_mean.exp(),
                                                                max_weight_prop=eval_obj.log_max_weight_prop.exp(),
                                                                ess=eval_obj.log_effective_sample_size.exp(),
                                                                ess_ci=eval_obj.ess_ci, idstr=name)
    posterior_estimator.save_data()
    return OutputWithName(posterior_output, name)


def get_evidence_two_ways(traj_length, dim, env, ys, condition_length):
    end_len = len(ys)-condition_length+1
    posterior = compute_posterior(A=env.A, Q=env.Q, C=env.C, R=env.R, num_observations=len(ys)-condition_length+1, dim=dim)
    td = condition_posterior(posterior, ys[0:end_len])

    end_ys = ys[0:end_len]

    eval_obj = evaluate_posterior(ys=end_ys, N=3, td=td, env=env)
    true = eval_obj.running_log_estimates[0]

    # true evidence
    lgv = get_linear_gaussian_variables(dim=dim, num_obs=len(ys))
    jvs = JointVariables(lgv.ys[0:end_len], A=env.A, C=env.C)
    print('true evidence: ', jvs.dist.log_prob(end_ys))
    print('True evidence: {}'.format(true))


def get_perturbed_posterior_filtering_output(table, traj_length, dim, epsilon, name, condition_length):
    posterior_output = ImportanceOutput(traj_length=traj_length, ys=None, dim=dim)
    # get importance weighted score for comparison

    for _ in range(NUM_REPEATS):
        posterior_evidence = compute_evidence(table=table, traj_length=traj_length, dim=dim)

        ys = posterior_evidence.ys
        env = posterior_evidence.env

        fps = compute_conditional_filtering_posteriors(table=table, num_obs=len(posterior_evidence.ys),
                                                       dim=dim, m=condition_length, condition_on_x=True,
                                                       ys=posterior_evidence.ys)

        eval_obj = evaluate_filtering_posterior(ys=ys, N=NUM_SAMPLES, tds=fps, epsilon=epsilon, env=env, m=condition_length)
        posterior_estimator = posterior_output.add_rl_estimator(running_log_estimates=eval_obj.running_log_estimates,
                                                                ci=eval_obj.ci, weight_mean=eval_obj.log_weight_mean.exp(),
                                                                max_weight_prop=eval_obj.log_max_weight_prop.exp(),
                                                                ess=eval_obj.log_effective_sample_size.exp(),
                                                                ess_ci=eval_obj.ess_ci, idstr=name)

    posterior_estimator.save_data()
    return OutputWithName(posterior_output, name)

def get_perturbed_posterior_filtering_fixed_obs_output(table, posterior_evidence, dim, epsilon, name, condition_length):
    ys = posterior_evidence.ys
    env = posterior_evidence.env

    fps = compute_conditional_filtering_posteriors(table=table, num_obs=len(posterior_evidence.ys),
                                                    dim=dim, m=condition_length, condition_on_x=True,
                                                    ys=posterior_evidence.ys)

    posterior_output = ImportanceOutput(traj_length=len(ys), ys=ys, dim=dim)
    for _ in range(NUM_REPEATS):
        # get importance weighted score for comparison

        eval_obj = evaluate_filtering_posterior(ys=ys, N=NUM_SAMPLES, tds=fps, epsilon=epsilon, env=env, m=condition_length)
        posterior_estimator = posterior_output.add_rl_estimator(running_log_estimates=eval_obj.running_log_estimates,
                                                                ci=eval_obj.ci, weight_mean=eval_obj.log_weight_mean.exp(),
                                                                max_weight_prop=eval_obj.log_max_weight_prop.exp(),
                                                                ess=eval_obj.log_effective_sample_size.exp(),
                                                                ess_ci=eval_obj.ess_ci, idstr=name)

    posterior_estimator.save_data()
    return OutputWithName(posterior_output, name)


class OutputWithName:
    def __init__(self, output, name):
        self.output = output
        self.name = name


def save_outputs_with_names_dim(outputs, distribution_type, label):
    columns = [str(output.output.dimension) for output in outputs]
    save_outputs_with_names(outputs, distribution_type, label, columns, 'dim')

def save_outputs_with_names_traj(outputs, distribution_type, label):
    columns = [str(output.output.traj_length) for output in outputs]
    save_outputs_with_names(outputs, distribution_type, label, columns, 'traj')

def save_outputs_with_names_general(outputs, distribution_type, label, output_type):
    columns = [str(output.output.traj_length) for output in outputs]
    save_outputs_with_names(outputs, distribution_type, label, columns, output_type)

def save_outputs_with_names(outputs, distribution_type, label, columns, output_type):
    os.makedirs('{}/{}'.format(TODAY, distribution_type), exist_ok=True)
    ess_output = torch.stack([torch.tensor(output.output[output.name].ess) for output in outputs], dim=1)
    df = pd.DataFrame(ess_output.numpy(), columns=columns)
    df.to_csv('{}/{}/{}_ESS_{}.csv'.format(TODAY, distribution_type, label, output_type))

def get_perturbed_posterior_outputs(traj_length, dim, epsilons):
    outputs = []
    for epsilon in epsilons:
        name = '{}(epsilon: {} dim: {} traj_len: {})'.format(POSTERIOR_DISTRIBUTION, epsilon, dim, traj_length)
        outputs += [get_perturbed_posterior_output(traj_length, dim, epsilon, name)]
    return outputs

def get_perturbed_posterior_fixed_obs_outputs(posterior_evidence, dim, epsilons):
    outputs = []
    for epsilon in epsilons:
        name = '{}(epsilon: {} dim: {} traj_len: {})'.format(POSTERIOR_DISTRIBUTION, epsilon, dim, len(posterior_evidence.ys))
        outputs += [get_perturbed_posterior_fixed_obs_output(posterior_evidence, dim, epsilon, name)]
    return outputs

def get_perturbed_posterior_filtering_outputs(table, traj_length, dim, epsilons, condition_length):
    outputs = []
    for epsilon in epsilons:
        name = '{}(epsilon: {} dim: {} traj_len: {})'.format(FILTERING_POSTERIOR_DISTRIBUTION, epsilon, dim, traj_length)
        outputs += [get_perturbed_posterior_filtering_output(table, traj_length, dim, epsilon, name, condition_length)]
    return outputs

def plot_ess_estimators(outputs_with_names, fixed_feature):
    for output in outputs_with_names:
        estimator = output.output[output.name]
        estimator.plot_ess()

def plot_ess_estimators_traj(outputs_with_names, traj_lengths):
    # estimates = []
    # for output in outputs_with_names:
    #     estimates.append(torch.tensor([repeat[-1] for repeat in output.output[output.name].ess]))
    # estimates = torch.stack(estimates, dim=1)

    outputs = torch.stack([torch.tensor(output.output[output.name].ess) for output in outputs_with_names], dim=1)
    quantiles = torch.tensor([0.05, 0.5, 0.95])
    lower_ci, med, upper_ci = torch.quantile(outputs, quantiles, dim=0)
    plt.plot(traj_lengths, med.squeeze(), label=outputs_with_names[0].name)
    plt.fill_between(traj_lengths, y1=lower_ci, y2=upper_ci, alpha=0.3)

    ax = plt.gca()
    ax.xaxis.get_major_locator().set_params(integer=True)

def plot_ess_estimators_dim(outputs_with_names, dims):
    outputs = torch.stack([torch.tensor(output.output[output.name].ess) for output in outputs_with_names], dim=1)
    quantiles = torch.tensor([0.05, 0.5, 0.95])
    lower_ci, med, upper_ci = torch.quantile(outputs, quantiles, dim=0)
    plt.plot(dims, med.squeeze(), label=outputs_with_names[0].name)
    plt.fill_between(dims, y1=lower_ci, y2=upper_ci, alpha=0.3)

    ax = plt.gca()
    ax.xaxis.get_major_locator().set_params(integer=True)

def plot_ess_estimators_condition_length(outputs_with_names, condition_lengths):
    outputs = torch.stack([torch.tensor(output.output[output.name].ess) for output in outputs_with_names], dim=1)
    quantiles = torch.tensor([0.05, 0.5, 0.95])
    lower_ci, med, upper_ci = torch.quantile(outputs, quantiles, dim=0)
    plt.plot(condition_lengths, med.squeeze(), label=outputs_with_names[0].name)
    plt.fill_between(condition_lengths, y1=lower_ci, y2=upper_ci, alpha=0.3)

    ax = plt.gca()
    ax.xaxis.get_major_locator().set_params(integer=True)

def plot_running_log_estimates(outputs_with_names):
    for output in outputs_with_names:
        estimator = output.output[output.name]
        estimator.plot_running_log_estimates()

def plot_convergence(outputs_with_names, traj_length, dim, true, name):
    plot_running_log_estimates(outputs_with_names)
    output = outputs_with_names[0]
    estimator = output.output[output.name]
    estimate = estimator.compute_evidence_estimate()

    xlen = estimator.running_log_estimate_repeats[0].squeeze().nelement()
    # plot em
    plt.scatter(x=xlen, y=true, label='True: {}'.format(true.item()), color='r')
    plt.scatter(x=xlen, y=estimate, label='Estimate: {}'.format(estimate.item()), color='b')
    plt.xlabel('Number of Samples')
    plt.ylabel('Prob. {} Estimate'.format('Evidence'))
    plt.title('Convergence of Prob. {} Estimate to True Prob. {} \n(trajectory length: {}, dimension: {})'.format('Evidence', 'Evidence', traj_length, dim))
    legend_without_duplicate_labels(plt.gca())
    model_name = model_without_directory(name)
    plt.savefig(TODAY+'/'+model_name+'(traj_length_{}_dim_{})Convergence.pdf'.format(traj_length, dim))
    wandb.save(TODAY+'/'+model_name+'Convergence.pdf')
    plt.close()

def posterior_convergence(posterior_evidence, dim, epsilons):
    posterior_outputs_with_names = get_perturbed_posterior_fixed_obs_outputs(posterior_evidence, dim, epsilons)
    plot_convergence(posterior_outputs_with_names, traj_length, dim, posterior_evidence.evidence, 'posterior')

def posterior_filtering_convergence(table, posterior_evidence, dim, epsilons):
    posterior_outputs_with_names = get_perturbed_posterior_filtering_fixed_obs_outputs(table, traj_length, dim, epsilons, condition_length=0)
    plot_convergence(posterior_outputs_with_names, traj_length, dim, posterior_evidence.evidence, 'posterior_filtering')

def posterior_filtering_conditional_convergence(table, posterior_evidence, dim, condition_length):
    name = '{}(dim: {} traj_len: {} condition_length: {})'.format(FILTERING_POSTERIOR_CONDITIONAL_DISTRIBUTION, dim,
                                                                  len(posterior_evidence.ys), condition_length)
    posterior_outputs_with_names = [get_perturbed_posterior_filtering_fixed_obs_output(table, posterior_evidence, dim, epsilon=0.,
                                                                                       name=name, condition_length=condition_length)]
    traj_length = len(posterior_evidence.ys)
    plot_convergence(posterior_outputs_with_names, traj_length, dim, posterior_evidence.evidence, 'posterior_filtering_conditional')

def prior_convergence(table, ys, truth, dim):
    prior_outputs_with_name = get_prior_output(table=table, ys=ys, dim=dim, sample=False)
    traj_length = len(ys)
    plot_convergence([prior_outputs_with_name], traj_length, dim, truth, 'prior')

def rl_convergence(linear_gaussian_env_type, table, ys, truth, dim, model_name):
    rl_outputs_with_name = get_rl_output(linear_gaussian_env_type, table=table, ys=ys, dim=dim, model_name=model_name)
    traj_length = len(ys)
    plot_convergence([rl_outputs_with_name], traj_length, dim, truth, model_name)

def plot_dim(traj_length, dim):
    dimensions = torch.arange(5, 11, 1)
    assert dim in dimensions
    torch.manual_seed(traj_length)
    table = create_dimension_table(dimensions)

    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    mu_0 = table[dim]['mu_0']
    Q_0 = table[dim]['Q_0']
    ys = generate_trajectory(traj_length, A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0)[0]
    ep = EvidencePlotter(num_samples=100, dim=dim, A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0)

    outputs = ep.plot_IS(traj_lengths=torch.tensor([traj_length]), As=[], Qs=[], num_samples=100, num_repeats=10, name='extra')

class EvidenceConvergence:
    def __init__(self, posterior_evidence, dim, lower_quantile, upper_quantile):
        self.posterior_evidence = posterior_evidence
        self.dim = dim
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def compute_data(self, data):
        lower_ci, med, upper_ci = torch.quantile(data, torch.tensor([self.lower_quantile, self.upper_quantile]), dim=0)


class PosteriorConvergence(EvidenceConvergence):
    def __init__(self, posterior_evidence, dim, epsilons):
        super(Posterior_Evidence, self).__init__(posterior_evidence, dim)
        self.epsilons = epsilons

    def get_output_with_name(self, epsilon):
        ys = self.posterior_evidence.ys
        true_posterior = self.posterior_evidence.td
        env = self.posterior_evidence.env

        name = '{}_{}'.format(POSTERIOR_DISTRIBUTION, epsilon)

        posterior_output = ImportanceOutput(traj_length=len(ys), ys=ys, dim=dim)
        # get importance weighted score for comparison
        td = dist.MultivariateNormal(true_posterior.mean, true_posterior.covariance_matrix + epsilon * torch.eye(true_posterior.mean.shape[0]))

        for _ in range(NUM_REPEATS):
            eval_obj = evaluate_posterior(ys=ys, N=NUM_SAMPLES, td=td, env=env)
            posterior_estimator = posterior_output.add_rl_estimator(running_log_estimates=eval_obj.running_log_estimates,
                                                                    ci=eval_obj.ci, weight_mean=eval_obj.log_weight_mean.exp(),
                                                                    max_weight_prop=eval_obj.log_max_weight_prop.exp(),
                                                                    ess=eval_obj.log_effective_sample_size.exp(),
                                                                    ess_ci=eval_obj.ess_ci, idstr=name)
        return OutputWithName(posterior_output, name)


class PriorConvergence(EvidenceConvergence):
    def __init__(self, posterior_evidence, dim):
        super(Posterior_Evidence, self).__init__(posterior_evidence, dim)

    def get_output_with_name(self, table):
        return get_prior_output(table=table, ys=self.posterior.ys, dim=self.dim, sample=False)


class RLConvergence(EvidenceConvergence):
    def __init__(self, posterior_evidence, dim, linear_gaussian_env_type):
        super(Posterior_Evidence, self).__init__(posterior_evidence, dim)
        self.name = 'RL'
        self.linear_gaussian_env_type = linear_gaussian_env_type

    def get_output_with_name(self, table):
        return get_rl_output(self.linear_gaussian_env_type, table=table, ys=self.posterior_evidence.ys, dim=self.dim, sample=False)


def compare_convergence(linear_gaussian_env_type, table, traj_length, dim, epsilons, model_name, condition_length):
    posterior_evidence = compute_evidence(table=table, traj_length=traj_length, dim=dim)
    posterior_convergence(posterior_evidence=posterior_evidence, dim=dim, epsilons=epsilons)
    prior_convergence(table=table, ys=posterior_evidence.ys, truth=posterior_evidence.evidence, dim=dim)
    try:
        rl_convergence(linear_gaussian_env_type, table=table, ys=posterior_evidence.ys,
                       truth=posterior_evidence.evidence, dim=dim,
                       model_name=model_name)
    except:
        pass

def get_posterior_ess_outputs(table, traj_length, dim, epsilon, condition_length):
    return get_perturbed_posterior_outputs(traj_length, dim, [epsilon])

def get_posterior_filtering_ess_outputs(table, traj_length, dim, epsilon):
    return get_perturbed_posterior_filtering_outputs(table=table, traj_length=traj_length, dim=dim, epsilons=[epsilon], condition_length=0)

def get_posterior_filtering_conditional_ess_outputs(table, traj_length, dim, condition_length):
    A = table[dim]['A']
    Q = table[dim]['Q']
    name = '{}(A: {} Q: {} condition_length: {})'.format(FILTERING_POSTERIOR_CONDITIONAL_DISTRIBUTION, A,
                                                         Q, condition_length)
    return [get_perturbed_posterior_filtering_output(table=table, traj_length=traj_length, dim=dim,
                                                     epsilon=0., name=name, condition_length=condition_length)]

def posterior_ess_traj(table, traj_lengths, dim, epsilon):
    distribution_type = POSTERIOR_DISTRIBUTION
    os.makedirs('{}/{}'.format(TODAY, distribution_type), exist_ok=True)
    outputs = []
    for traj_length in traj_lengths:
        outputs += get_posterior_ess_outputs(table, traj_length, dim, epsilon)
    save_outputs_with_names_traj(outputs, distribution_type, '{}(epsilon_{}_traj_lengths_{}_dim_{})'.format(distribution_type, epsilon, traj_lengths, dim))
    make_ess_plot_nice(outputs, fixed_feature_string='dimension', fixed_feature=dim,
                       num_samples=NUM_SAMPLES, num_repeats=NUM_REPEATS, traj_lengths=traj_lengths,
                       xlabel='Trajectory Length', distribution_type=distribution_type, name='posterior_{}'.format(epsilon))

def posterior_ess_dim(table, traj_length, dims, epsilon):
    distribution_type = POSTERIOR_DISTRIBUTION
    os.makedirs('{}/{}'.format(TODAY, distribution_type), exist_ok=True)
    outputs = []
    for dim in dims:
        outputs += get_posterior_ess_outputs(table, traj_length, dim, epsilon)
    save_outputs_with_names_dim(outputs, distribution_type, '{}(epsilon_{}_traj_length_{}_dims_{})'.format(distribution_type, epsilon, traj_length, dims))
    make_ess_plot_nice_dim(outputs, fixed_feature_string='traj_length', fixed_feature=traj_length,
                           num_samples=NUM_SAMPLES, num_repeats=NUM_REPEATS, dims=dims,
                           xlabel='Latent Dimension', distribution_type=distribution_type, name='posterior_{}'.format(epsilon))

def posterior_filtering_ess_traj(table, traj_lengths, dim, epsilon):
    distribution_type = FILTERING_POSTERIOR_DISTRIBUTION
    os.makedirs('{}/{}'.format(TODAY, distribution_type), exist_ok=True)
    outputs = []
    for traj_length in traj_lengths:
        outputs += get_posterior_filtering_ess_outputs(table, traj_length, dim, epsilon)
    save_outputs_with_names_traj(outputs, distribution_type,
                                 '{}(epsilon_{}_traj_lengths_{}_dim_{})'.format(distribution_type, epsilon, traj_lengths, dim))
    make_ess_plot_nice(outputs, fixed_feature_string='dimension', fixed_feature=dim,
                       num_samples=NUM_SAMPLES, num_repeats=NUM_REPEATS, traj_lengths=traj_lengths,
                       xlabel='Trajectory Length', distribution_type=distribution_type, name='posterior_filtering')

def posterior_filtering_ess_dim(table, traj_length, dims, epsilon):
    distribution_type = FILTERING_POSTERIOR_DISTRIBUTION
    os.makedirs('{}/{}'.format(TODAY, distribution_type), exist_ok=True)
    outputs = []
    for dim in dims:
        outputs += get_posterior_filtering_ess_outputs(table, traj_length, dim, epsilon)
    save_outputs_with_names_dim(outputs, distribution_type,
                                '{}({}_traj_length_{}_dims_{})'.format(distribution_type, epsilon, traj_length, dims))
    make_ess_plot_nice_dim(outputs, fixed_feature_string='traj_length', fixed_feature=traj_length,
                           num_samples=NUM_SAMPLES, num_repeats=NUM_REPEATS, dims=dims,
                           xlabel='Latent Dimension', distribution_type=distribution_type, name='posterior_{}'.format(epsilon))

def posterior_filtering_conditional_ess_traj(table, traj_lengths, dim, condition_length):
    distribution_type = FILTERING_POSTERIOR_CONDITIONAL_DISTRIBUTION
    os.makedirs('{}/{}'.format(TODAY, distribution_type), exist_ok=True)
    outputs = []
    for traj_length in traj_lengths:
        outputs += get_posterior_filtering_conditional_ess_outputs(table, traj_length, dim, condition_length)
    save_outputs_with_names_traj(outputs, distribution_type,
                                 '{}(traj_lengths_{}_dim_{}_condition_length_{})'.format(distribution_type, traj_lengths, dim, condition_length))
    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    make_ess_plot_nice(outputs, fixed_feature_string='dimension', fixed_feature=dim,
                       num_samples=NUM_SAMPLES, num_repeats=NUM_REPEATS, traj_lengths=traj_lengths,
                       xlabel='Trajectory Length', distribution_type=distribution_type, name='A_{}_Q_{}_C_{}_R_{}_posterior_filtering (conditioned on {} obs)'.format(A, Q, C, R, condition_length))

def prior_ess_traj(table, traj_lengths, dim):
    distribution_type = PRIOR_DISTRIBUTION
    os.makedirs('{}/{}'.format(TODAY, distribution_type), exist_ok=True)
    outputs = []
    for traj_length in traj_lengths:
        outputs += [get_prior_output(table=table, ys=None, dim=dim, sample=True, traj_length=traj_length)]
    save_outputs_with_names_dim(outputs, distribution_type, '{}(traj_lengths_{}_dim_{})'.format(distribution_type, traj_lengths, dim))
    make_ess_plot_nice(outputs, fixed_feature_string='dimension', fixed_feature=dim,
                       num_samples=NUM_SAMPLES, num_repeats=NUM_REPEATS, traj_lengths=traj_lengths,
                       xlabel='Trajectory Length', distribution_type=distribution_type, name='prior')

def prior_ess_dim(table, traj_length, dims):
    distribution_type = PRIOR_DISTRIBUTION
    os.makedirs('{}/{}'.format(TODAY, distribution_type), exist_ok=True)
    outputs = []
    for dim in dims:
        outputs += [get_prior_output(table=table, ys=None, dim=dim, sample=True, traj_length=traj_length)]
    save_outputs_with_names_dim(outputs, distribution_type, '{}(traj_length_{}_dims_{})'.format(distribution_type, traj_length, dims))
    make_ess_plot_nice_dim(outputs, fixed_feature_string='traj_length', fixed_feature=traj_length,
                           num_samples=NUM_SAMPLES, num_repeats=NUM_REPEATS, dims=dims,
                           xlabel='Latent Dimension', distribution_type=distribution_type, name='prior')

def rl_ess_traj(linear_gaussian_env_type, table, traj_lengths,
                dim, ent_coef, loss_type, condition_length,
                use_mlp_policy):
    distribution_type = RL_DISTRIBUTION
    os.makedirs('{}/{}'.format(TODAY, distribution_type), exist_ok=True)
    outputs = []
    for traj_length in traj_lengths:
        model_name = get_model_name(traj_length=traj_length, dim=dim,
                                    ent_coef=ent_coef, loss_type=loss_type,
                                    condition_length=condition_length,
                                    use_mlp_policy=use_mlp_policy)
        outputs += [get_rl_output(linear_gaussian_env_type, table=table, ys=None, dim=dim, model_name=model_name, traj_length=traj_length)]
    save_outputs_with_names_traj(outputs, distribution_type,
                                 '{}_{}(traj_lengths_{}_dim_{})'.format(distribution_type, loss_type, traj_lengths, dim))
    make_ess_plot_nice(outputs, fixed_feature_string='dimension', fixed_feature=dim,
                       num_samples=NUM_SAMPLES, num_repeats=NUM_REPEATS, traj_lengths=traj_lengths,
                       xlabel='Trajectory Length', distribution_type=distribution_type, name='{}_RL'.format(loss_type))

def rl_ess_dim(linear_gaussian_env_type, table, traj_length, dims, ent_coef, loss_type):
    distribution_type = RL_DISTRIBUTION
    os.makedirs('{}/{}'.format(TODAY, distribution_type), exist_ok=True)
    outputs = []
    for dim in dims:
        # ys = generate_trajectory(traj_length, A=single_gen_A, Q=single_gen_Q, C=single_gen_C, R=single_gen_R, mu_0=single_gen_mu_0, Q_0=single_gen_Q_0)[0]
        model_name = get_model_name(traj_length=traj_length, dim=dim, ent_coef=ent_coef, loss_type=loss_type)
        outputs += [get_rl_output(linear_gaussian_env_type, table=table, ys=None, dim=dim, model_name=model_name, traj_length=traj_length)]
    save_outputs_with_names_dim(outputs, distribution_type,
                                '{}_{}(traj_length_{}_dims_{})'.format(distribution_type, loss_type, traj_length, dims))
    make_ess_plot_nice_dim(outputs, fixed_feature_string='traj_length', fixed_feature=traj_length,
                           num_samples=NUM_SAMPLES, num_repeats=NUM_REPEATS, dims=dims,
                           xlabel='Latent Dimension', distribution_type=distribution_type, name='RL')

def execute_posterior_ess_traj(table, traj_lengths, epsilons, dim):
    os.makedirs(TODAY, exist_ok=True)
    for epsilon in epsilons:
        posterior_ess_traj(table=table, traj_lengths=traj_lengths, dim=dim, epsilon=epsilon)

def execute_posterior_ess_dim(table, traj_length, epsilons, dims):
    os.makedirs(TODAY, exist_ok=True)
    for epsilon in epsilons:
        posterior_ess_dim(table=table, traj_length=traj_length, dims=dims, epsilon=epsilon)

# def execute_compare_convergence_traj(table, traj_lengths, epsilons, dim, model_name):
#     os.makedirs(TODAY, exist_ok=True)
#     for traj_length in traj_lengths:
#         compare_convergence(table=table, traj_length=traj_length,
#                             dim=dim, epsilons=epsilons,
#                             model_name=model_name)

# def execute_compare_convergence_dim(table, traj_length, epsilons, dims, model_name):
#     os.makedirs(TODAY, exist_ok=True)
#     for dim in dims:
#         compare_convergence(table=table, traj_length=traj_length,
#                             dim=dim, epsilons=epsilons,
#                             model_name=model_name)

def posterior_filtering_conditional_ess_condition_length(table, traj_length, dim, condition_lengths):
    distribution_type = FILTERING_POSTERIOR_CONDITIONAL_DISTRIBUTION
    os.makedirs('{}/{}'.format(TODAY, distribution_type), exist_ok=True)
    outputs = []
    for condition_length in condition_lengths:
        outputs += get_posterior_filtering_conditional_ess_outputs(table, traj_length, dim, condition_length)
    save_outputs_with_names_general(outputs, distribution_type,
                                    '{}(traj_length_{}_dim_{}_condition_length_{})'.format(distribution_type, traj_length, dim, condition_length),
                                    output_type='condition_length')
    make_ess_plot_nice_condition_length(outputs, fixed_feature_string='traj_length', fixed_feature=traj_length,
                                        num_samples=NUM_SAMPLES, num_repeats=NUM_REPEATS, xlabel='Condition Length',
                                        condition_lengths=condition_lengths,
                                        distribution_type=distribution_type, name='posterior_filtering')

def execute_ess_traj(linear_gaussian_env_type, traj_lengths, dim,
                     epsilons, ent_coef, loss_type, condition_length,
                     use_mlp_policy):
    table = create_dimension_table(torch.tensor([dim]), random=False)
    os.makedirs(TODAY, exist_ok=True)
    rl_ess_traj(linear_gaussian_env_type, table=table, traj_lengths=traj_lengths,
                dim=dim, ent_coef=ent_coef, loss_type=loss_type,
                condition_length=condition_length, use_mlp_policy=use_mlp_policy)
    for epsilon in epsilons:
        # posterior_ess_traj(table=table, traj_lengths=traj_lengths, dim=dim, epsilon=epsilon)
        posterior_filtering_ess_traj(table=table, traj_lengths=traj_lengths, dim=dim, epsilon=epsilon)
    # prior_ess_traj(table=table, traj_lengths=traj_lengths, dim=dim)

def execute_ess_dim(linear_gaussian_env_type, table, traj_length, dims, epsilons, ent_coef, loss_type):
    os.makedirs(TODAY, exist_ok=True)
    for epsilon in epsilons:
        # posterior_ess_dim(table=table, traj_length=traj_length, dims=dims, epsilon=epsilon)
        posterior_filtering_ess_dim(table=table, traj_length=traj_length, dims=dims, epsilon=epsilon)
    # prior_ess_dim(table=table, traj_length=traj_length, dims=dims)
    # rl_ess_dim(linear_gaussian_env_type, table=table, traj_length=traj_length, dims=dims, ent_coef=ent_coef, loss_type=loss_type)

def trial_evidence(table, traj_length, dim, condition_length):
    posterior_evidence = compute_evidence(table=table, traj_length=traj_length, dim=dim)
    return posterior_evidence.evidence

def execute_filtering_posterior_convergence(table, traj_lengths, epsilons, dim, condition_length):
    os.makedirs(TODAY, exist_ok=True)
    for traj_length in traj_lengths:
        posterior_evidence = compute_evidence(table=table, traj_length=traj_length, dim=dim)
        posterior_filtering_convergence(table=table, posterior_evidence=posterior_evidence, dim=dim, epsilons=epsilons)

def execute_filtering_posterior_conditional_convergence(table, traj_lengths, dim, condition_length):
    os.makedirs(TODAY, exist_ok=True)
    for traj_length in traj_lengths:
        posterior_evidence = compute_evidence(table=table, traj_length=traj_length, dim=dim)
        posterior_filtering_conditional_convergence(table=table, posterior_evidence=posterior_evidence,
                                                    dim=dim, condition_length=condition_length)

def verify_filtering_posterior():
    os.makedirs(TODAY, exist_ok=True)
    epsilons = [-5e-2]
    traj_lengths = torch.arange(2, 5, 1)
    dim = 1
    table = create_dimension_table(torch.tensor([dim]), random=False)

    traj_length = 5
    condition_length = 3
    posterior_evidence = compute_evidence(table, traj_length, dim)

    ys = posterior_evidence.ys
    fps = compute_conditional_filtering_posteriors(table=table, num_obs=traj_length, dim=dim, ys=posterior_evidence.ys)

    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    mu_0 = table[dim]['mu_0']
    Q_0 = table[dim]['Q_0']
    env = LinearGaussianEnv(A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0, using_entropy_loss=True, ys=ys, sample=True)

    eval_obj = evaluate_filtering_posterior(ys=ys, N=2, tds=fps, env=env)
    actions = eval_obj.actions
    true_score = posterior_evidence.td.log_prob(torch.tensor(actions))

    actions = [None] + actions
    print('true posterior score: ', true_score)
    score = torch.tensor(0.)
    for i, (action, td) in enumerate(zip(actions[1:], fps)):
        score += td.condition(y_values=ys[i:], x_value=actions[i]).log_prob(action)
    print('filtering posterior score: ', score)

def test_train(traj_length, dim, condition_length, ent_coef, loss_type,
               learning_rate, clip_range, linear_gaussian_env_type,
               continue_training, ignore_reward, use_mlp_policy,
               rl_type):
    table = create_dimension_table(torch.tensor([dim]), random=False)
    posterior_evidence = compute_evidence(table, traj_length, dim)

    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    mu_0 = table[dim]['mu_0']
    Q_0 = table[dim]['Q_0']

    print("WARNING: not sampling new observations during training!")
    env = linear_gaussian_env_type(A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0,
                                   using_entropy_loss=(loss_type==ENTROPY_LOSS),
                                   ys=posterior_evidence.ys, sample=False)

    train(traj_length=traj_length, env=env, dim=dim, condition_length=condition_length,
          ent_coef=ent_coef, loss_type=loss_type,
          learning_rate=learning_rate, clip_range=clip_range,
          continue_training=continue_training,
          ignore_reward=ignore_reward, use_mlp_policy=use_mlp_policy,
          rl_type=rl_type)

def sample_variance_ratios(traj_length, model_name, condition_length, policy=None):
    """
    This function generates NUM_SAMPLES trajectories of length traj_length
    and computes the ratios of the filtering posterior variance to that of the rl agent.
    """

    if policy is None:
        # load rl policy
        _, policy = load_rl_model(model_name=model_name, device='cpu')  # assume dimensionality equals 1 so the variance is just a scalar

    # set up to create filtering posterior
    dim = 1  # assume dimension is 1 so that variances are scalars
    table = create_dimension_table(torch.tensor([dim]), random=False)

    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    mu_0 = table[dim]['mu_0']
    Q_0 = table[dim]['Q_0']

    # keep state occupancy for each state in traj_length
    state_occupancy = []

    # variance ratios
    mean_diffs = []
    variance_ratios = []

    if condition_length == traj_length:
        m = 0
    else:
        m = condition_length-1

    for k in range(NUM_SAMPLES):
        # generate a set of ys using the true model parameters
        traj_ys, traj_xs = generate_trajectory(traj_length, A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0)[0:2]
        # create filtering distribution given the ys
        _tds = compute_conditional_filtering_posteriors(table=table, num_obs=traj_length, dim=dim, ys=traj_ys)
        tds = _tds[0:traj_length-m]

        # get first filtering distribution p(x0 | y0:yT)
        td = tds[0].condition(y_values=traj_ys)

        # get first obs
        xt = traj_xs[0].reshape(1)
        y0 = traj_ys[0:condition_length]
        obs = torch.cat([torch.zeros_like(xt), y0]).reshape(1, -1)

        # collect ratio of mean and variance at each step
        policy_dist_zero = policy.get_distribution(obs).distribution
        mean_diff_steps = torch.tensor(td.mean() - policy_dist_zero.mean.detach())
        variance_ratio_steps = torch.tensor(td.covariance() / policy_dist_zero.scale)
        for j in range(1, len(tds)):
            td_fps = tds[j]
            y = traj_ys[j:]

            # filtering dis
            dst = td_fps.condition(y_values=y, x_value=xt)
            filtering_mean = dst.mean()
            filtering_variance = dst.covariance()

            # policy dist
            obs = torch.cat([xt, y[0:condition_length]]).reshape(1, -1)
            policy_dist = policy.get_distribution(obs).distribution

            # mean ratio
            rl_mean = policy_dist.mean
            mean_diff = (filtering_mean - rl_mean).reshape(1, 1)
            mean_diff_steps = torch.cat((mean_diff_steps.reshape(1, -1), mean_diff), dim=1)

            # variance ratio
            rl_variance = policy_dist.scale.detach()
            variance_ratio = (filtering_variance / rl_variance).reshape(1, 1)
            variance_ratio_steps = torch.cat((variance_ratio_steps.reshape(1, -1), variance_ratio), dim=1)

            # print('filtering_variance: {}'.format(filtering_variance))
            # print('rl_variance: {}'.format(rl_variance))
            # print('variance_ratio_steps {}'.format(variance_ratio_steps))

            # if torch.abs(variance_ratio[0, 0] - variance_ratio_steps[0, 0]).item() > 0.001:
            #     import pdb; pdb.set_trace()
            #     dst = td_fps.condition(y_values=y, x_value=xt)

            # get next hidden state
            xt = traj_xs[j].reshape(1)
        mean_diffs.append(mean_diff_steps)
        variance_ratios.append(variance_ratio_steps)
    return torch.stack(mean_diffs).reshape(NUM_SAMPLES, -1), torch.stack(variance_ratios).reshape(NUM_SAMPLES, -1)

def sample_empirical_state_occupancy(traj_length, model_name):
    """
    This function generates NUM_SAMPLES trajectories of length traj_length
    and computes the ratios of the filtering posterior variance to that of the rl agent.
    """

    # load rl policy
    _, policy = load_rl_model(model_name=model_name, device='cpu')  # assume dimensionality equals 1 so the variance is just a scalar

    # set up to create filtering posterior
    dim = 1  # assume dimension is 1 so that variances are scalars
    table = create_dimension_table(torch.tensor([dim]), random=False)

    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    mu_0 = table[dim]['mu_0']
    Q_0 = table[dim]['Q_0']

    # keep state occupancy for each state in traj_length
    state_occupancy = [[] for _ in range(traj_length)]

    for k in range(NUM_SAMPLES):
        # generate a set of ys using the true model parameters
        traj_ys, _ = generate_trajectory(traj_length, A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0)[0:2]
        # create filtering distribution given the ys
        tds = compute_conditional_filtering_posteriors(table=table, num_obs=traj_length, dim=dim, ys=traj_ys)

        for i in range(NUM_VARIANCE_SAMPLES):
            # get first filtering distribution p(x0 | y0:yT)
            td = tds[0].condition(y_values=traj_ys)

            # get first obs
            x0 = torch.zeros(dim)
            obs = torch.cat([x0, traj_ys[0].reshape(1)]).reshape(dim+1, 1)
            xt = torch.tensor(policy.predict(obs, deterministic=False)[0])
            state_occupancy[0].append(xt)

            for j in range(1, len(tds)):
                td_fps = tds[j]
                y = traj_ys[j:]
                # dst = td_fps.condition(y_values=y, x_value=xt)
                # filtering_variance = dst.covariance()
                obs = torch.cat([torch.tensor(xt), y[0].reshape(1)]).reshape(dim+1, 1)

                xt = torch.tensor(policy.predict(obs, deterministic=False)[0])
                state_occupancy[j].append(xt)

    for idx in range(len(state_occupancy)):
        state_occupancy[idx] = torch.cat(state_occupancy[idx])

    return torch.stack(state_occupancy, dim=1)

def sample_filtering_state_occupancy(traj_length):
    """
    This function generates NUM_SAMPLES trajectories of length traj_length
    and computes the ratios of the filtering posterior variance to that of the rl agent.
    """

    # set up to create filtering posterior
    dim = 1  # assume dimension is 1 so that variances are scalars
    table = create_dimension_table(torch.tensor([dim]), random=False)

    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    mu_0 = table[dim]['mu_0']
    Q_0 = table[dim]['Q_0']

    # keep state occupancy for each state in traj_length
    state_occupancy = [[] for _ in range(traj_length)]

    for k in range(NUM_SAMPLES):
        # generate a set of ys using the true model parameters
        traj_ys, _ = generate_trajectory(traj_length, A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0)[0:2]
        # create filtering distribution given the ys
        tds = compute_conditional_filtering_posteriors(table=table, num_obs=traj_length, dim=dim, ys=traj_ys)

        for i in range(NUM_VARIANCE_SAMPLES):
            # get first filtering distribution p(x0 | y0:yT)
            td = tds[0].condition(y_values=traj_ys)

            # get first obs
            x0 = torch.zeros(dim)
            xt = td.sample()
            state_occupancy[0].append(xt)

            for j in range(1, len(tds)):
                td_fps = tds[j]
                y = traj_ys[j:]
                dst = td_fps.condition(y_values=y, x_value=xt)
                xt = td.sample()
                state_occupancy[j].append(xt)

    for idx in range(len(state_occupancy)):
        state_occupancy[idx] = torch.cat(state_occupancy[idx])

    return torch.stack(state_occupancy, dim=1)

def plot_mean_diffs(means, quantiles, traj_length, ent_coef, loss_type, labels):
    basic_plot(datas=means, quantiles=quantiles, traj_length=traj_length, labels=labels,
               xlabel='Trajectory Step (of {})'.format(traj_length), ylabel='Mean Difference',
               title='Diff of Means of FP and RL ({} Coef: {})'.format(loss_type, ent_coef),
               save_path='{}/Difference of Means traj_len: {} ent_coef: {} loss_type: {}.pdf'.format(TODAY, traj_length, ent_coef, loss_type))

def plot_variance_ratios(vrs, quantiles, traj_length, ent_coef, loss_type, labels):
    basic_plot(datas=vrs, quantiles=quantiles, traj_length=traj_length, labels=labels,
               xlabel='Trajectory Step (of {})'.format(traj_length), ylabel='Variance Ratio',
               title='Ratio of Var of FP RL ({} Coef: {})'.format(loss_type, ent_coef),
               save_path='{}/Variance Ratio traj_len: {} ent_coef: {} loss_type: {}.pdf'.format(TODAY, traj_length, ent_coef, loss_type))

def basic_plot(datas, quantiles, traj_length, labels, xlabel, ylabel, title, save_path):
    for data, label in zip(datas, labels):
        data = data.detach()
        lwr, med, upr = torch.quantile(data, quantiles, dim=0)
        x_data = torch.arange(1, med.nelement()+1)
        plt.plot(x_data, med.squeeze(), label=label)
        plt.fill_between(x_data, y1=lwr, y2=upr, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    wandb.save(save_path)
    plt.close()

def execute_variance_ratio_runs(t_len, ent_coef, condition_length):
    labels = [FORWARD_KL, REVERSE_KL]
    forward_model_name = get_model_name(traj_length=traj_length, dim=dim, ent_coef=ent_coef,
                                        loss_type=FORWARD_KL, condition_length=traj_length)
    print('WARNING using wrong model_name in variance ratio')
    reverse_model_name = get_model_name(traj_length=traj_length, dim=dim, ent_coef=ent_coef,
                                        loss_type=REVERSE_KL, condition_length=traj_length)
    print('WARNING using wrong model_name in variance ratio')
    means = []
    vrs = []
    try:
        forward_means, forward_vrs = sample_variance_ratios(traj_length=t_len, model_name=forward_model_name, condition_length=condition_length)
        means.append(forward_means)
        vrs.append(forward_vrs)
    except:
        print('error sampling forward variance ratios')
    try:
        reverse_means, reverse_vrs = sample_variance_ratios(traj_length=t_len, model_name=reverse_model_name, condition_length=condition_length)
        means.append(reverse_means)
        vrs.append(reverse_vrs)
    except:
        print('error sampling reverse variance ratios')
    quantiles = torch.tensor([0.05, 0.5, 0.95])
    plot_mean_diffs(means=means, quantiles=quantiles, traj_length=t_len,
                    ent_coef=ent_coef, loss_type=loss_type, labels=labels)
    plt.figure()
    plot_variance_ratios(vrs=vrs, quantiles=quantiles, traj_length=t_len,
                         ent_coef=ent_coef, loss_type=loss_type, labels=labels)

def execute_state_occupancy(traj_length, ent_coef):
    labels = [FORWARD_KL, REVERSE_KL]
    forward_model_name = get_model_name(traj_length=traj_length, dim=1, ent_coef=ent_coef, loss_type=FORWARD_KL)
    reverse_model_name = get_model_name(traj_length=traj_length, dim=1, ent_coef=ent_coef, loss_type=REVERSE_KL)
    state_occupancies = []
    try:
        forward_state_occupancy = sample_empirical_state_occupancy(traj_length, forward_model_name)
        state_occupancies.append((forward_state_occupancy, 'Forward KL agent'))
    except:
        pass
    try:
        reverse_state_occupancy = sample_empirical_state_occupancy(traj_length, reverse_model_name)
        state_occupancies.append((reverse_state_occupancy, 'Reverse KL agent'))
    except:
        pass
    filtering_state_occupancy = sample_filtering_state_occupancy(traj_length)
    state_occupancies.append((filtering_state_occupancy, 'Filtering Posterior'))
    quantiles = torch.tensor([0.05, 0.5, 0.95], dtype=filtering_state_occupancy.dtype)
    plot_state_occupancy(state_occupancies=state_occupancies,
                         quantiles=quantiles, traj_length=traj_length, ent_coef=ent_coef, loss_type=loss_type,
                         today_dir=TODAY)

def execute_3d_state_occupancy(traj_length, ent_coef):
    forward_model_name = get_model_name(traj_length=traj_length, dim=1, ent_coef=ent_coef, loss_type='forward_kl')
    reverse_model_name = get_model_name(traj_length=traj_length, dim=1, ent_coef=ent_coef, loss_type='reverse_kl')
    state_occupancy_dict = {}
    try:
        forward_state_occupancy = sample_empirical_state_occupancy(traj_length, forward_model_name)
        state_occupancy_dict['Forward KL agent'] = forward_state_occupancy
    except:
        pass
    try:
        reverse_state_occupancy = sample_empirical_state_occupancy(traj_length, reverse_model_name)
        state_occupancy_dict['Reverse KL agent'] = reverse_state_occupancy
    except:
        pass
    filtering_state_occupancy = sample_filtering_state_occupancy(traj_length)
    state_occupancy_dict['Filtering Posterior'] = filtering_state_occupancy
    quantiles = torch.tensor([0.05, 0.5, 0.95], dtype=filtering_state_occupancy.dtype)
    plot_3d_state_occupancy(state_occupancy_dict=state_occupancy_dict,
                            quantiles=quantiles, traj_length=traj_length, ent_coef=ent_coef, loss_type=loss_type,
                            today_dir=TODAY)

def evaluate_agent(linear_gaussian_env_type, traj_length, dim, model_name, condition_length):
    table = create_dimension_table(torch.tensor([dim]), random=False)
    posterior_evidence = compute_evidence(table=table, traj_length=traj_length, dim=dim)
    rl_convergence(linear_gaussian_env_type, table=table, ys=posterior_evidence.ys,
                   truth=posterior_evidence.evidence, dim=dim,
                   model_name=model_name)

def get_full_name_of_ess_type(ess_type):
    if ess_type == 'traj':
        return 'Trajectory Length'
    elif ess_type == 'dim':
        return 'Dimensionality'
    else:
        raise NotImplementedError

def _plot_ess(ess_type):
    xlabel = get_full_name_of_ess_type(ess_type)

    ax = plt.gca()
    ax.xaxis.get_major_locator().set_params(integer=True)

    plt.xlabel('{}'.format(xlabel))
    plt.ylabel('Effective Sample Size')
    plt.title('Effective Sample Size vs. {}'.format(xlabel))
    legend_without_duplicate_labels(plt.gca())
    plt.savefig('{}/ess_{}.pdf'.format(TODAY, ess_type))
    wandb.save('{}/ess_{}.pdf'.format(TODAY, ess_type))
    plt.close()

def plot_ess_from_data_from_files(filenames, data_type):
    assert filenames
    for j, filename in enumerate(filenames):
        data_with_columns = load_ess_data(filename, data_type)
        plot_ess_data(data_with_columns, i=j+1)
        ess_type = data_with_columns.data_type
    _plot_ess(ess_type)

def plot_ess_from_data(directory, data_type, initial_idx=0):
    for j, fil in enumerate(os.listdir(os.fsencode(directory))):
        filename = os.fsdecode(fil)
        if filename.endswith('ESS.csv'):
            data_with_columns = load_ess_data(directory+filename, data_type)
            plot_ess_data(data_with_columns, i=j+1+initial_idx)
            ess_type = data_with_columns.data_type
    _plot_ess(ess_type)

def plot_ess_from_dir_partial_data(directory, data_type, initial_idx=0):
    data = None
    for j, fil in enumerate(os.listdir(os.fsencode(directory))):
        filename = os.fsdecode(fil)
        if 'ess' in filename.lower() and filename.endswith('.csv'):
            data_with_columns = load_ess_data(directory+filename, data_type)
            data_label = data_with_columns.data_label
            ess_type = data_with_columns.data_type
            data = data_with_columns.data
            quantiles = torch.tensor([0.05, 0.5, 0.95], dtype=data.dtype)
            lower_ci, med, upper_ci = torch.quantile(data, quantiles, dim=0)

            df = pd.read_csv('{}/{}'.format(directory, filename), index_col=0)
            x_vals = df.columns.astype(np.int)

            plt.plot(x_vals, med.squeeze(), label=data_label)
            plt.fill_between(x_vals, y1=lower_ci, y2=upper_ci, alpha=0.3)

    xlabel = get_full_name_of_ess_type(ess_type)
    ax = plt.gca()
    ax.xaxis.get_major_locator().set_params(integer=True)
    plt.xlabel('{}'.format(xlabel))
    plt.ylabel('Effective Sample Size')
    plt.title('Effective Sample Size vs. {}'.format(xlabel))
    legend_without_duplicate_labels(plt.gca())
    plt.savefig('{}/ess_{}.pdf'.format(TODAY, ess_type))
    wandb.save('{}/ess_{}.pdf'.format(TODAY, ess_type))
    plt.close()

def plot_ess_from_partial_data(filenames, data_type):
    assert filenames
    data = None
    for j, filename in enumerate(filenames):
        data_with_columns = load_ess_data(filename, data_type)
        ess_type = data_with_columns.data_type
        data_label = data_with_columns.data_label
        if data is None:
            data = data_with_columns.data
        else:
            data = torch.cat([data, data_with_columns.data], dim=1)
    quantiles = torch.tensor([0.05, 0.5, 0.95], dtype=data.dtype)
    lower_ci, med, upper_ci = torch.quantile(data, quantiles, dim=0)

    x_vals = torch.arange(1, med.nelement()+1)

    plt.plot(x_vals, med.squeeze(), label=data_label)
    # plt.fill_between(x_vals, y1=lower_ci, y2=upper_ci, alpha=0.3)

    xlabel = get_full_name_of_ess_type(ess_type)
    ax = plt.gca()
    ax.xaxis.get_major_locator().set_params(integer=True)
    plt.xlabel('{}'.format(xlabel))
    plt.ylabel('Effective Sample Size')
    plt.title('Effective Sample Size vs. {}'.format(xlabel))
    legend_without_duplicate_labels(plt.gca())
    plt.savefig('{}/ess_{}.pdf'.format(TODAY, ess_type))
    wandb.save('{}/ess_{}.pdf'.format(TODAY, ess_type))
    plt.close()

def get_env_type_from_arg(env_type_arg, condition_length=0):
    if env_type_arg == 'AllObservationsLinearGaussianEnv':
        return lambda **kwargs: AllObservationsLinearGaussianEnv(**kwargs, condition_length=condition_length)
    elif env_type_arg == 'LinearGaussianEnv':
        return LinearGaussianEnv
    elif env_type_arg == 'ConditionalObservationsLinearGaussianEnv':
        return lambda **kwargs: ConditionalObservationsLinearGaussianEnv(**kwargs, condition_length=condition_length)
    elif env_type_arg == 'EnsembleLinearGaussianEnv':
        return lambda **kwargs: EnsembleLinearGaussianEnv(**kwargs, condition_length=condition_length)

def compute_conditional_kl(td_fps, policy, prev_xt, ys, condition_length):
    # filtering dist
    dst = td_fps.condition(y_values=ys, x_value=prev_xt)
    filtering_dist = dst.get_dist()

    # policy dist
    obs = torch.cat([prev_xt, ys[0:condition_length]]).reshape(1, -1)
    pd = policy.get_distribution(obs).distribution

    covs = []
    for i in range(pd.scale.shape[0]):
        covs.append(torch.diag(pd.scale[i, :]))
    policy_dist = dist.MultivariateNormal(pd.mean, torch.stack(covs))

    return dist.kl_divergence(filtering_dist, policy_dist)

def compute_kl_w_prior(policy, prev_xt, ys, condition_length, A, Q):
    # create prior
    prior = get_state_transition_dist(prev_xt, A, Q)

    # policy dist
    obs = torch.cat([prev_xt, ys[0:condition_length]]).reshape(1, -1)
    pd = policy.get_distribution(obs).distribution

    covs = []
    for i in range(pd.scale.shape[0]):
        covs.append(torch.diag(pd.scale[i, :]))
    policy_dist = dist.MultivariateNormal(pd.mean, torch.stack(covs))

    return dist.kl_divergence(prior, policy_dist)

def compute_SAC_kl_w_prior(sac_policy, prev_xt, ys, condition_length, A, Q, num_samples):
    # create prior
    prior = get_state_transition_dist(prev_xt, A, Q)

    # sac_policy dist
    obs = torch.cat([prev_xt, ys[0:condition_length]]).reshape(1, -1)

    kl = torch.tensor([0.])
    for _ in range(num_samples):
        np_sac_action = sac_policy.predict(obs.t(), deterministic=False)[0]
        sac_action = torch.tensor(np_sac_action).reshape(1, -1)
        mean_actions, log_std, _ = sac_policy.actor.get_action_dist_params(obs)
        q_score = sac_policy.actor.action_dist.proba_distribution(mean_actions, log_std).log_prob(sac_action)
        action = torch.atanh(sac_action)
        p_score = prior.log_prob(action)
        kl += (q_score - p_score) / num_samples

    return kl

def dispatch_kl_with_prior(policy, prev_xt, ys, condition_length, A, Q, sac_num_samples):
    if isinstance(policy, FKLPolicy) or isinstance(policy, SACPolicy):
        return compute_SAC_kl_w_prior(policy, prev_xt, ys, condition_length, A, Q, sac_num_samples)
    return compute_kl_w_prior(policy, prev_xt, ys, condition_length, A, Q)

def compute_policy_kl_at_each_state(td_fps, policy, conditional_fps, prev_xts, ys, condition_length, dim):
    kls = []
    for i in range(len(ys)):
        prev_xt = prev_xts[i]

        td = td_fps[i].condition(y_values=ys[i:], x_value=prev_xt).get_dist()

        if len(ys[i:i+condition_length]) == condition_length:
            # construct observation
            if prev_xt is None:
                assert i==0
                obs = torch.cat([torch.zeros(dim), ys[i:i+condition_length]]).reshape(1, -1)
            else:
                obs = torch.cat([prev_xt, ys[i:i+condition_length]]).reshape(1, -1)

            # policy dist
            pd = policy.get_distribution(obs).distribution
            covs = []
            for i in range(pd.scale.shape[0]):
                covs.append(torch.diag(pd.scale[i, :]))
            conditional_dist = dist.MultivariateNormal(pd.mean, torch.stack(covs))
        else:
            # conditional smoothing dist
            # Note that this branch is taken only when there are fewer
            # than condition_length ys after index i
            conditional_dist = conditional_fps[i].condition(y_values=ys[i:i+condition_length], x_value=prev_xt).get_dist()

        kls.append(dist.kl_divergence(td, conditional_dist).detach())
    return kls

def compute_analytical_kl_at_each_state(td_fps, conditional_fps, prev_xts, ys, condition_length):
    kls = []

    for i in range(len(ys)):
        td = td_fps[i].condition(y_values=ys[i:], x_value=prev_xts[i]).get_dist()

        conditional_td = conditional_fps[i].condition(y_values=ys[i:i+condition_length], x_value=prev_xts[i]).get_dist()

        kls.append(dist.kl_divergence(td, conditional_td))
    return kls

def execute_evaluate_agent_until(linear_gaussian_env_type, traj_lengths, dim, loss_type, ent_coef, delta, condition_length, use_mlp_policy):
    table = create_dimension_table(torch.tensor([dim]), random=False)
    num_samples_data = []
    for traj_length in traj_lengths:
        posterior_evidence = compute_evidence(table=table, traj_length=traj_length, dim=dim)
        rl_output = ImportanceOutput(traj_length=traj_length, ys=posterior_evidence.ys, dim=dim)
        name = '{}(m={})'.format(RL_DISTRIBUTION, condition_length)
        for _ in range(NUM_REPEATS):
            model_name = get_model_name(traj_length=traj_length, dim=dim,
                                        ent_coef=ent_coef, loss_type=loss_type,
                                        condition_length=condition_length, 
                                        use_mlp_policy=use_mlp_policy)
            # model_name = 'agents/0.1_forward_kl_linear_gaussian_model_(traj_10_dim_1_condition_length_0).zip'
            eval_obj = evaluate_agent_until(posterior_evidence, linear_gaussian_env_type, traj_length=traj_length,
                                            dim=dim, model_name=model_name,
                                            using_entropy_loss=(loss_type == ENTROPY_LOSS), delta=delta)

            # add rl estimator
            rl_estimator = rl_output.add_rl_estimator(running_log_estimates=eval_obj.running_log_estimates,
                                                      ci=eval_obj.ci, weight_mean=eval_obj.log_weight_mean.exp(),
                                                      max_weight_prop=eval_obj.log_max_weight_prop.exp(),
                                                      ess=eval_obj.log_effective_sample_size.exp(),
                                                      ess_ci=eval_obj.ess_ci, idstr=name)
        rl_estimator.save_data()
        num_samples_data.append([len(repeat) for repeat in rl_estimator.running_log_estimate_repeats])

    # save data
    try:
        estimates_df = pd.DataFrame(torch.stack(num_samples_data).numpy())
        estimates_df.to_csv('{}/{}_EvaluateUntilData.csv'.format(SAVE_DIR, loss_type))
    except:
        pass

    avg_num_samples = [np.mean(traj_data) for traj_data in num_samples_data]
    # plot data
    plt.plot(traj_lengths, avg_num_samples, label=loss_type)
    legend_without_duplicate_labels(plt.gca())
    # plot confidence intervals
    quantiles = torch.tensor([0.05, 0.5, 0.95])
    samples_data = torch.stack([torch.tensor(data, dtype=quantiles.dtype) for data in num_samples_data])
    lower_ci, med, upper_ci = torch.quantile(samples_data, quantiles, dim=0)
    plt.fill_between(traj_lengths, y1=lower_ci, y2=upper_ci, alpha=0.3)

    plt.xlabel('Trajectory Length')
    plt.ylabel('Num Samples')
    plt.title('Num Samples Required for |truth / estimate| - 1 < {}'.format(delta))

    save_path = '{}/ent_coef_{}_loss_type_{}_dim_{}EvalUntilSampleSize.pdf'.format(rl_estimator.save_dir, ent_coef, loss_type, dim)
    plt.savefig(save_path)
    wandb.save(save_path)

def generate_rewards(traj_length, dim, loss_type):
    table = create_dimension_table(torch.tensor([dim]), random=False)
    posterior_evidence = compute_evidence(table=table, traj_length=traj_length, dim=dim)
    ys = posterior_evidence.ys

    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    mu_0 = table[dim]['mu_0']
    Q_0 = table[dim]['Q_0']

    # condition_length for this env doesn't matter; I'm just using this AllObsEnv so that I get rewards
    # only due to the *likelihood*
    env = AllObservationsLinearGaussianEnv(A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0, condition_length=0,
                                           using_entropy_loss=True, ys=ys, sample=True)
    env.reset()
    all_tds = []
    for m in range(traj_length):
        all_tds.append(compute_conditional_filtering_posteriors(table=table, num_obs=len(posterior_evidence.ys),
                                                                dim=dim, m=m, condition_on_x=True, ys=ys))
    all_rewards = []
    labels = []
    for m in range(traj_length):
        td_fps = all_tds[m]
        condition_length = m if m > 0 else traj_length
        labels.append('m={}'.format(condition_length))
        td = all_tds[m][0].condition(y_values=ys[0:condition_length])
        xt = td.mean()
        _, reward, done, _ = env.step(xt)
        all_rewards.append([reward])
        for j in range(1, traj_length):
            y = ys[j:j+condition_length]
            td = td_fps[j]
            dst = td.condition(y_values=y, x_value=xt)
            xt = dst.mean()
            _, reward, done, _ = env.step(xt)
            all_rewards[m].append(reward)
        all_rewards[m] = torch.tensor(all_rewards[m])
        env.reset()

    return all_rewards, labels, ys

def smoothing_reward(traj_length, dim, loss_type):
    torch.manual_seed(10)
    avg_rewards = []
    for _ in range(NUM_SAMPLES):
        all_rewards, labels, _ = generate_rewards(traj_length, dim, loss_type)
        if not avg_rewards:
            for m in range(len(all_rewards)):
                avg_rewards.append(all_rewards[m] / NUM_SAMPLES)
        else:
            for m in range(len(avg_rewards)):
                avg_rewards[m] += all_rewards[m] / NUM_SAMPLES
    linestyles = ['solid'] * len(labels)
    plot_smoothing_reward(torch.arange(1, traj_length+1), avg_rewards, labels, linestyles)

def plot_smoothing_reward(xlabels, all_rewards, labels, linestyles):
    for all_reward, label, linestyle in zip(all_rewards, labels, linestyles):
        plt.plot(xlabels, all_reward, label=label, linestyle=linestyle)
        plt.xlabel('State')
        plt.ylabel('Reward')
        plt.title('Deterministic Reward at Each State')
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -1.3), ncol=int(len(labels)/2))
        plt.savefig('{}/(traj_length_{})DeterministicRewardPlot.pdf'.format(TODAY, traj_length))
        wandb.save('{}/(traj_length_{})DeterministicRewardPlot.pdf'.format(TODAY, traj_length))

def policy_reward(traj_length, dim, linear_gaussian_env_type, model_name, ys, condition_length):
    # evaluate rl policy
    _, policy = load_rl_model(model_name=model_name, device='cpu')

    table = create_dimension_table(torch.tensor([dim]), random=False)
    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    mu_0 = table[dim]['mu_0']
    Q_0 = table[dim]['Q_0']

    env = linear_gaussian_env_type(A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0,
                                   using_entropy_loss=(loss_type==ENTROPY_LOSS),
                                   ys=ys, sample=True)
    eval_obj_rl = evaluate(d=policy, N=1, env=env, deterministic=True)
    label = 'rl m={}'.format(condition_length)
    return torch.tensor(eval_obj_rl.liks), label

def compare_single_trajectory_rewards(traj_length, dim, loss_type, linear_gaussian_env_type, model_name, rl_condition_length):
    all_rewards, labels, ys = generate_rewards(traj_length, dim, loss_type)
    linestyles = ['solid'] * len(labels)

    liks, label = policy_reward(traj_length, dim, linear_gaussian_env_type, model_name, ys, rl_condition_length)

    all_rewards.append(liks)
    labels.append(label)
    linestyles.append('dashed')

    plot_smoothing_reward(torch.arange(1, traj_length+1), all_rewards, labels, linestyles)
    plt.close()

def compare_rewards(traj_length, dim, loss_type, linear_gaussian_env_type, model_name, rl_condition_length):
    torch.manual_seed(10)
    avg_rewards = []
    for _ in range(NUM_SAMPLES):
        all_rewards, labels, ys = generate_rewards(traj_length, dim, loss_type)
        liks, rl_label = policy_reward(traj_length, dim, linear_gaussian_env_type, model_name, ys, rl_condition_length)

        if not avg_rewards:
            for m in range(len(all_rewards)):
                avg_rewards.append(all_rewards[m] / NUM_SAMPLES)
            avg_rewards.append(liks / NUM_SAMPLES)
        else:
            for m in range(len(all_rewards)):
                avg_rewards[m] += all_rewards[m] / NUM_SAMPLES
            avg_rewards[-1] += liks / NUM_SAMPLES

    linestyles = ['solid'] * len(labels) + ['dashed']
    labels.append(rl_label)
    plot_smoothing_reward(torch.arange(1, traj_length+1), avg_rewards, labels, linestyles)

def plot_posterior_variance(table, traj_length, dim, condition_length, true_variances=None, save_dir=None):
    posterior_evidence = compute_evidence(table=table, traj_length=traj_length, dim=dim)
    ys = posterior_evidence.ys

    tds = compute_conditional_filtering_posteriors(table=table, num_obs=len(posterior_evidence.ys),
                                                   dim=dim, m=condition_length, condition_on_x=True,
                                                   ys=ys)

    m = condition_length if condition_length > 0 else traj_length

    td = tds[0].condition(y_values=ys[0:m])
    variances = [td.covariance()]
    xt = td.sample()
    last_state_of_constant_variance = 1
    if true_variances:
        first_state_of_correct_variance = 2 if torch.abs(variances[0] - true_variances[0]) > 0.001 else 1
    for j in range(1, len(tds)):
        td_fps = tds[j]
        y = ys[j:j+m]
        dst = td_fps.condition(y_values=y, x_value=xt)
        xt = dst.sample()
        variance = dst.covariance()

        # find where states have constant variance
        if torch.abs(variance - variances[-1]) < 0.0001 and last_state_of_constant_variance == j:
            last_state_of_constant_variance = j+1
        variances.append(variance)

        # find where variances are equal to the true smoothing posterior
        if true_variances and torch.abs(variance - true_variances[j]) > 0.0001:
            first_state_of_correct_variance = j+2

    x_values = torch.arange(1, traj_length+1)
    plt.plot(x_values, variances)
    # plt.scatter(x=last_state_of_constant_variance, y=variances[last_state_of_constant_variance-1],
    #             label='This state and prior have constant variance', color='r')
    if true_variances:
        plt.scatter(x=first_state_of_correct_variance, y=variances[first_state_of_correct_variance-1],
                    label='This state and after coincide with true smoothing posterior.', color='g')
        plt.legend()
    plt.xlabel('States')
    plt.ylabel('Variance')
    ax = plt.gca()
    ax.set_xticks(x_values)
    plt.title('Truncated (m={}) Smoothing Posterior Variance At Each State'.format(condition_length))

    A = table[dim]['A'].item()
    Q = table[dim]['Q'].item()
    C = table[dim]['C'].item()
    R = table[dim]['R'].item()

    if save_dir:
        os.makedirs('{}/{}'.format(TODAY, save_dir), exist_ok=True)
        plt.savefig('{}/{}/TruncatedSmoothingPosteriorVariance(m={} traj_length={} A={} Q={} C={} R={}).pdf'.format(TODAY, save_dir, condition_length, traj_length, A, Q, C, R))
        wandb.save('{}/{}/TruncatedSmoothingPosteriorVariance(m={} traj_length={} A={} Q={} C={} R={}).pdf'.format(TODAY, save_dir, condition_length, traj_length, A, Q, C, R))
    else:
        plt.savefig('{}/TruncatedSmoothingPosteriorVariance(m={} traj_length={} A={} Q={} C={} R={}).pdf'.format(TODAY, condition_length, traj_length, A, Q, C, R))
        wandb.save('{}/TruncatedSmoothingPosteriorVariance(m={} traj_length={} A={} Q={} C={} R={}).pdf'.format(TODAY, condition_length, traj_length, A, Q, C, R))
    plt.close()

    return variances

def plot_posterior_mean(ys, traj_length, dim, condition_length, xs=None):
    # torch.manual_seed(10)
    tds = compute_conditional_filtering_posteriors(table=table, num_obs=len(posterior_evidence.ys),
                                                   dim=dim, m=condition_length, condition_on_x=True,
                                                   ys=ys)

    m = condition_length if condition_length > 0 else traj_length

    td = tds[0].condition(y_values=ys[0:m])
    means = [td.mean()]
    empty_xs = xs is None
    if empty_xs:
        xs = [td.sample()]
    for j in range(1, len(tds)):
        td_fps = tds[j]
        y = ys[j:j+m]
        dst = td_fps.condition(y_values=y, x_value=xs[j-1])
        if empty_xs:
            xs.append(dst.sample())
        means.append(dst.mean())

    print(means)

    x_values = torch.arange(1, traj_length+1)
    plt.plot(x_values, means)
    plt.xlabel('States')
    plt.ylabel('Mean')
    plt.legend()
    ax = plt.gca()
    ax.set_xticks(x_values)
    plt.title('Truncated (m={}) Smoothing Posterior Mean At Each State'.format(condition_length))

    A = table[dim]['A'].item()
    Q = table[dim]['Q'].item()
    C = table[dim]['C'].item()
    R = table[dim]['R'].item()

    plt.savefig('{}/TruncatedSmoothingPosteriorMean(m={} traj_length={} A={} Q={} C={} R={}).pdf'.format(TODAY, condition_length, traj_length, A, Q, C, R))
    wandb.save('{}/TruncatedSmoothingPosteriorMean(m={} traj_length={} A={} Q={} C={} R={}).pdf'.format(TODAY, condition_length, traj_length, A, Q, C, R))
    plt.close()

    return means, xs

def execute_pure_rl_ensemble(traj_lengths, dim, ent_coef, condition_length, use_mlp_policy):
    table = create_dimension_table(torch.tensor([dim]), random=False)
    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    Q_0 = table[dim]['Q_0']
    mu_0 = table[dim]['mu_0']

    outputs = []
    rl_ds = []
    _, policy = load_rl_model(model_name='/opt/agents/0.1_reverse_kl_linear_gaussian_model_(traj_10_dim_1_condition_length_5_use_mlp_policy_True).zip', device='cpu')
    rl_ds.append(policy)
    _, policy = load_rl_model(model_name='/opt/agents/0.1_reverse_kl_linear_gaussian_model_(traj_10_dim_1_condition_length_4_use_mlp_policy_True).zip', device='cpu')
    rl_ds.append(policy)
    _, policy = load_rl_model(model_name='/opt/agents/0.1_reverse_kl_linear_gaussian_model_(traj_10_dim_1_condition_length_3_use_mlp_policy_True).zip', device='cpu')
    rl_ds.append(policy)
    _, policy = load_rl_model(model_name='/opt/agents/0.1_reverse_kl_linear_gaussian_model_(traj_10_dim_1_condition_length_2_use_mlp_policy_True).zip', device='cpu')
    rl_ds.append(policy)
    _, policy = load_rl_model(model_name='/opt/agents/0.1_reverse_kl_linear_gaussian_model_(traj_10_dim_1).zip', device='cpu')
    rl_ds.append(policy)

    for traj_length in traj_lengths:
        # rl_ds = []
        # for m in torch.arange(condition_length, 0, -1):
        #     model_name = get_model_name(traj_length=traj_length, dim=dim,
        #                                 ent_coef=ent_coef, loss_type=loss_type,
        #                                 condition_length=m, use_mlp_policy=use_mlp_policy)

        #     _, policy = load_rl_model(model_name=model_name, device='cpu')
        #     rl_ds.append(policy)

        name = '{}(traj_len {} dim {})'.format(RL_DISTRIBUTION, traj_length, dim)
        rl_output = ImportanceOutput(traj_length=traj_length, ys=None, dim=dim)
        for _ in range(NUM_REPEATS):
            ys = generate_trajectory(traj_length, A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0)[0]
            env = linear_gaussian_env_type(A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0,
                                           using_entropy_loss=(loss_type==ENTROPY_LOSS),
                                           ys=ys, sample=False)

            eval_obj = evaluate_pure_rl_ensemble(rl_ds=rl_ds, N=NUM_SAMPLES, env=env,
                                                 condition_length=condition_length, deterministic=False)

            # add rl estimator
            rl_estimator = rl_output.add_rl_estimator(running_log_estimates=eval_obj.running_log_estimates,
                                                      ci=eval_obj.ci, weight_mean=eval_obj.log_weight_mean.exp(),
                                                      max_weight_prop=eval_obj.log_max_weight_prop.exp(),
                                                      ess=eval_obj.log_effective_sample_size.exp(),
                                                      ess_ci=eval_obj.ess_ci, idstr=name)
        rl_estimator.save_data()

        outputs += [OutputWithName(rl_output, name)]

    save_outputs_with_names_traj(outputs, RL_DISTRIBUTION, 'pure_rl_ensemble')
    make_ess_plot_nice(outputs, fixed_feature_string='dimension', fixed_feature=dim,
                       num_samples=NUM_SAMPLES, num_repeats=NUM_REPEATS, traj_lengths=traj_lengths,
                       xlabel='Trajectory Length', distribution_type=RL_DISTRIBUTION, name='RL ensemble')

def execute_rl_posterior_ensemble(traj_lengths, dim, condition_length, model_name):
    table = create_dimension_table(torch.tensor([dim]), random=False)
    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    Q_0 = table[dim]['Q_0']
    mu_0 = table[dim]['mu_0']

    _, policy = load_rl_model(model_name=model_name, device='cpu')

    outputs = []
    for traj_length in traj_lengths:
        name = '{}(traj_len {} dim {})'.format(RL_DISTRIBUTION, traj_length, dim)
        rl_output = ImportanceOutput(traj_length=traj_length, ys=None, dim=dim)
        for _ in range(NUM_REPEATS):
            ys = generate_trajectory(traj_length, A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0)[0]
            env = linear_gaussian_env_type(A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0,
                                           using_entropy_loss=(loss_type==ENTROPY_LOSS),
                                           ys=ys, sample=False)

            # create filtering distribution given the ys
            _tds = compute_conditional_filtering_posteriors(table=table, num_obs=traj_length, dim=dim, ys=env.ys)
            tds = _tds[traj_length-condition_length+1:]

            eval_obj = evaluate_rl_posterior_ensemble(rl_d=policy, tds=tds, N=NUM_SAMPLES, env=env,
                                                    condition_length=condition_length, deterministic=False)

            # add rl estimator
            rl_estimator = rl_output.add_rl_estimator(running_log_estimates=eval_obj.running_log_estimates,
                                                    ci=eval_obj.ci, weight_mean=eval_obj.log_weight_mean.exp(),
                                                    max_weight_prop=eval_obj.log_max_weight_prop.exp(),
                                                    ess=eval_obj.log_effective_sample_size.exp(),
                                                    ess_ci=eval_obj.ess_ci, idstr=name)
        rl_estimator.save_data()

        outputs += [OutputWithName(rl_output, name)]

    save_outputs_with_names_traj(outputs, RL_DISTRIBUTION, 'rl_posterior_ensemble')
    make_ess_plot_nice(outputs, fixed_feature_string='dimension', fixed_feature=dim,
                       num_samples=NUM_SAMPLES, num_repeats=NUM_REPEATS, traj_lengths=traj_lengths,
                       xlabel='Trajectory Length', distribution_type=RL_DISTRIBUTION, name='RL+posterior ensemble')

def get_smoothing_and_conditional_posteriors(traj_length, dim, condition_length):
    table = create_dimension_table(torch.tensor([dim]), random=False)

    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    Q_0 = table[dim]['Q_0']
    mu_0 = table[dim]['mu_0']

    ys, xs = generate_trajectory(traj_length, A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0)[0:2]

    tds = compute_conditional_filtering_posteriors(table=table, num_obs=traj_length, dim=dim, ys=ys)

    conditional_tds = compute_conditional_filtering_posteriors(table=table, num_obs=traj_length, dim=dim, m=condition_length, ys=ys)

    return tds, conditional_tds, ys, xs, table

def execute_compute_analytical_kl_at_each_state(traj_length, dim, condition_length):
    tds, conditional_tds, ys, xs, table = get_smoothing_and_conditional_posteriors(traj_length, dim, condition_length)

    # get all of the "previous xs" for the conditional distribution
    # Note that the first xt is None since we want p(x_0|y_{0:T}) without condition on an x
    # Also note that len(xs) = traj_length + 1 because an extra xt is generated at the end which should be ignored
    # Therefore the previous xts that we want are None, as well as all of the xts used to generate a subsequent xt for scoring
    prev_xts = [None] + [x.reshape(1) for x in xs[0:-2]]
    kls = compute_analytical_kl_at_each_state(tds, conditional_tds, prev_xts, ys, condition_length)

def execute_compute_policy_kl_at_each_state(traj_length, dim, condition_length, model_name):
    tds, conditional_tds, ys, xs, _ = get_smoothing_and_conditional_posteriors(traj_length, dim, condition_length)

    _, policy = load_rl_model(model_name=model_name, device='cpu')

    # get all of the "previous xs" for the conditional distribution
    # Note that the first xt is None since we want p(x_0|y_{0:T}) without condition on an x
    # Also note that len(xs) = traj_length + 1 because an extra xt is generated at the end which should be ignored
    # Therefore the previous xts that we want are None, as well as all of the xts used to generate a subsequent xt for scoring
    prev_xts = [None] + [x.reshape(1) for x in xs[0:-2]]
    kls = compute_policy_kl_at_each_state(tds, policy, conditional_tds, prev_xts, ys, condition_length, dim)

def find_num_samples_for_condition_length(traj_length, dim, condition_length, delta=0.1):
    table = create_dimension_table(torch.tensor([dim]), random=False)

    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    Q_0 = table[dim]['Q_0']
    mu_0 = table[dim]['mu_0']

    posterior_evidence = compute_evidence(table=table, traj_length=traj_length, dim=dim)

    ys = posterior_evidence.ys
    env = posterior_evidence.env

    num_samples = []
    for _ in range(NUM_REPEATS):
        fps = compute_conditional_filtering_posteriors(table=table, num_obs=traj_length,
                                                       dim=dim, m=condition_length, condition_on_x=True,
                                                       ys=posterior_evidence.ys)

        num_samples.append(evaluate_smoothing_posterior_until(ys=ys, tds=fps, truth=posterior_evidence.evidence,
                                                              delta=delta, env=env, m=condition_length))
    print('repeating {} times, it took this many samples: {}'.format(NUM_REPEATS, num_samples))
    return torch.tensor(num_samples, dtype=torch.float)

def find_num_samples_for_condition_lengths(traj_length, dim, condition_lengths):
    delta = 0.03
    meds = []
    lowers = []
    uppers = []
    for condition_length in condition_lengths:
        print('finding num samples for condition length: {}'.format(condition_length))
        num_samples = find_num_samples_for_condition_length(traj_length, dim, condition_length, delta=delta)
        quantiles = torch.tensor([0.05, 0.5, 0.95])
        lower_ci, med, upper_ci = torch.quantile(num_samples, quantiles, dim=0)
        lowers.append(lower_ci)
        meds.append(med.squeeze())
        uppers.append(upper_ci)
    plt.plot(condition_lengths, meds)
    plt.fill_between(condition_lengths, y1=lowers, y2=uppers, alpha=0.3)
    plt.xlabel('Condition Lengths')
    plt.ylabel('Num Samples')
    plt.legend()
    ax = plt.gca()
    ax.set_xticks(condition_lengths)
    plt.title('Num Samples for Evidence Estimate to be Within {} of Truth'.format(delta))
    plt.savefig('{}/NumSamples(traj_length={}).pdf'.format(TODAY, traj_length))
    wandb.save('{}/NumSamples(traj_length={}).pdf'.format(TODAY, traj_length))
    plt.close()

def find_num_samples_for_traj_lengths(traj_lengths, dim, condition_length):
    delta = 0.03
    meds = []
    lowers = []
    uppers = []
    for traj_length in traj_lengths:
        print('finding num samples for traj length: {}'.format(traj_length))
        num_samples = find_num_samples_for_condition_length(traj_length, dim, condition_length, delta=delta)
        quantiles = torch.tensor([0.05, 0.5, 0.95])
        lower_ci, med, upper_ci = torch.quantile(num_samples, quantiles, dim=0)
        lowers.append(lower_ci)
        meds.append(med.squeeze())
        uppers.append(upper_ci)
    plt.plot(traj_lengths, meds)
    plt.fill_between(traj_lengths, y1=lowers, y2=uppers, alpha=0.3)
    plt.xlabel('Trajectory Lengths')
    plt.ylabel('Num Samples')
    plt.legend()
    ax = plt.gca()
    ax.set_xticks(traj_lengths)
    plt.title('Num Samples for Evidence Estimate to be Within {} of Truth'.format(delta))
    plt.savefig('{}/NumSamples(condition_length={}).pdf'.format(TODAY, condition_length))
    wandb.save('{}/NumSamples(condition_length={}).pdf'.format(TODAY, condition_length))
    plt.close()

def vi_variance_ratio(args, policy):
    forward_means, forward_vrs = sample_variance_ratios(traj_length=args.traj_length, model_name='',
                                                        condition_length=args.traj_length, policy=policy)
    means = [forward_means]
    vrs = [forward_vrs]
    quantiles = torch.tensor([0.05, 0.5, 0.95])
    labels = [VI_DISTRIBUTION]
    plot_mean_diffs(means=means, quantiles=quantiles, traj_length=args.traj_length,
                    ent_coef=args.ent_coef, loss_type=args.loss_type, labels=labels)
    plt.figure()
    plot_variance_ratios(vrs=vrs, quantiles=quantiles, traj_length=args.traj_length,
                         ent_coef=args.ent_coef, loss_type=args.loss_type, labels=labels)

def evaluate_vi_policy(vlgm, model_name, traj_length):
    policy = vlgm.extract_policy()

    rl_output = ImportanceOutput(traj_length=traj_length, ys=None, dim=dim)

    print('Q: {}'.format(vlgm.args.Q))
    env = EnsembleLinearGaussianEnv(A=vlgm.args.A, Q=vlgm.args.Q, C=vlgm.args.C, R=vlgm.args.R,
                                    mu_0=vlgm.args.mu_0, Q_0=vlgm.args.Q, condition_length=vlgm.args.condition_length,
                                    using_entropy_loss=False,
                                    ys=None, traj_length=vlgm.args.traj_length, sample=True)

    for _ in range(vlgm.args.num_repeats):
        eval_obj = evaluate(d=policy, N=vlgm.args.num_samples, env=env)

        # add rl confidence interval
        rl_estimator = rl_output.add_rl_estimator(running_log_estimates=eval_obj.running_log_estimates,
                                                  ci=eval_obj.ci, weight_mean=eval_obj.log_weight_mean.exp(),
                                                  max_weight_prop=eval_obj.log_max_weight_prop.exp(),
                                                  ess=eval_obj.log_effective_sample_size.exp(),
                                                  ess_ci=eval_obj.ess_ci, idstr=model_name)
    rl_estimator.save_data()
    return OutputWithName(rl_output, model_name)

def vi_ess_traj(args):
    distribution_type = VI_DISTRIBUTION
    os.makedirs('{}/{}'.format(TODAY, distribution_type), exist_ok=True)
    outputs = []
    for traj_length in args.ess_traj_lengths:
        # create new args object to create a new VLGM
        new_args = deepcopy(args)
        new_args.traj_length = traj_length

        # load VLGM for this traj_length
        vlgm = get_vlgm(new_args)
        vlgm.load_models()
        model_name = 'VariationalInference'
        outputs += [evaluate_vi_policy(vlgm, model_name, traj_length)]
    save_outputs_with_names_traj(outputs, distribution_type,
                                 '{}(traj_lengths_{}_dim_{})'.format(distribution_type, args.ess_traj_lengths, args.dim))
    make_ess_plot_nice(outputs, fixed_feature_string='dimension', fixed_feature=args.dim,
                       num_samples=args.num_samples, num_repeats=args.num_repeats, traj_lengths=args.ess_traj_lengths,
                       xlabel='Trajectory Length', distribution_type=distribution_type, name='VariationalInference')


if __name__ == "__main__":
    args, _ = get_args()
    subroutine = args.subroutine
    SAVE_DIR = args.save_dir
    continue_training = args.continue_training

    # MODEL = 'agents/'+save_dir+'/{}_{}_linear_gaussian_model_(traj_{}_dim_{})'
    if subroutine != 'train_agent':
        run = wandb.init(project='linear_gaussian_model', save_code=True, entity='iai')
        os.makedirs(SAVE_DIR, exist_ok=True)

    os.makedirs('agents/', exist_ok=True)
    os.makedirs(TODAY, exist_ok=True)

    traj_length = args.traj_length
    dim = args.dim
    ent_coef = args.ent_coef
    loss_type = args.loss_type
    filenames = args.filenames
    ess_dir = args.ess_dir
    initial_idx = args.initial_idx
    data_type = args.data_type
    ess_dims = args.ess_dims
    ess_traj_lengths = args.ess_traj_lengths
    ess_condition_lengths = args.ess_condition_lengths
    condition_length = args.condition_length if args.condition_length > 0 else 0
    linear_gaussian_env_type = get_env_type_from_arg(args.env_type, condition_length=condition_length)

    use_mlp_policy = args.use_mlp_policy
    rl_type = get_rl_type_from_str(args.rl_type)
    model_name = get_model_name(traj_length=traj_length, dim=dim,
                                ent_coef=ent_coef, loss_type=loss_type,
                                condition_length=condition_length, 
                                use_mlp_policy=use_mlp_policy,
                                rl_type=rl_type)

    learning_rate = args.learning_rate
    clip_range = args.clip_range

    NUM_SAMPLES = args.num_samples
    RL_TIMESTEPS = args.rl_timesteps
    delta = args.delta
    ignore_reward = args.ignore_reward

    if subroutine == 'train_agent':
        print('executing: {}'.format('train_agent'))
        test_train(traj_length=traj_length, dim=dim, condition_length=condition_length,
                   ent_coef=ent_coef, loss_type=loss_type, learning_rate=learning_rate,
                   clip_range=clip_range, linear_gaussian_env_type=linear_gaussian_env_type,
                   continue_training=continue_training, ignore_reward=ignore_reward, use_mlp_policy=use_mlp_policy,
                   rl_type=rl_type)
    elif subroutine == 'evaluate_agent':
        print('executing: {}'.format('evaluate_agent'))
        evaluate_agent(linear_gaussian_env_type, traj_length, dim, model_name, condition_length=condition_length)
    elif subroutine == 'train_and_eval':
        print('executing: {}'.format('train_and_eval'))
        test_train(traj_length=traj_length, dim=dim, condition_length=condition_length,
                   ent_coef=ent_coef, loss_type=loss_type, learning_rate=learning_rate,
                   clip_range=clip_range, linear_gaussian_env_type=linear_gaussian_env_type,
                   continue_training=continue_training, ignore_reward=ignore_reward, use_mlp_policy=use_mlp_policy,
                   rl_type=rl_type)
        evaluate_agent(linear_gaussian_env_type, traj_length, dim, model_name, condition_length=condition_length)
    elif subroutine == 'evaluate_until':
        print('executing: {}'.format('evaluate_until'))
        execute_evaluate_agent_until(linear_gaussian_env_type=linear_gaussian_env_type,
                                     traj_lengths=ess_traj_lengths, dim=dim, loss_type=loss_type,
                                     ent_coef=ent_coef, delta=delta, condition_length=condition_length,
                                     use_mlp_policy=use_mlp_policy)
    elif subroutine == 'ess_traj':
        print('WARNING we probably should NOT be executing: {}\ncheck main.py'.format('ess_traj'))
        traj_lengths = torch.cat([torch.arange(2, 11), torch.arange(12, 17)])
        epsilons = [-5e-3]
        execute_ess_traj(linear_gaussian_env_type, traj_lengths=traj_lengths,
                         dim=dim, epsilons=epsilons, ent_coef=ent_coef, loss_type=loss_type,
                         condition_length=condition_length, use_mlp_policy=use_mlp_policy)
    elif subroutine == 'posterior_filtering_ess_traj':
        print('executing: {}'.format('posterior_filtering_ess_traj'))
        epsilons = [-5e-3]
        table = create_dimension_table(torch.tensor([dim]), random=False)
        posterior_filtering_ess_traj(table=table, traj_lengths=ess_traj_lengths, dim=dim, epsilon=epsilon)
    elif subroutine == 'posterior_filtering_conditional_ess_traj':
        print('executing: {}'.format('posterior_filtering_conditional_ess_traj'))
        table = create_dimension_table(torch.tensor([dim]), random=False)
        table[dim]['R'] = torch.tensor(0.4).reshape(1,1)
        # table[dim]['C'] = torch.tensor(1.0).reshape(1,1)
        # table[dim]['R'] = torch.tensor(1.0).reshape(1,1)
        # table[dim]['A'] = torch.tensor(1.0).reshape(1,1)
        # posterior_filtering_conditional_ess_traj(table=table, traj_lengths=ess_traj_lengths, dim=dim, condition_length=condition_length)
        posterior_filtering_conditional_ess_traj(table=table, traj_lengths=ess_traj_lengths, dim=dim, condition_length=1)
        table[dim]['R'] = torch.tensor(1.1).reshape(1,1)
        posterior_filtering_conditional_ess_traj(table=table, traj_lengths=ess_traj_lengths, dim=dim, condition_length=1)
        table[dim]['R'] = torch.tensor(2.2).reshape(1,1)
        posterior_filtering_conditional_ess_traj(table=table, traj_lengths=ess_traj_lengths, dim=dim, condition_length=1)
        # posterior_filtering_conditional_ess_traj(table=table, traj_lengths=ess_traj_lengths, dim=dim, condition_length=2)
        # posterior_filtering_conditional_ess_traj(table=table, traj_lengths=ess_traj_lengths, dim=dim, condition_length=3)
        # posterior_filtering_conditional_ess_traj(table=table, traj_lengths=ess_traj_lengths, dim=dim, condition_length=4)
    elif subroutine == 'posterior_filtering_conditional_ess_condition_length':
        print('executing: {}'.format('posterior_filtering_conditional_ess_condition_length'))
        table = create_dimension_table(torch.tensor([dim]), random=False)
        posterior_filtering_conditional_ess_condition_length(table=table, traj_length=traj_length, dim=dim, condition_lengths=ess_condition_lengths)
    elif subroutine == 'prior_ess_traj':
        print('executing: {}'.format('prior_ess_traj'))
        table = create_dimension_table(torch.tensor([dim]), random=False)
        prior_ess_traj(table=table, traj_lengths=ess_traj_lengths, dim=dim)
    elif subroutine == 'rl_ess_traj':
        print('executing: {}'.format('rl_ess_traj'))
        table = create_dimension_table(torch.tensor([dim]), random=False)
        rl_ess_traj(linear_gaussian_env_type, table=table, traj_lengths=ess_traj_lengths, dim=dim,
                    ent_coef=ent_coef, loss_type=loss_type, condition_length=condition_length, use_mlp_policy=use_mlp_policy)
    elif subroutine == 'ess_dim':
        print('executing: {}'.format('ess_dim'))
        dims = [x for x in range(1, 50)]
        epsilons = [5e-3]
        table = create_dimension_table(torch.tensor(dims), random=False)
        execute_ess_dim(linear_gaussian_env_type, table, traj_length=traj_length, dims=dims, epsilons=epsilons, ent_coef=ent_coef, loss_type=loss_type)
    elif subroutine == 'posterior_filtering_ess_dim':
        print('executing: {}'.format('posterior_filtering_ess_dim'))
        epsilons = [-5e-3]
        table = create_dimension_table(torch.tensor(ess_dims), random=False)
        posterior_filtering_ess_dim(table=table, traj_length=traj_length, dims=ess_dims, epsilon=epsilon)
    elif subroutine == 'prior_ess_dim':
        print('executing: {}'.format('prior_ess_dim'))
        table = create_dimension_table(torch.tensor(ess_dims), random=False)
        prior_ess_dim(table=table, traj_length=traj_length, dims=ess_dims)
    elif subroutine == 'rl_ess_dim':
        print('executing: {}'.format('rl_ess_dim'))
        table = create_dimension_table(torch.tensor(ess_dims), random=False)
        rl_ess_dim(linear_gaussian_env_type, table=table, traj_length=traj_length, dims=ess_dims, ent_coef=ent_coef, loss_type=loss_type)
    elif subroutine == 'load_ess_partial_data':
        print('executing: {}'.format('load_ess_dim'))
        plot_ess_from_partial_data(filenames, data_type)
    elif subroutine == 'load_ess_data_from_files':
        print('executing: {}'.format('load_ess_data_from_files'))
        plot_ess_from_data_from_files(filenames, data_type)
    elif subroutine == 'load_ess_data':
        print('executing: {}'.format('load_ess_data'))
        plot_ess_from_dir_partial_data(ess_dir, data_type, initial_idx)
    elif subroutine == 'state_occupancy':
        print('executing: {}'.format('state_occupancy'))
        execute_state_occupancy(traj_length=traj_length, ent_coef=ent_coef)
    elif subroutine == '3d_state_occupancy':
        print('executing: {}'.format('3d_state_occupancy'))
        execute_3d_state_occupancy(traj_length=traj_length, ent_coef=ent_coef)
    elif subroutine == 'variance_ratio':
        print('executing: {}'.format('variance_ratio'))
        # execute_variance_ratio_runs(t_len=traj_length, ent_coef=ent_coef, loss_type=loss_type, model_name=model_name)
        execute_variance_ratio_runs(t_len=traj_length, ent_coef=ent_coef, condition_length=condition_length)
    elif subroutine == 'compare_rewards':
        print('executing: {}'.format('compare_rewards'))
        # compare_single_trajectory_rewards(traj_length, dim, loss_type, linear_gaussian_env_type, model_name, condition_length)
        compare_rewards(traj_length, dim, loss_type, linear_gaussian_env_type, model_name, condition_length)
    elif subroutine == 'posterior_filtering_conditional_convergence':
        print('executing: {}'.format('posterior_filtering_conditional_convergence'))
        table = create_dimension_table(torch.tensor([dim]), random=False)
        execute_filtering_posterior_conditional_convergence(table, ess_traj_lengths, dim, condition_length)
    elif subroutine == 'plot_variance':
        print('executing: {}'.format('plot_variance'))
        save_dir = str(time.time())
        table = create_dimension_table(torch.tensor([dim]), random=False)
        table[dim]['Q'] = torch.tensor(1.0).reshape(1,1)
        table[dim]['C'] = torch.tensor(1.0).reshape(1,1)
        table[dim]['R'] = torch.tensor(1.0).reshape(1,1)
        table[dim]['A'] = torch.tensor(0.8).reshape(1,1)
        for R in torch.arange(0.2, 2.4, 0.1):
            table[dim]['R'] = R.reshape(1,1)
            true_variances = plot_posterior_variance(table, traj_length, dim, condition_length=0, save_dir=save_dir)
            plot_posterior_variance(table, traj_length, dim, condition_length=condition_length, true_variances=true_variances, save_dir=save_dir)
        true_variances = plot_posterior_variance(table, traj_length, dim, condition_length=0, save_dir=save_dir)
        plot_posterior_variance(table, traj_length, dim, condition_length=condition_length, true_variances=true_variances, save_dir=save_dir)
    elif subroutine == 'plot_mean':
        print('executing: {}'.format('plot_mean'))
        table = create_dimension_table(torch.tensor([dim]), random=False)
        posterior_evidence = compute_evidence(table=table, traj_length=traj_length, dim=dim)
        ys = posterior_evidence.ys
        means, xs = plot_posterior_mean(ys, traj_length, dim, condition_length=0, xs=None)
        plot_posterior_mean(ys, traj_length, dim, condition_length=condition_length, xs=xs)
        plot_posterior_mean(ys, traj_length, dim, condition_length=2, xs=xs)
    elif subroutine == 'smoothing_reward':
        print('executing: {}'.format('smoothing_reward'))
        smoothing_reward(traj_length, dim, loss_type)
    elif subroutine == 'evaluate_rl_posterior_ensemble':
        print('executing: {}'.format('evaluate_rl_posterior_ensemble'))
        execute_rl_posterior_ensemble(ess_traj_lengths, dim, condition_length, model_name)
    elif subroutine == 'evaluate_pure_rl_ensemble':
        print('executing: {}'.format('evaluate_pure_rl_ensemble'))
        execute_pure_rl_ensemble(ess_traj_lengths, dim, ent_coef, condition_length, use_mlp_policy)
    elif subroutine == 'compute_analytical_kl_at_each_state':
        print('executing: {}'.format('compute_analytical_kl_at_each_state'))
        kls = execute_compute_analytical_kl_at_each_state(traj_length, dim, condition_length)
    elif subroutine == 'compute_policy_kl_at_each_state':
        print('executing: {}'.format('compute_policy_kl_at_each_state'))
        kls = execute_compute_policy_kl_at_each_state(traj_length, dim, condition_length, model_name)
    elif subroutine == 'find_num_samples_for_condition_lengths':
        print('executing: {}'.format('find_num_samples_for_condition_lengths'))
        find_num_samples_for_condition_lengths(traj_length, dim, ess_condition_lengths)
    elif subroutine == 'find_num_samples_for_traj_lengths':
        print('executing: {}'.format('find_num_samples_for_traj_lengths'))
        find_num_samples_for_traj_lengths(ess_traj_lengths, dim, condition_length)
    elif subroutine == 'vi_variance_ratio':
        print('executing: {}'.format('vi_variance_ratio'))
        vlgm = VariationalLGM(args)
        vlgm.load_models()
        policy = vlgm.extract_policy()
        vi_variance_ratio(args, policy)
    elif subroutine == 'vi_ess_traj':
        print('executing: {} condition_length: {}'.format('vi_ess_traj', args.condition_length))
        vi_ess_traj(args)
    else:
        print('executing: {}'.format('custom'))
        table = create_dimension_table(torch.tensor([dim]), random=False)
        # epsilons = [-5e-3, 5e-3]
        epsilons = [0.]
        posterior_evidence = compute_evidence(table=table, traj_length=traj_length, dim=dim)
        posterior_filtering_convergence(table, posterior_evidence, dim, epsilons)

    # epsilons = [-5e-3, 0.0, 5e-3]
    # epsilons = [-5e-2, 0.0]
    # traj_lengths = torch.cat([torch.arange(2, 11), torch.arange(12, 20)])
    # dim = 1
    # ent_coef = 10.0
    # forward_kl = 'forward_kl'
    # forward_model_name = MODEL.format(ent_coef, forward_kl, traj_length, dim)
    # reverse_kl = 'reverse_kl'
    # reverse_model_name = '{}_{}_'.format(ent_coef, reverse_kl) + 'linear_gaussian_model_(traj_{}_dim_{})'
    # dims = np.array([2, 4, 6, 8])

    # table = create_dimension_table(torch.tensor([dim]), random=False)

    # traj plots
    # execute_compare_convergence_traj(table=table, traj_lengths=traj_lengths, epsilons=epsilons, dim=dim)
    # execute_ess_traj(traj_lengths=traj_lengths, dim=dim, epsilons=epsilons)
    # posterior_evidence = compute_evidence(table=table, traj_length=traj_length, dim=dim)
    # rl_convergence(table=table, ys=posterior_evidence.ys,
    #                truth=posterior_evidence.evidence, dim=dim,
    #                model_name=forward_model_name)
    # dim plots
    # dims = np.arange(2, 30, 1)
    # dims = np.arange(10, 22, 1)
    # table = create_dimension_table(dims, random=False)
    # traj_length = torch.tensor(5)
    # execute_compare_convergence_dim(table=table, traj_length=traj_length, epsilons=epsilons, dims=dims)

    # execute_ess_dim(table, traj_lengths[0], dims, epsilons, model_name=forward_kl)

    # prior_ess_dim(traj_lengths=torch.arange(2, 17, 1), dim=1)
    # execute_posterior_ess_dim(table=table, traj_length=traj_length, epsilons=epsilons, dim=dim)
    # rl_ess(traj_lengths=torch.arange(1, 10, 1), dim=1)

    # evidence trials
    # traj_length = 5
    # truth = trial_evidence(table, traj_length, dim)

    # execute_filtering_posterior_convergence(table, traj_lengths, epsilons, dim)
    # test_train(traj_length=traj_length, dim=dim, ent_coef=ent_coef, loss_type=loss_type)

    # t_lens = torch.arange(2, 10)
    # dims = np.arange(1, 10)
    # for dim in dims:
    #    # execute_variance_ratio_runs(t_len=t_len, ent_coef=ent_coef, loss_type=loss_type)
    #    test_train(traj_length=10, dim=dim, ent_coef=ent_coef, loss_type=loss_type)
    #    test_train(traj_length=10, dim=dim+9, ent_coef=ent_coef, loss_type=loss_type)
    # test_train(traj_length=15, dim=dim, ent_coef=ent_coef, loss_type=loss_type)
    # test_train(traj_length=15, dim=dim, ent_coef=ent_coef, loss_type='forward_kl')
    # test_train(traj_length=t_len, dim=dim, ent_coef=ent_coef, loss_type=loss_type)

    # execute_variance_ratio_runs()
    # execute_state_occupancy()

    # t_lens = [10, 15]
    # ent_coef = 0.1
    # loss_type = 'forward_kl'
    # for t_len in t_lens:
    #     execute_variance_ratio_runs(t_len=t_len, ent_coef=ent_coef, loss_type=loss_type)

    # load_rl_model(model_name=model_name, device='cpu')
