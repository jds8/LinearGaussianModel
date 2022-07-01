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
    single_gen_A, single_gen_Q, single_gen_C, single_gen_R, single_gen_mu_0, single_gen_Q_0, \
    test_A, test_Q, test_C, test_R, test_mu_0, test_Q_0, \
    state_transition, score_state_transition, gen_covariance_matrix
    # A, Q, C, R, mu_0, Q_0, \
import wandb
from linear_gaussian_env import LinearGaussianEnv, LinearGaussianSingleYEnv
from math_utils import logvarexp, importance_sampled_confidence_interval, log_effective_sample_size, log_max_weight_proportion, log_mean
from plot_utils import legend_without_duplicate_labels
from linear_gaussian_prob_prog import GaussianRandomVariable, LinearGaussian

# model name
MODEL = 'linear_gaussian_model_(traj_{}_dim_{})'

TODAY = date.today().strftime("%b-%d-%Y")

RL_TIMESTEPS = 1000000
NUM_SAMPLES = 10000
NUM_REPEATS = 100

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, env, verbose=1, eval_freq=10000, log_freq=100,
                    gradient_save_freq = 0, run_id=''):
        super(CustomCallback, self).__init__(verbose)
        self.env = env
        self.eval_freq = eval_freq
        self.log_freq = log_freq
        self.eval_incr = 1
        self.best_mean_reward = -np.Inf
        self.verbose = verbose
        self.gradient_save_freq = gradient_save_freq
        self.current_mod = 1

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
            while not done:
                action = policy(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                horizon += 1
            if isinstance(info, list):
                info = info[0]
            return total_reward, horizon, info

        next_total_reward, next_horizon, wp = eval_policy_helper(policy,env)
        total_reward += next_total_reward
        horizon += next_horizon
        # avg_wp += wp

        return total_reward, horizon#, avg_wp

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        # Evaluate the model
        policy = lambda obs_: self.model.predict(obs_, deterministic=True)[0]
        avg_return, avg_horizon = self.eval_policy(policy, self.training_env)
        self.training_env.reset()
        # log to wandb
        wandb.log({'det_avg_return':avg_return,
                   'det_avg_horizon':avg_horizon,
                   'time_steps': self.num_timesteps,
                   'updates': self.model._n_updates})
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
            avg_return, avg_horizon, avg_wp = self.eval_policy(policy, self.training_env)
            self.training_env.reset()
            #
            wandb.log({'det_avg_return':avg_return,
                       'det_avg_horizon':avg_horizon,
                       'det_avg_wp':avg_wp,
                       'stoch_avg_return': np.mean([val['l'] for val in self.model.ep_info_buffer]),
                       'stoch_avg_horizon': np.mean([val['r'] for val in self.model.ep_info_buffer]),
                       'time_steps': self.num_timesteps,
                       'updates': self.model._n_updates})
            # New best model, you could save the agent here
            if avg_return > self.best_mean_reward:
                self.best_mean_reward = avg_return
                self.save_model()

        # otherwise just log stochastic info
        else:
            wandb.log({'stoch_avg_return': np.mean([val['l'] for val in self.model.ep_info_buffer]),
                       'stoch_avg_horizon': np.mean([val['r'] for val in self.model.ep_info_buffer]),
                       'time_steps': self.num_timesteps,
                       'updates': self.model._n_updates})

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        if self.env.states:
            q_log_prob = self.model.policy.to(self.env.states[-1].device).evaluate_actions(obs=self.env.states[-1].t(), actions=self.env.actions[-1])[1]
            # note that in this case the weights are p(y|x)p(x)/q(x)
            wandb.log({'log weights': self.env.liks[-1] + self.env.p_log_probs[-1] - q_log_prob})

        return True


def train(traj_length, env, dim):
    run = wandb.init(project='linear_gaussian_model training', save_code=True, entity='iai')

    # network archictecture
    arch = [1024 for _ in range(3)]
    # create policy
    model = PPO('MlpPolicy', env, ent_coef=0.01, policy_kwargs=dict(net_arch=[dict(pi=arch, vf=arch)]), device='cpu')

    # train policy
    model.learn(total_timesteps=RL_TIMESTEPS, callback=CustomCallback(env, verbose=1))

    # save model
    model.save(MODEL.format(traj_length, dim))


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


def evaluate(ys, d, N, env=None):
    run = wandb.init(project='linear_gaussian_model evaluation', save_code=True, entity='iai')
    print('\nevaluating...')
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
        if isinstance(xs[0], torch.Tensor):
            xs = torch.cat(xs).reshape(-1, env.traj_length)
        else:
            xs = torch.tensor(np.array(xs)).reshape(-1, env.traj_length)

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

        # p(y)
        # evidence_est += torch.exp(log_num - p_x)/N

    # log_evidence_estimate = torch.logsumexp(log_p_y_over_qs, -1) - torch.log(N)
    # log_evidence_true = analytical_score(true_ys=ys, A=A, Q=Q, C=C, R=R, mu0=mu_0, Q0=Q_0)
    # evidence_estimate = torch.exp(log_evidence_estimate)
    # evidence_true = torch.exp(log_evidence_true)
    # print('log evidence estimate: {}'.format(log_evidence_estimate))
    # print('log evidence true: {}'.format(log_evidence_true))
    # print('evidence estimate: {}'.format(evidence_estimate))
    # print('evidence true: {}'.format(evidence_true))
    # print('abs difference of evidence estimate and evidence: {}'.format(abs(evidence_true-evidence_estimate)))

    # calculate variance estmate as
    # $\hat{\sigma}_{int}^2=(n(n-1))^{-1}\sum_{i=1}^n(f_iW_i-\overline{fW})^2$
    sigma_est = torch.sqrt( (logvarexp(log_p_y_over_qs) - torch.log(torch.tensor(len(log_p_y_over_qs) - 1, dtype=torch.float32))).exp() )
    return EvaluationObject(running_log_estimates=torch.tensor(running_log_evidence_estimates), sigma_est=sigma_est,
                            xts=xts, states=states, actions=actions, priors=priors, liks=liks,
                            log_weights=log_p_y_over_qs)

def load_rl_model(device, traj_length, dim):
    # load model
    model = PPO.load(MODEL.format(traj_length, dim)+'.zip')
    policy = model.policy.to(device)
    return model, policy

def importance_estimate(ys, A, Q, C, R, mu_0, Q_0, N, env=None):
    print('\nimportance estimate\n')
    pd = ProposalDist(A=A, Q=Q)

    # create env
    if env is None:
        env = LinearGaussianEnv(A=A, Q=Q,
                                C=C, R=R,
                                mu_0=mu_0,
                                Q_0=Q_0, ys=ys,
                                traj_length=len(ys),
                                sample=False)

    return evaluate(ys, pd, N, env)

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

def rl_estimate(ys, dim, N, env=None):
    print('\nrl_estimate\n')
    _, policy = load_rl_model(ys.device, traj_length=len(ys), dim=dim)
    return evaluate(ys, d=policy, N=N, env=env)

def full_sweep(ys=None, train_model=False, traj_length=1, env_class=LinearGaussianSingleYEnv):
    if ys is None:
        ys, xs, priors, liks = generate_trajectory(traj_length, gen_A, gen_Q, gen_C, gen_R, gen_mu_0, gen_Q_0)
        states = torch.cat([torch.zeros(1, 1), xs[0:-1]])
        print('\ngenerated trajs\n')
        for s, x, y, lik, prior in zip(states, xs, ys, liks, priors):
            print('s_t = {} a_t = {} y_t = {} where p(a_t|s_t) = N({}, {}) = {} and p(y_t|a_t) = N({}, {}) = {}'.format(s.item(), x.item(), y.item(), (A*s).item(), Q.item(), prior, (C*x).item(), R.item(), lik))
        print('\nys: {}'.format(ys))
        if train_model:
            # create env
            env = env_class(A, Q, C, R, mu_0, Q_0, traj_length=traj_length, sample=True)
            train(traj_length=traj_length, env=env)
    rl_estimate(ys=ys, N=NUM_SAMPLES, dim=gen_A.shape[0])


class Estimator:
    def __init__(self, running_log_estimates, ci,
                 weight_mean, max_weight_prop, ess, ess_ci, label='RL'):
        self.running_log_estimate_repeats = [running_log_estimates]
        self.ci = ci
        self.weight_mean = weight_mean
        self.max_weight_prop = max_weight_prop
        self.ess = ess
        self.ess_ci = ess_ci
        self.label = label

    def add_repeat(self, running_log_estimates):
        self.running_log_estimate_repeats.append(running_log_estimates)

    def plot(self):
        # plot prob estimates
        x_vals = torch.arange(1, len(self.running_log_estimate_repeats[0].squeeze())+1)
        lower_ci, med, upper_ci = torch.quantile(torch.stack(self.running_log_estimate_repeats).exp(), torch.tensor([0.05, 0.5, 0.95]), dim=0)
        plt.plot(x_vals, med.squeeze(), label=self.label)
        plt.fill_between(x_vals, y1=lower_ci, y2=upper_ci, alpha=0.3)


class ISEstimator(Estimator):
    def __init__(self, A, Q, running_log_estimates, ci,
                 weight_mean, max_weight_prop, ess, ess_ci):
        super().__init__(running_log_estimates, ci, weight_mean, max_weight_prop, ess, ess_ci, label="_A {}\n_Q {}".format(A, Q))
        self.A = A
        self.Q = Q
        self.weight_mean = weight_mean
        self.max_weight_prop = max_weight_prop
        self.ess = ess


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
            self.is_estimators[idstr].add_repeat(running_log_estimates)
        else:
            self.is_estimators[idstr] = ISEstimator(A, Q, running_log_estimates,
                                                    ci, weight_mean, max_weight_prop, ess, ess_ci)
            self[idstr] = self.is_estimators[idstr]
        return self.is_estimators[idstr]

    def add_rl_estimator(self, running_log_estimates, ci, weight_mean, max_weight_prop, ess, ess_ci, idstr):
        if idstr in self.rl_estimators:
            self.rl_estimators[idstr].add_repeat(running_log_estimates)
        else:
            self.rl_estimators[idstr] = Estimator(running_log_estimates, ci, weight_mean, max_weight_prop, ess, ess_ci)
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
    plt.figure()
    plt.gca().set_yscale('log')
    traj_length = None
    dims = []
    ess = []
    for output in outputs:
        # Ensure all outputs have same trajectory length
        if traj_length is None:
            traj_length = output.traj_length
        else:
            assert(traj_length == output.traj_length)

        dims.append(output.dimension)

        for key, val in output.is_estimators.items():
            ess.append(val.ess)
        for key, val in output.rl_estimators.items():
            ess.append(val.ess)

    plt.scatter(dims, ess)

    plt.xlabel('Dimension of X')
    plt.ylabel('Effective Sample Size')
    plt.title('Effective Sample Size Versus Hidden Dimension\n(num samples: {} trajectory length: {})'.format(num_samples, traj_length))
    legend_without_duplicate_labels(plt.gca())
    plt.savefig('{}/ess_versus_dimension_(traj_len: {}).png'.format(TODAY, traj_length))

def make_ess_plot(outputs, dimension, num_samples):
    plt.figure()
    plt.gca().set_yscale('log')
    traj_lengths = []
    ess = {}
    for output in outputs:
        traj_lengths.append(output.traj_length)
        for key, val in output.is_estimators.items():
            if key not in ess:
                ess[key] = [val.ess]
            else:
                ess[key].append(val.ess)
        for key, val in output.rl_estimators.items():
            if key not in ess:
                ess[key] = [val.ess]
            else:
                ess[key].append(val.ess)

    plt.scatter(traj_lengths, ess.values())
    # for key, val in ess.items():
    #     plt.scatter(traj_lengths, val, label=key)

    plt.xlabel('Trajectory Length')
    plt.ylabel('Effective Sample Size')
    plt.title('Effective Sample Size Versus Trajectory Length\n (num samples: {})'.format(num_samples))
    legend_without_duplicate_labels(plt.gca())
    plt.savefig('{}/ess_plot_dimension_{}.png'.format(TODAY, dimension))

def make_ci_plot(outputs, dimension):
    plt.figure()
    colors = iter(cm.rainbow(np.linspace(0, 1, len(outputs[0].is_estimators)+1)))
    color_map = {}
    traj_lengths = []
    ci = {}
    true_vals = []
    for output in outputs:
        traj_lengths.append(output.traj_length)
        for key, val in output.is_estimators.items():
            if key not in color_map:
                color_map[key] = next(colors)
            # plt.scatter(x=output.traj_length, y=val.ci[0], color=color_map[key], label=key)
            # plt.scatter(x=output.traj_length, y=val.ci[1], color=color_map[key], label=key)
            plt.axvline(x=output.traj_length, ymin=val.ci[0], ymax=val.ci[1], color=color_map[key], label=key)
            break
        # for key, val in output.rl_estimators.items():
        #     if key not in color_map:
        #         color_map[key] = next(colors)
        #     plt.scatter(x=output.traj_length, y=val.ci[0], color=color_map[key], label=key)
        #     plt.scatter(x=output.traj_length, y=val.ci[1], color=color_map[key], label=key)
        #     break
        true_vals.append(output.truth)

    for traj_len, true_val in zip(traj_lengths, true_vals):
        plt.scatter(traj_len, true_val, label='true prob.', marker='x', color='b')

    plt.xlabel('Trajectory Length')
    plt.ylabel('Confidence Interval')
    plt.title('Confidence Interval (true sample size) Versus Trajectory Length')
    legend_without_duplicate_labels(plt.gca())
    plt.savefig('{}/ci_plot_dimension_{}.png'.format(TODAY, dimension))

def make_ess_ci_plot(outputs, dimension):
    plt.figure()
    colors = iter(cm.rainbow(np.linspace(0, 1, len(outputs[0].is_estimators)+1)))
    color_map = {}
    traj_lengths = []
    ci = {}
    true_vals = []
    for output in outputs:
        traj_lengths.append(output.traj_length)
        for key, val in output.is_estimators.items():
            if key not in color_map:
                color_map[key] = next(colors)
            # plt.scatter(x=output.traj_length, y=val.ess_ci[0], color=color_map[key], label=key)
            # plt.scatter(x=output.traj_length, y=val.ess_ci[1], color=color_map[key], label=key)
            plt.axvline(x=output.traj_length, ymin=val.ess_ci[0], ymax=val.ess_ci[1], color=color_map[key], label=key)
            break
        # for key, val in output.rl_estimators.items():
        #     if key not in color_map:
        #         color_map[key] = next(colors)
        #     plt.scatter(x=output.traj_length, y=val.ess_ci[0], color=color_map[key], label=key)
        #     plt.scatter(x=output.traj_length, y=val.ess_ci[1], color=color_map[key], label=key)
        #     break
        true_vals.append(output.truth)

    for traj_len, true_val in zip(traj_lengths, true_vals):
        plt.scatter(traj_len, true_val, label='true prob.', marker='x', color='b')

    plt.xlabel('Trajectory Length')
    plt.ylabel('Confidence Interval')
    plt.title('Confidence Interval (effective sample size) Versus Trajectory Length')
    legend_without_duplicate_labels(plt.gca())
    plt.savefig('{}/ess_ci_plot_dimension_{}.png'.format(TODAY, dimension))


def display_diagnostics(outputs):
    for output in outputs:
        output.display_self()
    

def make_table_of_confidence_intervals(outputs, name='Event'):
    columns = []
    rows = []
    cell_text = []
    for i, output in enumerate(outputs):
        cell_text.append([])
        rows.append("Traj Length: {}".format(output.traj_length))
        for key, is_est in output.is_estimators.items():
            columns.append('IS (A: {}, Q: {})'.format(is_est.A, is_est.Q))
            cell_text[i].append(is_est.ci)
        for key, rl_est in output.rl_estimators.items():
            columns.append('RL')
            cell_text[i].append(rl_est.ci)
        columns.append('truth')
        cell_text[i].append(output.truth)
    fig, axs = plt.subplots(2,1)
    axs[0].axis('tight')
    axs[0].axis('off')
    axs[0].table(cellText=cell_text,
            rowLabels=tuple(rows),
            colLabels=tuple(columns),
            loc='top')
    plt.savefig('{}/{}_table_confidence_interval.png'.format(TODAY, name))


class Plotter:
    def __init__(self, name, A=None, Q=None, C=None, R=None, mu_0=None, Q_0=None, dim=torch.tensor(1)):
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

    def reset_dimension(self, dim):
        self.dimension = dim
        self.A = torch.rand(dim, dim)
        self.Q = gen_covariance_matrix(dim)
        self.C = torch.rand(1, dim)
        self.R = torch.rand(1, 1)
        self.mu_0 = torch.zeros(dim)
        self.Q_0 = self.Q

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
                num_samples=10000, num_repeats=10):
        if len(As) > 0:
            assert self.dimension == As[0].shape[0]
        outputs = []
        for traj_length in traj_lengths:
            plt.figure(plt.gcf().number+1)

            # generate ys from gen_A, gen_Q params
            ys = self.sample_trajectory(traj_length)

            # collect output
            output = ImportanceOutput(traj_length, ys, dim=self.dimension)

            # get evidence estimate using true params, add it to output, and plot
            log_true = self.get_true_score(ys, traj_length)
            true = round(log_true.exp().item(), 8)
            output.set_truth(true)
            print('true prob.: {}'.format(true))
            plt.axhline(y=true, color='b')

            # generate environment
            env = self.generate_env(ys, traj_length)

            # get evidence estimates using IS with other params
            for i, _A in enumerate(As):
                for j, _Q in enumerate(Qs):
                    for _ in range(num_repeats):
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
                    is_estimator.plot()

            # add RL plot
            try:
                for _ in range(num_repeats):
                    eval_obj = rl_estimate(ys, dim=self.dimension, N=num_samples, env=env)
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
                # plot em
                rl_estimator.plot()

            except:
                print('error could not load: {}'.format(MODEL.format(traj_length, self.dimension)+'.zip'))

            outputs.append(output)

            plt.xlabel('Number of Samples')
            plt.ylabel('Prob. {} Estimate'.format(self.name))
            plt.title('Convergence of Prob. {} Estimate to True Prob. {} \n(trajectory length: {}, dimension: {})'.format(self.name, self.name, traj_length, self.dimension))
            plt.legend()
            plt.savefig('{}/new_traj_length_{}_dimension_{}_{}_convergence.png'.format(TODAY, traj_length, self.dimension, self.name))
        return outputs


class EvidencePlotter(Plotter):
    def __init__(self, num_samples, A=None, Q=None, C=None, R=None, mu_0=None, Q_0=None, dim=torch.tensor(1)):
        self.num_samples = num_samples
        super().__init__('Evidence', A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0, dim=dim)

    def sample_trajectory(self, traj_length):
        print('warning, outputting only zeros as trajectory')
        return torch.zeros(traj_length)
        # return generate_trajectory(traj_length, A=self.A, Q=self.Q, C=self.C, R=self.R, mu_0=self.mu_0, Q_0=self.Q_0)[0]

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
                                 sample=False)


class EventPlotter(Plotter):
    def __init__(self, event_prob, fix_event=False, A=None, Q=None, C=None, R=None, mu_0=None, Q_0=None, dim=torch.tensor(1)):
        super().__init__('Event', A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0, dim=dim)
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
                                        sample=False,
                                        event_prob=self.event_prob)

    def plot_IS(self, traj_lengths=torch.arange(10, 121, 10), As=torch.arange(0.2, 0.6, 0.2), Qs=torch.arange(0.2, 0.6, 0.2),
                num_samples=10000, num_repeats=10):
        for t in traj_lengths:
            d = y_dist(t)
            print('true dist (for traj_length {}) (dim {}): N({}, {})'.format(t, self.dimension, d.mean.item(), d.variance.item()))
        return super().plot_IS(traj_lengths, As, Qs, num_samples=num_samples, num_repeats=num_repeats)

def kl_divergence(traj_length, p, q):
    return 0.

def collect_and_plot_dimension_outputs(ep, As, Qs, traj_length, num_samples, num_repeats):
    dim_outputs = []
    for dimension in torch.arange(1, 25, 5):
        ep.reset_dimension(dimension)
        outputs = ep.plot_IS(traj_lengths=torch.tensor([traj_length]), As=As, Qs=Qs,
                             num_samples=num_samples, num_repeats=num_repeats)
        dim_outputs.append(outputs[0])
    make_ess_versus_dimension_plot(outputs=dim_outputs, num_samples=num_samples)


def make_trajectory_plots(plotter, traj_lengths, As, Qs, dimension, num_samples, num_repeats):
    outputs = plotter.plot_IS(traj_lengths=traj_lengths, As=As, Qs=Qs, num_samples=num_samples, num_repeats=num_repeats)
    # make_table_of_confidence_intervals(outputs, name='EventWithCI')
    display_diagnostics(outputs)
    make_ess_plot(outputs, dimension, num_samples)

    # make_ci_plot(outputs, dimension)
    # make_ess_ci_plot(outputs, dimension)

def plot_event_stuff():
    event_prob = torch.tensor(0.02)
    traj_length = 10
    env = LinearGaussianSingleYEnv(A=single_gen_A, Q=single_gen_Q,
                                   C=single_gen_C, R=single_gen_R,
                                   mu_0=single_gen_mu_0,
                                   Q_0=single_gen_Q_0, ys=None,
                                   traj_length=traj_length,
                                   sample=True,
                                   event_prob=event_prob)
    dimension = 1
    train(traj_length, env, dim=dimension.item())

    os.makedirs(TODAY, exist_ok=True)
    As = [] #[torch.rand(dimension, dimension)]
    Qs = [] #[gen_covariance_matrix(dimension)]
    num_samples = 10000
    num_repeats = 10
    ep = EventPlotter(event_prob=event_prob, fix_event=True, dim=dimension)
    # collect_and_plot_dimension_outputs(ep=ep, As=[torch.rand(dimension, dimension)],
    #                                    Qs=[gen_covariance_matrix(dimension)],
    #                                    traj_length=traj_length, num_samples=num_samples,
    #                                    num_repeats=num_repeats)
    ep.dimension = dimension
    make_trajectory_plots(plotter=ep, event_prob=event_prob, As=As, Qs=Qs, dimension=dimension, num_samples=num_samples, num_repeats=num_repeats)

def train_dimensions(traj_length):
    torch.manual_seed(traj_length)
    os.makedirs(TODAY, exist_ok=True)
    dimensions = torch.arange(1, 11, 1)
    for dimension in dimensions:
        Q = gen_covariance_matrix(dimension)
        env = LinearGaussianEnv(A=torch.rand(dimension, dimension),
                                Q=Q,
                                C=torch.rand(1, dimension), R=torch.rand(1, 1),
                                mu_0=torch.zeros(dimension),
                                Q_0=Q, ys=None,
                                traj_length=traj_length,
                                sample=True)
        train(traj_length, env, dim=dimension.item())


def plot_evidence_vs_trajectory():
    os.makedirs(TODAY, exist_ok=True)
    # traj_lengths = torch.arange(1, 121, 10)
    traj_lengths = torch.arange(1, 11, 1)
    dimension = 1
    # for traj_length in traj_lengths:
    #     env = LinearGaussianEnv(A=single_gen_A, Q=single_gen_Q,
    #                             C=single_gen_C, R=single_gen_R,
    #                             mu_0=single_gen_mu_0,
    #                             Q_0=single_gen_Q_0, ys=None,
    #                             traj_length=traj_length,
    #                             sample=True)
    #     train(traj_length, env, dim=dimension.item())

    num_samples = NUM_SAMPLES
    num_repeats = NUM_REPEATS
    ep = EvidencePlotter(num_samples=num_samples, dim=dimension, A=single_gen_A, Q=single_gen_Q, C=single_gen_C, R=single_gen_R, mu_0=single_gen_mu_0, Q_0=single_gen_Q_0)
    make_trajectory_plots(plotter=ep, traj_lengths=traj_lengths, As=[], Qs=[], dimension=dimension, num_samples=num_samples, num_repeats=num_repeats)

    dim_traj_length = 10
    train_dimensions(traj_length=dim_traj_length)
    collect_and_plot_dimension_outputs(ep=ep, As=[], Qs=[], traj_length=dim_traj_length, num_samples=num_samples, num_repeats=num_repeats)


if __name__ == "__main__":
    plot_evidence_vs_trajectory()
