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
    state_transition, score_state_transition, gen_covariance_matrix, \
    score_initial_state, score_y
    # A, Q, C, R, mu_0, Q_0, \
import wandb
from linear_gaussian_env import LinearGaussianEnv, LinearGaussianSingleYEnv
from math_utils import logvarexp, importance_sampled_confidence_interval, log_effective_sample_size, log_max_weight_proportion, log_mean
from plot_utils import legend_without_duplicate_labels
from linear_gaussian_prob_prog import GaussianRandomVariable, LinearGaussian, VecLinearGaussian

# model name
# MODEL = 'trial_linear_gaussian_model_(traj_{}_dim_{})'
MODEL = 'new_linear_gaussian_model_(traj_{}_dim_{})'
# MODEL = 'from_borg/rl_agents/linear_gaussian_model_(traj_{}_dim_{})'

TODAY = date.today().strftime("%b-%d-%Y")

RL_TIMESTEPS = 100000
NUM_SAMPLES = 100
NUM_REPEATS = 20

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

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # it looks like this is only invoked if there are more than 1000 timesteps in training
        pass

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
    params = {}
    run = wandb.init(project='linear_gaussian_model training', save_code=True, config=params, entity='iai')

    # network archictecture
    arch = [1024 for _ in range(3)]
    # create policy
    model = PPO('MlpPolicy', env, ent_coef=0.01, policy_kwargs=dict(net_arch=[dict(pi=arch, vf=arch)]), device='cpu', verbose=1)

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

def rl_estimate(ys, dim, N, env=None, traj_length=0, device='cpu'):
    if ys is not None:
        traj_length = len(ys)
        device = ys.device
    print('\nrl_estimate\n')
    _, policy = load_rl_model(device, traj_length=traj_length, dim=dim)
    return evaluate(ys, d=policy, N=N, env=env)

class Estimator:
    def __init__(self, running_log_estimates, ci,
                 weight_mean, max_weight_prop, ess, ess_ci, label='RL'):
        self.running_log_estimate_repeats = [running_log_estimates]
        self.ci = ci
        self.weight_mean = weight_mean
        self.max_weight_prop = max_weight_prop
        self.ess = [ess]
        self.ess_ci = ess_ci
        self.label = label

    def add_repeat(self, running_log_estimates, ess):
        self.running_log_estimate_repeats.append(running_log_estimates)
        self.ess.append(ess)

    def plot(self, ax=None):
        ax = ax if ax is not None else plt
        # plot prob estimates
        x_vals = torch.arange(1, self.running_log_estimate_repeats[0].squeeze().nelement()+1)
        lower_ci, med, upper_ci = torch.quantile(torch.stack(self.running_log_estimate_repeats).exp(), torch.tensor([0.05, 0.5, 0.95]), dim=0)
        plt.plot(x_vals, med.squeeze(), label=self.label)
        plt.fill_between(x_vals, y1=lower_ci, y2=upper_ci, alpha=0.3)


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
            try:
                self.is_estimators[idstr] = ISEstimator(A, Q, running_log_estimates,
                                                        ci, weight_mean, max_weight_prop, ess, ess_ci, label=idstr)
            except:
                import pdb; pdb.set_trace()
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
    plt.figure()
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
    plt.savefig('{}/ess_versus_dimension_(traj_len: {}).png'.format(TODAY, traj_length))

def make_ess_plot(outputs, dimension, num_samples, name=''):
    plt.figure()
    plt.gca().set_yscale('log')
    traj_lengths = []
    ess = {}
    for output in outputs:
        traj_lengths.append(output.traj_length)
        if output.is_estimators:
            for key, val in output.is_estimators.items():
                if key not in ess:
                    ess[output.traj_length] = val.ess
                else:
                    ess[output.traj_length] += val.ess
        else:
            for key, val in output.rl_estimators.items():
                if key not in ess:
                    ess[output.traj_length] = val.ess
                else:
                    ess[output.traj_length] += val.ess

    plt.scatter(traj_lengths, [torch.median(torch.tensor(e)) for k, e in ess.items()])
    for traj_length, (k, e) in zip(traj_lengths, ess.items()):
        lower, upper = torch.quantile(torch.tensor(e), torch.tensor([0.05, 0.95]), dim=0)
        plt.vlines(x=traj_length, ymin=lower.item(), ymax=upper.item())
        num_repeats = len(e)

    # plt.scatter(traj_lengths, ess.values())
    # for key, val in ess.items():
    #     plt.scatter(traj_lengths, val, label=key)

    plt.xlabel('Trajectory Length')
    plt.ylabel('Effective Sample Size')
    plt.title('Effective Sample Size Versus Trajectory Length\n (num samples: {}, num repeats: {})'.format(num_samples, num_repeats))
    legend_without_duplicate_labels(plt.gca())
    plt.savefig('{}/{}_ess_plot_dimension_{}.png'.format(TODAY, name, dimension))

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
                    is_estimator.plot()

                # add RL plot
                try:
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

                except:
                    print('error could not load: {}'.format(MODEL.format(traj_length, self.dimension)+'.zip'))

            try:
                # plot em
                rl_estimator.plot()
            except:
                pass

            outputs.append(output)

            plt.xlabel('Number of Samples')
            plt.ylabel('Prob. {} Estimate'.format(self.name))
            plt.title('Convergence of Prob. {} Estimate to True Prob. {} \n(trajectory length: {}, dimension: {})'.format(self.name, self.name, traj_length, self.dimension))
            plt.legend()
            plt.savefig('{}/{}traj_length_{}_dimension_{}_{}_convergence.png'.format(TODAY, name, traj_length, self.dimension, self.name))
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

def kl_divergence(traj_length, p, q):
    return 0.

def collect_and_plot_dimension_outputs(ep, dimensions, table, As, Qs, traj_length, num_samples, num_repeats):
    dim_outputs = []
    for dimension in dimensions:
        ep.reset(table, dimension)
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
    train(traj_length, env, dim=dimension)

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

def train_dimensions(traj_length, dimensions, table):
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
        train(traj_length, env, dim=dimension)

def create_dimension_table(dimensions, random=False):
    table = {}
    if random:
        for dim in dimensions:
            dimension = dim.item()
            table[dimension] = {}
            table[dimension]['A'] = torch.rand(dimension, dimension)
            table[dimension]['Q'] = gen_covariance_matrix(dimension)
            table[dimension]['C'] = torch.rand(1, dimension)
            table[dimension]['R'] = torch.rand(1, 1)
            table[dimension]['mu_0'] = torch.zeros(dimension)
            table[dimension]['Q_0'] = table[dimension]['Q']
    else:
        for dim in dimensions:
            dimension = dim.item()
            table[dimension] = {}
            table[dimension]['A'] = torch.eye(dimension, dimension)
            table[dimension]['Q'] = torch.eye(dimension)
            table[dimension]['C'] = torch.eye(1, dimension)
            table[dimension]['R'] = torch.eye(1, 1)
            table[dimension]['mu_0'] = torch.zeros(dimension)
            table[dimension]['Q_0'] = table[dimension]['Q']
    return table

def plot_evidence_vs_trajectory():
    os.makedirs(TODAY, exist_ok=True)
    dimension = 1
    traj_lengths = torch.arange(3, 7, 1)
    # traj_lengths = torch.arange(10, 31, 10)
    # for traj_length in traj_lengths:
    #     env = LinearGaussianEnv(A=single_gen_A, Q=single_gen_Q,
    #                             C=single_gen_C, R=single_gen_R,
    #                             mu_0=single_gen_mu_0,
    #                             Q_0=single_gen_Q_0, ys=None,
    #                             traj_length=traj_length,
    #                             sample=True)
    #     train(traj_length, env, dim=dimension)

    num_samples = NUM_SAMPLES
    num_repeats = NUM_REPEATS
    ep = EvidencePlotter(num_samples=num_samples, dim=dimension, A=single_gen_A, Q=single_gen_Q, C=single_gen_C, R=single_gen_R, mu_0=single_gen_mu_0, Q_0=single_gen_Q_0)
    make_trajectory_plots(plotter=ep, traj_lengths=traj_lengths, As=[], Qs=[], dimension=dimension, num_samples=num_samples, num_repeats=num_repeats)

def plot_evidence_vs_dim():
    os.makedirs(TODAY, exist_ok=True)
    dim_traj_length = 5
    ep = EvidencePlotter(num_samples=NUM_SAMPLES, dim=1, A=single_gen_A, Q=single_gen_Q, C=single_gen_C, R=single_gen_R, mu_0=single_gen_mu_0, Q_0=single_gen_Q_0)
    dimensions = torch.arange(1, 10, 1)
    # dimensions = [torch.tensor(3)]

    torch.manual_seed(dim_traj_length)
    table = create_dimension_table(dimensions, random=False)
    # train_dimensions(traj_length=dim_traj_length, dimensions=dimensions, table=table)
    collect_and_plot_dimension_outputs(ep=ep, dimensions=dimensions, table=table, As=[], Qs=[], traj_length=dim_traj_length, num_samples=NUM_SAMPLES, num_repeats=NUM_REPEATS)

def compute_posterior(num_observations):
    w = GaussianRandomVariable(mu=0., sigma=torch.sqrt(single_gen_Q), name="w")
    v = GaussianRandomVariable(mu=0., sigma=torch.sqrt(single_gen_R), name="v")
    xt = GaussianRandomVariable(mu=single_gen_mu_0, sigma=torch.sqrt(single_gen_Q_0), name="x")
    xs = [xt]
    ys = []
    posterior_xt_prev_given_yt_prev = None
    for i in range(num_observations):
        yt = LinearGaussian(a=single_gen_C, x=xt, b=v, name="y")
        ys.append(yt)
        xt = LinearGaussian(a=single_gen_A, x=xt, b=w, name="x")
        xs.append(xt)

    prior = xs[0].prior()
    for i in range(1, num_observations):
        prior *= xs[i].likelihood()

    C = single_gen_C * torch.eye(num_observations)

    noise = v.prior()**num_observations

    # find full likelihood
    ys = VecLinearGaussian(a=C, x=prior, b=noise)

    # compute posterior
    return ys.posterior_vec()

def condition_posterior(td, obs):
    return td.dist(arg={'value': obs})

def evaluate_posterior(ys, N, td, env=None):
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
    for i in range(N):
        done = False
        # get first obs
        obs = env.reset()
        # keep track of xs
        xs = td.sample()
        # keep track of prior over actions p(x)
        log_p_x = torch.tensor(0.).reshape(1, -1)
        log_p_y_given_x = torch.tensor(0.).reshape(1, -1)
        # collect actions, likelihoods
        states = [obs]
        actions = xs
        if len(xs) > 1:
            xts = torch.cat([env.prev_xt.reshape(-1), xs[0:-1]])
        else:
            xts = env.prev_xt.reshape(-1)
        prior_reward = score_initial_state(x0=xs[0].reshape(1), mu_0=env.mu_0, Q_0=env.Q_0)  # the reward for the initial state
        lik_reward = score_y(y_test=ys[0], x_t=xs[0], C=env.C, R=env.R)  # the first likelihood reward
        priors = [prior_reward]
        log_p_x += prior_reward
        liks = [lik_reward]
        log_p_y_given_x += lik_reward
        total_reward = prior_reward + lik_reward
        if len(xts) > 1:
            for xt, prev_xt, y in zip(xs[1:], xts[1:], ys[1:]):
                prior_reward = score_state_transition(xt=xt.reshape(1), prev_xt=prev_xt, A=env.A, Q=env.Q)
                lik_reward = score_y(y_test=y, x_t=xt, C=env.C, R=env.R)  # the first likelihood reward
                total_reward += prior_reward + lik_reward
                states.append(obs)
                priors.append(prior_reward)
                liks.append(lik_reward)
                log_p_x += prior_reward
                log_p_y_given_x += lik_reward
        # log p(x,y)
        log_p_y_x = log_p_y_given_x + log_p_x
        log_q = td.log_prob(xs)
        if N == 1:
            log_p_y_over_qs[i] = (log_p_y_x - log_q).item()
            running_log_evidence_estimates.append(log_p_y_over_qs[0])
        else:
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


class PosteriorEvidence:
    def __init__(self, td, ys, evidence, env):
        self.td = td
        self.ys = ys
        self.evidence = evidence
        self.env =env


def compute_evidence(traj_length, dim):
    os.makedirs(TODAY, exist_ok=True)
    table = create_dimension_table(torch.tensor([dim]), random=False)

    ys = generate_trajectory(traj_length, A=single_gen_A, Q=single_gen_Q, C=single_gen_C, R=single_gen_R, mu_0=single_gen_mu_0, Q_0=single_gen_Q_0)[0]
    env = LinearGaussianEnv(A=single_gen_A, Q=single_gen_Q,
                            C=single_gen_C, R=single_gen_R,
                            mu_0=single_gen_mu_0,
                            Q_0=single_gen_Q_0, ys=ys,
                            sample=True)

    # A = table[dim]['A']
    # Q = table[dim]['Q']
    # C = table[dim]['C']
    # R = table[dim]['R']
    # mu_0 = table[dim]['mu_0']
    # Q_0 = table[dim]['Q_0']

    # ys = generate_trajectory(traj_length, A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0)[0]
    # env = LinearGaussianEnv(A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0, ys=ys, sample=False)

    posterior = compute_posterior(len(ys))
    td = condition_posterior(posterior, ys)

    eval_obj = evaluate_posterior(ys=ys, N=1, td=td, env=env)
    true = eval_obj.running_log_estimates[0].exp()
    print('True evidence: {}'.format(true))

    return PosteriorEvidence(td, ys, true, env)

def get_prior_output(ys, dim, sample, traj_length=0):
    if ys is not None:
        traj_length = len(ys)
    prior_output = ImportanceOutput(traj_length=traj_length, ys=ys, dim=dim)
    name = 'prior (A: {}, Q: {})'.format(single_gen_A, single_gen_Q)
    for _ in range(NUM_REPEATS):
        eval_obj = importance_estimate(ys, A=single_gen_A, Q=single_gen_Q, C=single_gen_C, R=single_gen_R, mu_0=single_gen_mu_0, Q_0=single_gen_Q_0, N=NUM_SAMPLES, sample=sample, traj_length=traj_length)
        is_estimator = prior_output.add_is_estimator(A=single_gen_A, Q=single_gen_Q,
                                                     running_log_estimates=eval_obj.running_log_estimates,
                                                     ci=eval_obj.ci,
                                                     weight_mean=eval_obj.log_weight_mean.exp(),
                                                     max_weight_prop=eval_obj.log_max_weight_prop.exp(),
                                                     ess=eval_obj.log_effective_sample_size.exp(), ess_ci=eval_obj.ess_ci,
                                                     idstr=name)
    return OutputWithName(prior_output, name)

def get_rl_output(ys, dim, sample, traj_length=0):
    if ys is not None:
        traj_length = len(ys)
    rl_output = ImportanceOutput(traj_length=traj_length, ys=ys, dim=dim)
    name = 'rl (traj_len {} dim {})'.format(traj_length, dim)
    for _ in range(NUM_REPEATS):
        env = LinearGaussianEnv(A=single_gen_A, Q=single_gen_Q,
                                C=single_gen_C, R=single_gen_R,
                                mu_0=single_gen_mu_0,
                                Q_0=single_gen_Q_0, ys=ys,
                                traj_length=traj_length,
                                sample=sample)

        eval_obj = rl_estimate(ys, dim=dim, N=NUM_SAMPLES, env=env, traj_length=traj_length)
        # add rl confidence interval
        rl_estimator = rl_output.add_rl_estimator(running_log_estimates=eval_obj.running_log_estimates,
                                                  ci=eval_obj.ci, weight_mean=eval_obj.log_weight_mean.exp(),
                                                  max_weight_prop=eval_obj.log_max_weight_prop.exp(),
                                                  ess=eval_obj.log_effective_sample_size.exp(),
                                                  ess_ci=eval_obj.ess_ci, idstr=name)

    return OutputWithName(rl_output, name)

def get_perturbed_posterior_output(posterior_evidence, dim, epsilon, name):
    ys = posterior_evidence.ys
    true_posterior = posterior_evidence.td
    env = posterior_evidence.env

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


class OutputWithName:
    def __init__(self, output, name):
        self.output = output
        self.name = name


def get_perturbed_posterior_outputs(posterior_evidence, dim, epsilons):
    outputs = []
    for epsilon in epsilons:
        name = 'posterior {}'.format(epsilon)
        output = get_perturbed_posterior_output(posterior_evidence, dim, epsilon, name)
        outputs.append(output)
    return outputs

def plot_estimators(outputs_with_names, ax=None):
    plt.figure()
    for output in outputs_with_names:
        estimator = output.output[output.name]
        estimator.plot(ax)

def plot_convergence(outputs_with_names, traj_length, dim, true, name):
    plot_estimators(outputs_with_names)

    # plot em
    plt.scatter(x=NUM_SAMPLES, y=true, label='True: {}'.format(true.item()), color='r')
    plt.xlabel('Number of Samples')
    plt.ylabel('Prob. {} Estimate'.format('Evidence'))
    plt.title('Convergence of Prob. {} Estimate to True Prob. {} \n(trajectory length: {}, dimension: {})'.format('Evidence', 'Evidence', traj_length, dim))
    legend_without_duplicate_labels(plt.gca())
    plt.savefig('{}/{}_traj_length_{}_dimension_{}_{}_convergence.png'.format(TODAY, name, traj_length, dim, 'Evidence'))

def posterior_convergence(posterior_evidence, dim, epsilons):
    posterior_outputs_with_names = get_perturbed_posterior_outputs(posterior_evidence, dim, epsilons)
    traj_length = len(posterior_evidence.ys)
    plot_convergence(posterior_outputs_with_names, traj_length, dim, posterior_evidence.evidence, 'posterior')

def prior_convergence(ys, truth, dim):
    prior_outputs_with_name = get_prior_output(ys, dim, sample=False)
    traj_length = len(ys)
    plot_convergence([prior_outputs_with_name], traj_length, dim, truth, 'prior')

def rl_convergence(ys, truth, dim):
    rl_outputs_with_name = get_rl_output(ys, dim, sample=False)
    traj_length = len(ys)
    plot_convergence([rl_outputs_with_name], traj_length, dim, truth, 'rl (traj_length {} dim {})'.format(traj_length, dim))

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

def plot_traj():
    os.makedirs(TODAY, exist_ok=True)
    traj_lengths = [7, 8, 9, 10]
    for traj_length in traj_lengths:
        env = LinearGaussianEnv(A=single_gen_A, Q=single_gen_Q,
                                C=single_gen_C, R=single_gen_R,
                                mu_0=single_gen_mu_0,
                                Q_0=single_gen_Q_0, ys=None,
                                traj_length=traj_length,
                                sample=True)
        train(traj_length, env=env, dim=1)
        ys = generate_trajectory(traj_length, A=single_gen_A, Q=single_gen_Q, C=single_gen_C, R=single_gen_R, mu_0=single_gen_mu_0, Q_0=single_gen_Q_0)[0]
        ep = EvidencePlotter(num_samples=NUM_SAMPLES, dim=1, A=single_gen_A, Q=single_gen_Q, C=single_gen_C, R=single_gen_R, mu_0=single_gen_mu_0, Q_0=single_gen_Q_0)

    outputs = ep.plot_IS(traj_lengths=traj_lengths, As=[], Qs=[], num_samples=NUM_SAMPLES, num_repeats=NUM_REPEATS, name='extra')
    make_ess_plot(outputs, dimension=1, num_samples=NUM_SAMPLES)

def compare_convergence(traj_length, dim, epsilons):
    posterior_evidence = compute_evidence(traj_length, dim)
    posterior_convergence(posterior_evidence, dim, epsilons)
    prior_convergence(posterior_evidence.ys, posterior_evidence.evidence, dim)
    rl_convergence(posterior_evidence.ys, posterior_evidence.evidence, dim)

def posterior_ess(traj_lengths, dim, epsilons):
    os.makedirs(TODAY, exist_ok=True)
    for epsilon in epsilons:
        outputs = []
        for traj_length in traj_lengths:
            posterior_evidence = compute_evidence(traj_length, dim)
            posterior_outputs_with_names = get_perturbed_posterior_outputs(posterior_evidence, dim, [epsilon])
            outputs += [o.output for o in posterior_outputs_with_names]
        make_ess_plot(outputs, dim, NUM_SAMPLES, name='posterior_{}'.format(epsilon))
        plt.figure()

def prior_ess(traj_lengths, dim):
    os.makedirs(TODAY, exist_ok=True)
    outputs = []
    for traj_length in traj_lengths:
        # ys = generate_trajectory(traj_length, A=single_gen_A, Q=single_gen_Q, C=single_gen_C, R=single_gen_R, mu_0=single_gen_mu_0, Q_0=single_gen_Q_0)[0]
        prior_output_with_name = get_prior_output(ys=None, dim=dim, sample=True, traj_length=traj_length)
        outputs += [prior_output_with_name.output]
    make_ess_plot(outputs, dim, NUM_SAMPLES, name='prior')
    plt.figure()

def rl_ess(traj_lengths, dim):
    os.makedirs(TODAY, exist_ok=True)
    outputs = []
    for traj_length in traj_lengths:
        # ys = generate_trajectory(traj_length, A=single_gen_A, Q=single_gen_Q, C=single_gen_C, R=single_gen_R, mu_0=single_gen_mu_0, Q_0=single_gen_Q_0)[0]
        rl_output_with_name = get_rl_output(ys=None, dim=dim, sample=True, traj_length=traj_length)
        outputs += [rl_output_with_name.output]
    make_ess_plot(outputs, dim, NUM_SAMPLES, name='rl (traj_length {} dim {})'.format(traj_length, dim))
    plt.figure()

def execute_posterior_ess():
    os.makedirs(TODAY, exist_ok=True)
    epsilons = [-1e-3, 0., 1e-3]
    posterior_ess(traj_lengths=torch.arange(16, 30, 1), dim=1, epsilons=epsilons)

def execute_compare_convergence(traj_lengths):
    os.makedirs(TODAY, exist_ok=True)
    epsilons = [-1e-3, 1e-3]
    for traj_length in traj_lengths:
        compare_convergence(traj_length=traj_length, dim=1, epsilons=epsilons)

def execute_ess(traj_lengths, dim):
    os.makedirs(TODAY, exist_ok=True)
    posterior_ess(traj_lengths, dim, epsilons)
    prior_ess(traj_lengths, dim)
    rl_ess(traj_lengths, dim)


if __name__ == "__main__":
    os.makedirs(TODAY, exist_ok=True)
    # execute_compare_convergence(torch.arange(2, 10, 1))
    # prior_ess(traj_lengths=torch.arange(2, 17, 1), dim=1)
    execute_posterior_ess()
    # rl_ess(traj_lengths=torch.arange(1, 10, 1), dim=1)

    # plot_traj()
    # plot_evidence_vs_trajectory()
    # plot_evidence_vs_dim()
