#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
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
    A, Q, C, R, mu_0, Q_0, \
    state_transition, score_state_transition
import wandb
from linear_gaussian_env import LinearGaussianEnv, LinearGaussianSingleYEnv
from math_utils import logvarexp, importance_sampled_confidence_interval, log_effective_sample_size, log_max_weight_proportion, log_mean

# model name
MODEL = 'linear_gaussian_model'


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
            wandb.log({'log weights': self.env.p_log_probs[-1] - q_log_prob})

        return True


def train(traj_length, env):
    run = wandb.init(project='linear_gaussian_model', save_code=True, entity='iai')

    # network archictecture
    arch = [1024 for _ in range(3)]
    # create policy
    model = PPO('MlpPolicy', env, ent_coef=0.01, policy_kwargs=dict(net_arch=[dict(pi=arch, vf=arch)]), device='cpu')

    # train policy
    model.learn(total_timesteps=10000, callback=CustomCallback(env, verbose=1))

    # save model
    model.save(MODEL)


class ProposalDist:
    def __init__(self, A, Q):
        if isinstance(A, torch.Tensor):
            self.A = A
        else:
            self.A = torch.tensor(A).reshape(1, -1)
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

        # We can ignore the "y" part of obs
        prev_xt = obs.reshape(-1, 1)[0, :]

        return (None, score_state_transition(actions, prev_xt, self.A, self.Q))

    def predict(self, obs, deterministic=False):
        # We can ignore the "y" part of obs
        prev_xt = obs.reshape(-1, 1)[0, :]
        return state_transition(prev_xt, self.A, self.Q)


class EvaluationObject:
    def __init__(self, running_log_estimates, var_est, xts, states, actions, priors, liks, log_weights):
        self.running_log_estimates = running_log_estimates
        self.var_est = var_est
        self.xts = xts
        self.states = states
        self.actions = actions
        self.priors = priors
        self.liks = liks
        self.log_weights = log_weights

        self._compute_statistics()

    def _compute_statistics(self):
        self._compute_confidence_interval()
        self._compute_importance_weight_diagnostics()

    def _compute_confidence_interval(self):
        self.ci = importance_sampled_confidence_interval(torch.exp(self.running_log_estimates[-1]), self.var_est,
                                                         len(self.running_log_estimates), epsilon=torch.tensor(0.05))

    def _compute_importance_weight_diagnostics(self):
        self.log_weight_mean = log_mean(self.log_weights)
        self.log_max_weight_prop = log_max_weight_proportion(self.log_weights)
        self.log_effective_sample_size = log_effective_sample_size(self.log_weights)


def evaluate(ys, d, env=None):
    print('\nevaluating...')
    if env is None:
        # create env
        if env is None:
            env = LinearGaussianEnv(A=gen_A, Q=gen_Q,
                                    C=gen_C, R=gen_R,
                                    mu_0=gen_mu_0,
                                    Q_0=gen_Q_0, ys=ys,
                                    sample=False)

    # collect joint p(x,y)
    joints = []
    # evidence estimate
    evidence_est = torch.tensor(0.).reshape(1, -1)
    log_evidence_est = torch.tensor(0.).reshape(1, -1)
    # evaluate N times
    N = torch.tensor(1000)
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
        while not done:
            xt = d.predict(obs, deterministic=False)[0]
            xts.append(env.prev_xt)
            obs, reward, done, info = env.step(xt)
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
        # log_num = log_joint(xs=xs, ys=ys, A=A, Q=Q, C=C, R=R, mu0=mu_0, Q0=Q_0)
        # log p(y)
        # log_evidence_est += log_num - p_x - torch.log(N)
        # q(x_1,x_2,...,x_t)

        # log p(x,y)
        log_p_y_x = log_p_y_given_x + log_p_x
        log_qrobs = torch.zeros(len(env.states))
        for j in range(len(env.states)):
            state = env.states[j]
            action = actions[j]
            log_qrobs[j] = d.evaluate_actions(obs=state.t(), actions=action)[1].item()

        log_q = torch.sum(log_qrobs)
        log_p_y_over_qs[i] = (log_p_y_x - log_q).item()
        running_log_evidence_estimates.append(torch.logsumexp(log_p_y_over_qs[0:i+1], -1) - torch.log(torch.tensor(i+1.)))
        log_weights.append(log_p_x - log_q)

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

    sigma_sq = logvarexp(log_p_y_over_qs).exp()
    return EvaluationObject(running_log_estimates=running_log_evidence_estimates, var_est=sigma_sq,
                            xts=xts, states=states, actions=actions, priors=priors, liks=liks, log_weights=torch.tensor(log_weights))

def load_rl_model(device):
    # load model
    model = PPO.load(MODEL+'.zip')
    policy = model.policy.to(device)
    return model, policy

def importance_estimate(ys, A=gen_A, Q=gen_Q, env=None):
    print('\nimportance estimate\n')
    pd = ProposalDist(A=A, Q=Q)

    # create env
    if env is None:
        env = LinearGaussianEnv(A=gen_A, Q=gen_Q,
                                C=gen_C, R=gen_R,
                                mu_0=gen_mu_0,
                                Q_0=gen_Q_0, ys=ys,
                                traj_length=len(ys),
                                sample=False)

    return evaluate(ys, pd, env)

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

    return importance_estimate(ys, A=A, Q=Q)

def rl_estimate(ys, env=None):
    print('\nrl_estimate\n')
    _, policy = load_rl_model(ys.device)
    return evaluate(ys, policy, env)

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
    rl_estimate(ys=ys)

def plot_log_diffs(log_values, log_true, label):
    # compute log ratio of estimate to true value
    diffs = torch.tensor(log_values) - log_true
    plt.plot(torch.arange(1, len(diffs.squeeze())+1), diffs.squeeze(), label=label)


class Estimator:
    def __init__(self, ci, weight_mean, max_weight_prop, ess):
        self.ci = ci
        self.weight_mean = weight_mean
        self.max_weight_prop = max_weight_prop
        self.ess = ess


class ISEstimator(Estimator):
    def __init__(self, A, Q, ci, weight_mean, max_weight_prop, ess):
        super().__init__(ci, weight_mean, max_weight_prop, ess)
        self.A = A
        self.Q = Q
        self.weight_mean = weight_mean
        self.max_weight_prop = max_weight_prop
        self.ess = ess


class ImportanceOutput(dict):
    def __init__(self, traj_length, ys):
        self.traj_length = traj_length
        self.ys = ys
        self.is_estimators = {}
        self.rl_estimators = {}

    def add_is_estimator(self, A, Q, ci, weight_mean, max_weight_prop, ess, idstr):
        if idstr in self.is_estimators:
            raise NotImplementedError
        self.is_estimators[idstr] = ISEstimator(A, Q, ci, weight_mean, max_weight_prop, ess)

    def add_rl_estimator(self, ci, weight_mean, max_weight_prop, ess, idstr):
        if idstr in self.rl_estimators:
            raise NotImplementedError
        self.rl_estimators[idstr] = RLEstimator(ci, weight_mean, max_weight_prop, ess)

    def display_self(self):
        for key, val in self.items():
            print('{}: {}'.format(key, val))


def make_ess_plot(outputs):
    plt.figure()
    traj_lengths = []
    ess = {}
    true_vals = []
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
        true_vals.append(output.traj_length)
    for key, val in ess.items():
        plt.scatter(traj_lengths, val, label=key)
        plt.scatter(traj_lengths, true_vals, label=key, marker='x', color='r')

    plt.xlabel('Trajectory Length')
    plt.ylabel('Effective Sample Size')
    plt.title('Effective Sample Size Versus Trajectory Length')
    plt.legend()
    plt.savefig('./ess_plot.png')


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
        cell_text[i].append(output['truth'])
    fig, axs = plt.subplots(2,1)
    axs[0].axis('tight')
    axs[0].axis('off')
    axs[0].table(cellText=cell_text,
            rowLabels=tuple(rows),
            colLabels=tuple(columns),
            loc='top')
    plt.savefig('./{}_table_confidence_interval.png'.format(name))


class Plotter:
    def __init__(self, name):
        self.name = name

    def generate_env(self, ys, traj_length):
        raise NotImplementedError

    def sample_trajectory(self, traj_length):
        raise NotImplementedError

    def get_true_score(self, ys):
        raise NotImplementedError

    def plot_IS(self, traj_lengths=[1, 10], As=torch.arange(0.2, 0.6, 0.2), Qs=torch.arange(0.2, 0.6, 0.2)):
        outputs = []
        for traj_length in traj_lengths:
            plt.figure(plt.gcf().number+1)

            # generate ys from gen_A, gen_Q params
            ys = self.sample_trajectory(traj_length)

            # collect output
            output = ImportanceOutput(traj_length, ys)

            # get evidence estimate using true params and add it to output
            log_true = self.get_true_score(ys, traj_length)
            output['truth'] = round(log_true.exp().item(), 4)
            print('true prob.: {}'.format(round(log_true.exp().item(), 4)))

            # generate environment
            env = self.generate_env(ys, traj_length)

            # get evidence estimates using IS with other params
            for i, _A in enumerate(As):
                rounded_A = round(_A.item(), 1)
                for j, _Q in enumerate(Qs):
                    rounded_Q = round(_Q.item(), 1)

                    # get is estimate of evidence using _A and _Q params
                    eval_obj = importance_estimate(ys, A=_A.reshape(1, 1), Q=_Q.reshape(1, 1), env=env)
                    print('IS A: {}, Q: {} estimate: {}'.format(rounded_A, rounded_Q, eval_obj.running_log_estimates[-1].exp()))

                    # plot differences
                    plot_log_diffs(eval_obj.running_log_estimates, log_true, label='A: {}, Q: {}'.format(rounded_A, rounded_Q))

                    # add importance confidence interval
                    output.add_is_estimator(A=rounded_A, Q=rounded_Q, ci=eval_obj.ci,
                                            weight_mean=eval_obj.log_weight_mean.exp(),
                                            max_weight_prop=eval_obj.log_max_weight_prop.exp(),
                                            ess=eval_obj.log_effective_sample_size.exp(),
                                            idstr='A: {}, Q: {}'.format(rounded_A, rounded_Q))

            # add RL plot
            try:
                eval_obj = rl_estimate(ys, env)
                print('rl estimate: {}'.format(eval_obj.running_log_estimates[-1].exp()))
                plot_log_diffs(eval_obj.running_log_estimates, log_true, label='RL')
                output.add_rl_estimator(eval_obj.ci, weight_mean=eval_obj.log_weight_mean.exp(),
                                        max_weight_prop=eval_obj.log_max-weight_prop.exp(),
                                        ess=eval_obj.log_effective_sample_size.exp(), idstr='rl1')

            except:
                pass

            outputs.append(output)

            plt.xlabel('Number of Samples')
            plt.ylabel('log Ratio of Evidence Estimate with True Evidence')
            plt.title('Convergence of Evidence Estimate to True Evidence (trajectory length: {})'.format(traj_length))
            plt.legend()
            plt.savefig('./traj_length_{}_{}_convergence.png'.format(traj_length, self.name))
        return outputs


class EvidencePlotter(Plotter):
    def __init__(self):
        super().__init__('Evidence')

    def sample_trajectory(self, traj_length):
        return generate_trajectory(traj_length)[0]

    def get_true_score(self, ys, traj_length):
        # ignore traj_length
        return importance_estimate(ys, A=gen_A, Q=gen_Q).running_log_estimates[-1]

    def generate_env(self, ys, traj_length):
        # ignore traj_length
        return LinearGaussianEnv(A=gen_A, Q=gen_Q,
                                 C=gen_C, R=gen_R,
                                 mu_0=gen_mu_0,
                                 Q_0=gen_Q_0, ys=ys,
                                 traj_length=len(ys),
                                 sample=False)


class EventPlotter(Plotter):
    def __init__(self, threshold):
        super().__init__('Event')
        self.threshold = threshold

    def sample_trajectory(self, num_steps):
        ys, score, d = sample_y(num_steps=num_steps)
        return (ys > self.threshold).type(ys.dtype)

    def get_true_score(self, ys, traj_length):
        d = y_dist(traj_length)
        return dist.Bernoulli(1-d.cdf(self.threshold)).log_prob(ys)

    def generate_env(self, ys, traj_length):
        return LinearGaussianSingleYEnv(A=gen_A, Q=gen_Q,
                                        C=gen_C, R=gen_R,
                                        mu_0=gen_mu_0,
                                        Q_0=gen_Q_0, ys=ys,
                                        traj_length=traj_length,
                                        sample=False,
                                        threshold=self.threshold)

    def plot_IS(self, traj_lengths=[1, 10]):
        for t in traj_lengths:
            d = y_dist(t)
            print('true dist (for traj_length {}): N({}, {})'.format(t, d.mean.item(), d.variance.item()))
        return super().plot_IS(traj_lengths)

if __name__ == "__main__":
    # ep = EvidencePlotter()
    # outputs = ep.plot_IS()

    # traj_length = 10
    # env = LinearGaussianSingleYEnv(A=gen_A, Q=gen_Q,
    #                                C=gen_C, R=gen_R,
    #                                mu_0=gen_mu_0,
    #                                Q_0=gen_Q_0, ys=None,
    #                                traj_length=traj_length,
    #                                sample=True,
    #                                threshold=0.5)
    # train(traj_length, env)

    ep = EventPlotter(threshold=torch.tensor(0.5))
    outputs = ep.plot_IS()
    make_table_of_confidence_intervals(outputs, name='Event')
    display_diagnostics(outputs)
    import pdb; pdb.set_trace()
    make_ess_plot(outputs)
