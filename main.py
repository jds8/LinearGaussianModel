#!/usr/bin/env python3
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
    A, Q, C, R, mu_0, Q_0, \
    state_transition, score_state_transition
from linear_gaussian_env import LinearGaussianEnv, LinearGaussianSingleYEnv

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
    model.learn(total_timesteps=100000, callback=CustomCallback(env, verbose=1))

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
    N = torch.tensor(10000)
    # collect log( p(x,y)/q(x) )
    log_p_y_over_qs = torch.zeros(N)
    # keep track of log evidence estimates up to N sample trajectories
    running_log_evidence_estimates = []
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
        actions = []
        priors = []
        liks = []
        xts = []
        while not done:
            xt = d.predict(obs, deterministic=False)[0]
            xts.append(env.prev_xt)
            obs, reward, done, info = env.step(xt)
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

    return running_log_evidence_estimates, xts, env.states, actions, priors, liks

def logp(ys, t, C=1, A=1, Q=1, R=1, mu=0):
    return dist.MultivariateNormal(C*A**t*mu, C**2*Q**2*t + R**2).log_prob(ys)

def create_iso_plot():
    ys = torch.arange(-100, 100, 0.1)
    A = torch.tensor(0.8).reshape(1, 1)
    Q = torch.tensor(0.2).reshape(1, 1)
    C = torch.tensor(0.8).reshape(1, 1)
    R = torch.tensor(0.2).reshape(1, 1)
    mu_0 = torch.ones_like(ys[0])
    Q_0 = Q
    time_steps = 1000
    scores = None
    # E[y1] = C*E[x1] = C*A*E[x0]
    # Var[y1] = C**2*Var[x1] + R**2
    # E[yt] = C*E[xt] = C*A*E[x_{t-1}] = C*A**t*E[x0]
    # Var[yt] = C**2*Var[xt] + R**2 = C**2*(A**2*Var[x_{t-1}]+Q**2) + R**2 = C**2*(A**2*(A**2*Var[x_{t-2}] + Q**2) + Q**2)+R**2
    # = C**2*(A**(2*t)*Var[x_0] + Q**2) + R**2 = C**2*Q**2*(A**(2t+2)-1)/(A**2-1) + R**2
    for t in range(1, time_steps+1):
        if scores is None:
            scores = logp(ys, t, C=gen_C, A=gen_A, Q=gen_Q, R=gen_R, mu=gen_mu_0).reshape(-1,1)
        else:
            scores = torch.cat([scores, logp(ys, t, C=gen_C, A=gen_A, Q=gen_Q, R=gen_R, mu=gen_mu_0).reshape(-1,1)], dim=1)
    plt.imshow(scores)#, cmap='hot', interpolation='nearest')
    plt.show()
    print(scores)

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

    running_log_evidence_estimates, xts, states, actions, priors, liks = evaluate(ys, pd, env)
    return ys, running_log_evidence_estimates

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
    # _, policy = load_rl_model(ys.device)
    # print('\nrl output')
    # for s, x, y, prior, lik in zip(states, xs, ys, priors, liks):
    #     print('s_t = {} a_t = {} y_t = {} where p(a_t|s_t) = N({}, {}) = {} and p(y_t|a_t) = N({}, {}) = {}'.format(
    #         s.item(), x.item(), y.item(), (A*s).item(), Q.item(), prior.item(), (C*x).item(), R.item(), lik.item()))
    # joint_p = torch.tensor(liks).sum() + torch.tensor(priors).sum()
    # log_qrobs = torch.zeros(len(states))
    # for j in range(len(states)):
    #     state = states[j]
    #     action = xs[j]
    #     y = ys[j]
    #     log_qrobs[j] = policy.evaluate_actions(obs=torch.cat([state, y]).reshape(1, -1), actions=action)[1].item()
    #     print('log_qrob state: {}'.format(state))
    #     print('log_qrob action: {}'.format(action))
    # print('log qrobs: {}'.format(log_qrobs))
    # print('evidence estimate: {}'.format(joint_p - torch.sum(log_qrobs)))

    # print('\ntesting covariance')
    # test_covariance(ys)

def plot_log_diffs(log_values, log_true, label):
    # compute log ratio of estimate to true value
    diffs = torch.tensor(log_values) - log_true
    plt.plot(torch.arange(1, len(diffs.squeeze())+1), diffs.squeeze(), label=label)

    return diffs

def plot_IS(env_gen, sample_fun, get_true_log_prob_fun, name, traj_lengths=[10], extra_fun=None): #traj_lengths=[1, 10, 50]):
    for traj_length in traj_lengths:
        plt.figure(plt.gcf().number+1)

        # generate ys from gen_A, gen_Q params
        ys = sample_fun(traj_length)

        # get evidence estimate using true params
        log_true = get_true_log_prob_fun(ys)

        # generate environment
        env = env_gen(ys)

        # get evidence estimates using IS with other params
        As = torch.arange(0.2, 0.6, 0.2)
        Qs = torch.arange(0.2, 0.6, 0.2)
        for _A in As:
            for _Q in Qs:
                # get is estimate of evidence using _A and _Q params
                _, log_estimates = importance_estimate(ys, A=_A.reshape(1, 1), Q=_Q.reshape(1, 1), env=env)
                print('IS estimate: {}', log_estimates[-1])

                plot_log_diffs(log_estimates, log_true, label='A: {}, Q: {}'.format(round(_A.item(), 1), round(_Q.item(), 1)))

        # add RL plot
        try:
            running_estimate, _, _, _, _, _ = rl_estimate(ys, env)
            print('rl estimate: {}', running_estimate[-1])
            plot_log_diffs(running_estimate, log_true, label='RL')
        except:
            pass

        if extra_fun is not None:
            extra_fun(ys)

        plt.xlabel('Number of Samples')
        plt.ylabel('log Ratio of Evidence Estimate with True Evidence')
        plt.title('Convergence of Evidence Estimate to True Evidence (trajectory length: {})'.format(traj_length))
        plt.legend()
        plt.savefig('/home/jsefas/linear-gaussian-model/traj_length_{}_{}_convergence.png'.format(traj_length, name))

def plot_evidence_IS(env):
    sample_fun = lambda num_steps: generate_trajectory(num_steps)[0]
    get_true_log_prob_fun = lambda ys: importance_estimate(ys, A=gen_A, Q=gen_Q)[1]

    def env_gen(ys):
        return LinearGaussianEnv(A=gen_A, Q=gen_Q,
                                 C=gen_C, R=gen_R,
                                 mu_0=gen_mu_0,
                                 Q_0=gen_Q_0, ys=ys,
                                 traj_length=len(ys),
                                 sample=False)

    plot_IS(env_gen=env_gen, sample_fun=sample_fun, get_true_log_prob_fun=get_true_log_prob_fun, name="Evidence")

def plot_event_IS(num_steps=10, threshold=torch.tensor(0.5)):
    d = y_dist(num_steps)
    print('true dist: N({}, {})'.format(d.mean.item(), d.variance.item()))

    def sample_ys_fun(num_steps):
        ys, score, d = sample_y(num_steps=num_steps)
        return (ys > threshold).type(ys.dtype)

    def sample_true_fun(ys):
        return dist.Bernoulli(1-d.cdf(threshold)).log_prob(ys)

    def env_gen(ys):
        return LinearGaussianSingleYEnv(A=gen_A, Q=gen_Q,
                                        C=gen_C, R=gen_R,
                                        mu_0=gen_mu_0,
                                        Q_0=gen_Q_0, ys=ys,
                                        traj_length=num_steps,
                                        sample=False,
                                        threshold=threshold)

    def extra_fun(ys):
        p_gt_h = dist.Bernoulli(1-d.cdf(threshold)).log_prob(ys)
        print('true probability: {}', p_gt_h)

    plot_IS(env_gen=env_gen, sample_fun=sample_ys_fun, get_true_log_prob_fun=sample_true_fun, name="event", extra_fun=extra_fun)

def compare_y_event_dist(num_steps=10):
    y_saps = []
    for i in range(10000):
        ys, _, _, _ = generate_trajectory(num_steps=num_steps)
        y_saps.append(ys[-1])
    y_saps = torch.tensor(y_saps)
    print('empirical dist: N({}, {})'.format(y_saps.mean(), y_saps.var()))


if __name__ == "__main__":
    # plot_evidence_IS()

    plot_event_IS()

    # ys = None
    # traj_length=10
    # full_sweep(ys, train_model=True, traj_length=traj_length)

    # importance estimate
    # ys, log_evidence_estimates = test_importance_sampler(traj_length=traj_length, A=torch.tensor(0.6).reshape(1, -1), Q=torch.tensor(0.1).reshape(1, -1))

    # monte carlo estimate
    # _, log_evidence_mc = importance_estimate(ys, A=gen_A, Q=gen_Q)

    # compute log ratio of IS estimate to MC value
    # diffs = log_evidence_estimates - log_evidence_mc[-1]
    # plt.plot(diffs, range(1, len(diffs)+1))

    # # rl estimate
    # rl_estimate(ys)
