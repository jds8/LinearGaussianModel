# imports
import math
import gym
import torch
import torch.distributions as dist
import numpy as np
from generative_model import y_dist, sample_y, generate_trajectory, score_y, score_initial_state, score_state_transition


class AbstractLinearGaussianEnv(gym.Env):
    def __init__(self, A, Q, C, R, mu_0, Q_0, traj_length=1, threshold=0.5, ys=None, sample=False):
        # define action space
        self.action_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(mu_0.shape[0],), dtype=float)

        # define observation sapce
        self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(mu_0.shape[0] + R.shape[0], 1), dtype=float)

        # data
        self.ys = ys
        self.traj_length = max(len(ys), traj_length)
        self.sample = sample

        # current index into data and max index
        self.index = 0

        # true parameters
        self.A = A
        self.Q = Q
        self.C = C
        self.R = R
        self.mu_0 = mu_0
        self.Q_0 = Q_0

        # store previous hidden state xt
        self.prev_state = None
        self.prev_xt = None

        self.p_log_probs = []
        self.states = []
        self.actions = []

    def compute_lik_reward(self, xt):
        raise NotImplementedError
        return None

    def get_next_y(self, done):
        raise NotImplementedError
        return None

    def step(self, action):
        # cast action to the appropriate torch.tensor and dtype
        if isinstance(action, torch.Tensor):
            xt = action.type(torch.float32)
        else:
            xt = torch.tensor(action, dtype=torch.float32)

        # get liklihood score
        lik_reward = self.compute_lik_reward(xt)

        self.index += 1

        # score next state against prior
        if self.index == 1:
            prior_reward = score_initial_state(xt, self.mu_0, self.Q_0)
        else:
            prior_reward = score_state_transition(xt, self.prev_xt, self.A, self.Q)
        self.p_log_probs.append(prior_reward)
        self.actions.append(xt)
        self.states.append(self.prev_state)
        reward = lik_reward.sum() + prior_reward.sum()

        # check done
        done = self.index >= self.traj_length

        # add p(y_i|x_i), p(x_i|x_{i-1}), x_i, x_{i-1} to info for future estimates
        info = {'prior_reward': prior_reward,
                'lik_reward': lik_reward,
                'action': xt,
                'xt': self.prev_xt}

        # get next y
        yout = self.get_next_y(done)

        # update previous xt
        self.prev_xt = xt

        self.prev_state = torch.cat([self.prev_xt.reshape(-1, 1), yout.reshape(-1, 1)])

        # return stuff
        return self.prev_state, reward.item(), done, info

    def generate(self):
        raise NotImplementedError
        return None

    def reset(self):
        # set initial observation to be 0s
        self.prev_xt = torch.zeros_like(self.mu_0)
        self.index = 0
        self.states = []
        self.actions = []
        self.p_log_probs = []

        if self.sample:
            self.ys = self.generate()
        first_y = self.ys[0]

        self.prev_state = torch.cat([self.prev_xt, first_y.reshape(-1, 1)])

        return self.prev_state


class LinearGaussianEnv(AbstractLinearGaussianEnv):
    def __init__(self, A, Q, C, R, mu_0, Q_0, traj_length=1, ys=None, sample=False):
        super().__init__(A, Q, C, R, mu_0, Q_0, traj_length=traj_length, ys=ys, sample=sample)

    def compute_lik_reward(self, xt):
        # get y test and increment index
        ytest = self.ys[self.index]
        # score next observation (ytest) against the likelihood distribution
        return score_y(ytest, xt, self.C, self.R)

    def get_next_y(self, done):
        return self.ys[self.index] if not done else torch.zeros_like(self.ys[0])

    def generate(self):
        return generate_trajectory(self.traj_length)


class LinearGaussianSingleYEnv(AbstractLinearGaussianEnv):
    def __init__(self, A, Q, C, R, mu_0, Q_0, traj_length=1, ys=None, sample=False, threshold=0.5):
        super().__init__(A, Q, C, R, mu_0, Q_0, traj_length=traj_length, ys=ys, sample=sample)

        d = y_dist(self.traj_length, A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0)
        # Note: I am transforming this distribution from MultivariateNormal to Normal so
        # that I can compute the cdf. Moreover, the second parameter of a MultivariateNormal
        # is the covariance whereas for a Normal, it's the standard deviation
        self.y_dist = dist.Normal(d.mean, torch.sqrt(d.variance))

        if isinstance(threshold, torch.Tensor):
            self.threshold = threshold
        else:
            self.threshold = torch.tensor(threshold)

    def compute_lik_reward(self, xt):
        if self.index < (self.traj_length - 1):
            return torch.zeros_like(self.ys)
        else:
            return dist.Bernoulli(1-self.y_dist.cdf(self.threshold)).log_prob(self.ys)

    def get_next_y(self, done):
        return self.ys if not done else torch.zeros_like(self.ys)

    def generate(self):
        y, score, d = sample_y(self.traj_length)
        return y>self.threshold
