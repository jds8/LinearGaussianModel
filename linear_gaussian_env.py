# imports
import math
import gym
import torch
import torch.distributions as dist
import numpy as np
from generative_model import y_dist, sample_y, generate_trajectory, score_y, score_initial_state, score_state_transition
from linear_gaussian_prob_prog import GaussianRandomVariable, LinearGaussian


class AbstractLinearGaussianEnv(gym.Env):
    def __init__(self, A, Q, C, R, mu_0, Q_0, traj_length=1, ys=None, sample=False):
        # define action space
        self.action_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(mu_0.shape[0],), dtype=float)

        self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(mu_0.shape[0] + R.shape[0], 1), dtype=float)

        # data
        self.ys = ys
        y_len = len(ys) if ys is not None else 0
        self.traj_length = max(y_len, traj_length)
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
        self.liks = []
        self.states = []
        self.actions = []

    def compute_conditionals(self):
        self.w = GaussianRandomVariable(mu=0., sigma=torch.sqrt(self.Q), name='w')
        self.v = GaussianRandomVariable(mu=0., sigma=torch.sqrt(self.R), name='v')
        xt = GaussianRandomVariable(mu=self.mu_0, sigma=torch.sqrt(self.Q_0), name='x0')
        self.xs = [xt]
        self.ys = []
        for i in range(self.traj_length):
            yt = LinearGaussian(a=self.C, x=xt, b=self.v, name="y")
            xt = LinearGaussian(a=self.A, x=xt, b=self.w, name='x')
            self.xs.append(xt)
            self.ys.append(yt)

    def compute_joint(self):
        self.compute_conditionals()
        joint = self.xs[0].prior() * self.ys[0].likelihood()
        for i in range(1, self.traj_length):
            joint *= self.xs[i].likelihood()
            joint *= self.ys[i].likelihood()
        return joint

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
        self.liks.append(lik_reward)

        self.index += 1

        # score next state against prior
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

        prev_state_shape = self.prev_xt.nelement()
        self.prev_state = torch.cat([self.prev_xt.reshape(prev_state_shape, 1), first_y.reshape(-1, 1)])

        return self.prev_state


class LinearGaussianEnv(AbstractLinearGaussianEnv):
    def __init__(self, A, Q, C, R, mu_0, Q_0, traj_length=1, ys=None, sample=False):
        # define observation sapce
        # self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(traj_length+1, 1), dtype=float)

        super().__init__(A, Q, C, R, mu_0, Q_0, traj_length=traj_length, ys=ys, sample=sample)

    def compute_lik_reward(self, xt):
        # get y test and increment index
        ytest = self.ys[self.index]
        # score next observation (ytest) against the likelihood distribution
        return score_y(ytest, xt, self.C, self.R)

    def get_next_y(self, done):
        return self.ys[self.index] if not done else torch.zeros_like(self.ys[0])

    def generate(self):
        return generate_trajectory(self.traj_length, A=self.A, Q=self.Q, C=self.C, R=self.R, mu_0=self.mu_0, Q_0=self.Q_0)[0]


class LinearGaussianSingleYEnv(AbstractLinearGaussianEnv):
    def __init__(self, A, Q, C, R, mu_0, Q_0, traj_length=1, ys=None, sample=False, event_prob=0.2):
        # define observation sapce
        self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(mu_0.shape[0] + R.shape[0], 1), dtype=float)

        super().__init__(A, Q, C, R, mu_0, Q_0, traj_length=traj_length, ys=ys, sample=sample)

        if isinstance(event_prob, torch.Tensor):
            self.event_prob = event_prob
        else:
            self.event_prob = torch.tensor(event_prob)

    def compute_lik_reward(self, xt):
        if self.index < (self.traj_length - 1):
            return torch.zeros_like(self.ys)
        else:
            d = dist.Normal(self.C*xt, torch.sqrt(self.R))
            return dist.Bernoulli(self.event_prob).log_prob(self.ys)

    def get_next_y(self, done):
        return self.ys if not done else torch.zeros_like(self.ys)

    def generate(self):
        y, score, d = sample_y(self.traj_length)
        threshold = d.icdf(1-self.event_prob)
        return (y > threshold).type(y.dtype)
