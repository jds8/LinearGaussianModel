# imports
import math
import gym
import torch
import torch.distributions as dist
import numpy as np
from generative_model import y_dist, sample_y, generate_trajectory, score_y, score_initial_state, score_state_transition
from linear_gaussian_prob_prog import GaussianRandomVariable, LinearGaussian
from filtering_posterior import _compute_conditional_filtering_posteriors


class AllObservationsAbstractLinearGaussianEnv(gym.Env):
    def __init__(self, A, Q, C, R, mu_0, Q_0, condition_length, using_entropy_loss=False, traj_length=1, ys=None, sample=False):
        # data
        self.ys = ys
        y_len = len(ys) if ys is not None else 0
        self.traj_length = max(y_len, traj_length)
        self.sample = sample

        # number of observations on which to condition
        self.condition_length = condition_length if condition_length > 0 else self.traj_length

        # define action space
        self.action_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(mu_0.shape[0],), dtype=float)

        self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(mu_0.shape[0] + self.condition_length * R.shape[0], 1), dtype=float)

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
        self.rewards = []
        self.states = []
        self.actions = []
        self.prev_xts = [self.prev_xt]

        # whether we are using entropy or KL regularization
        self.using_entropy_loss = torch.tensor(using_entropy_loss, dtype=torch.float32)

        # compute analytical filtering distributions
        self.tds = None

        # keep track of the ys we are conditioning on (previous_condition_ys is for logging purposes)
        self.condition_ys = None
        self.previous_condition_ys = None

    def compute_conditionals(self):
        self.w = GaussianRandomVariable(mu=0., sigma=torch.sqrt(self.Q), name='w')
        self.v = GaussianRandomVariable(mu=0., sigma=torch.sqrt(self.R), name='v')
        xt = GaussianRandomVariable(mu=self.mu_0, sigma=torch.sqrt(self.Q_0), name='x0')
        self.traj_xs = [xt]
        self.traj_ys = []
        for i in range(self.traj_length):
            yt = LinearGaussian(a=self.C, x=xt, b=self.v, name="y")
            xt = LinearGaussian(a=self.A, x=xt, b=self.w, name='x')
            self.traj_xs.append(xt)
            self.traj_ys.append(yt)

    def compute_joint(self):
        self.compute_conditionals()
        joint = self.traj_xs[0].prior() * self.traj_ys[0].likelihood()
        for i in range(1, self.traj_length):
            joint *= self.traj_xs[i].likelihood()
            joint *= self.traj_ys[i].likelihood()
        return joint

    def compute_lik_reward(self, xt):
        raise NotImplementedError
        return None

    def step(self, action):
        # cast action to the appropriate torch.tensor and dtype
        if isinstance(action, torch.Tensor):
            xt = action.type(torch.float32)
        else:
            xt = torch.tensor(action, dtype=torch.float32)

        # get likelihood score
        lik_reward = self.compute_lik_reward(xt)
        self.liks.append(lik_reward)

        self.index += 1

        # score next state against prior
        prior_reward = score_state_transition(xt, self.prev_xt, self.A, self.Q)
        self.p_log_probs.append(prior_reward)
        self.actions.append(xt)
        self.states.append(self.prev_state)
        reward = lik_reward.sum() + self.using_entropy_loss * prior_reward.sum()
        self.rewards.append(reward)

        # check done
        done = self.get_done()

        # add p(y_i|x_i), p(x_i|x_{i-1}), x_i, x_{i-1} to info for future estimates
        info = {'prior_reward': prior_reward,
                'lik_reward': lik_reward,
                'action': xt,
                'xt': self.prev_xt}

        self.previous_condition_ys = self.condition_ys
        self.condition_ys = self.get_condition_ys()

        # update previous xt
        self.prev_xt = xt
        self.prev_xts.append(self.prev_xt)

        self.prev_state = torch.cat([self.prev_xt.reshape(-1, 1), self.condition_ys.reshape(-1, 1)])

        # return stuff
        return self.prev_state, reward.item(), done, info

    def generate(self):
        raise NotImplementedError
        return None

    def get_condition_ys(self):
        raise NotImplementedError

    def reset(self):
        # set initial observation to be 0s
        self.prev_xt = torch.zeros_like(self.mu_0)
        self.index = 0
        self.states = []
        self.actions = []
        self.p_log_probs = []
        self.liks = []
        self.rewards = []
        self.prev_xts = [self.prev_xt]

        if self.sample:
            self.ys = self.generate()

        # create filtering distribution given the ys
        self.tds = _compute_conditional_filtering_posteriors(A=self.A, Q=self.Q, C=self.C, R=self.R, mu_0=self.mu_0, Q_0=self.Q_0,
                                                             num_obs=len(self.ys), dim=self.prev_xt.nelement(),
                                                             m=self.condition_length, ys=self.ys)

        prev_state_shape = self.prev_xt.nelement()

        # set condition_ys and previous_condition_ys to be the same upon reset
        self.condition_ys = self.get_condition_ys()
        self.previous_condition_ys = self.condition_ys

        # see observation space in __init__ for details on why prev_state is defined this way
        self.prev_state = torch.cat([self.prev_xt.reshape(prev_state_shape, 1), self.condition_ys.reshape(-1, 1)])

        return self.prev_state


class AbstractConditionalLinearGaussianEnv(AllObservationsAbstractLinearGaussianEnv):
    def __init__(self, A, Q, C, R, mu_0, Q_0, using_entropy_loss, condition_length, traj_length=1, ys=None, sample=False):
        super().__init__(A, Q, C, R, mu_0, Q_0, condition_length, using_entropy_loss,
                         traj_length=traj_length, ys=ys, sample=sample)

    def compute_lik_reward(self, xt):
        # get y test and increment index
        ytest = self.ys[self.index]
        # score next observation (ytest) against the likelihood distribution
        return score_y(ytest, xt, self.C, self.R)

    def generate(self):
        return generate_trajectory(self.traj_length, A=self.A, Q=self.Q, C=self.C, R=self.R, mu_0=self.mu_0, Q_0=self.Q_0)[0]


class AllObservationsLinearGaussianEnv(AbstractConditionalLinearGaussianEnv):
    def __init__(self, A, Q, C, R, mu_0, Q_0, using_entropy_loss, condition_length, traj_length=1, ys=None, sample=False):
        super().__init__(A, Q, C, R, mu_0, Q_0, condition_length, using_entropy_loss,
                         traj_length=traj_length, ys=ys, sample=sample)

    def get_condition_ys(self):
        return self.ys

    def get_done(self):
        done = self.index >= self.traj_length


class ConditionalObservationsLinearGaussianEnv(AbstractConditionalLinearGaussianEnv):
    def __init__(self, A, Q, C, R, mu_0, Q_0, using_entropy_loss, condition_length, traj_length=1, ys=None, sample=False):
        super().__init__(A, Q, C, R, mu_0, Q_0, condition_length, using_entropy_loss,
                         traj_length=traj_length, ys=ys, sample=sample)

    def get_condition_ys(self):
        return self.ys[self.index:self.index+self.condition_length]

    def get_done(self):
        return self.index+self.condition_length > self.traj_length
