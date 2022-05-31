# imports
import gym
from stable_baselines3 import PPO
import torch
import torch.distributions as dist
import torch.nn as nn
import numpy as np
from copy import deepcopy
import time
from typing import List

def state_transition_from_dist(distribution):
    """ Computes a state transition from x_t using distribution """
    return distribution.rsample()

def state_transition(x_t, A, Q):
    """
    Computes state transition from x_t defined by x_{t+1} = Ax_t + w
    where $w \sim N(0, Q)$
    """
    return state_transition_from_dist(dist.MultivariateNormal(A*x_t, Q))

def score_state_transition(xt, prev_xt, A, Q):
    """ Scores xt against the prior N(A*prev_xt, Q) """
    return dist.MultivariateNormal(A*prev_xt, Q).log_prob(xt)

def score_y_from_dist(y_test, distribution):
    """ Scores y_test against a distribution object """
    return distribution.log_prob(y_test)

def score_y(y_test, x_t, C, R):
    """
    Scores y_test under the likelihood defined by y_t = Ax_t + v
    where $v \sim N(0, R)$
    """
    return score_y_from_dist(y_test, dist.MultivariateNormal(C*x_t, R))

def neg_two_log_prob(xs, ys, A, Q, C, R, mu0, Q0):
    assert len(xs) == len(ys)

    tau = len(xs)
    p, k = C.shape

    y_term = ys - C*xs
    r_term = y_term.t() * torch.inverse(R) * y_term + torch.logdet(R)*tau

    q_term = torch.tensor(0.)
    for i in range(len(xs)-1):
        xt = xs[i]
        xt_plus_1 = xs[i+1]
        x_term = xt_plus_1 - A*xt
        q_term += x_term.t() * torch.inverse(Q) * x_term + torch.logdet(Q)

    x0_term = xs[0] - mu0
    zero_term = x0_term.t() * torch.inverse(Q0) * x0_term + tau * (p + k) * torch.log(2*torch.tensor(torch.pi))
    return r_term + q_term + zero_term

def log_joint(xs, ys, A, Q, C, R, mu0, Q0):
    return -neg_two_log_prob(xs, ys, A, Q, C, R, mu0, Q0) / 2

def compute_joint(xs, ys, A, Q, C, R, mu0, Q0):
    return torch.exp(-neg_two_log_prob(xs, ys, A, Q, C, R, mu0, Q0) / 2)

def get_start_state(mu_0, Q_0):
    return dist.MultivariateNormal(mu_0, Q_0).rsample()

def evidence(ys, A, Q, C, R, mu0, Q0):
    return 0.


class LazyWrapper:
    def __init__(self, f):
        self.f = f

    def __mul__(self, x):
        def fun(*args, **kwargs):
            return x * self.f(*args, **kwargs)
        return LazyWrapper(fun)

    def __rmul__(self, x):
        return self * x

    def __add__(self, x):
        def fun(*args, **kwargs):
            return x + self.f(*args, **kwargs)
        return LazyWrapper(fun)

    def __radd__(self, x):
        return self + x

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)


class TorchDistributionInfo:
    def __init__(self, evaluate):
        self.evaluate = evaluate

    def __call__(self, arg):
        return self.evaluate(**arg)


class GaussianDistribution:
    def __init__(self, dist, left, right=None, is_posterior=False):
        self.dist = dist
        self.left = left
        self.right = right
        self.is_posterior = is_posterior
        self.is_conditional = isinstance(self.dist, TorchDistributionInfo)

    def __mul__(self, other):
        if self.is_posterior ^ other.is_posterior:
            return self.mul_posterior(other)

        assert (self.right in other.left) ^ (other.right in self.left)  # ensures forms like p(x|y)p(y)

        if self.right in other.left:
            # in the special case that len(left) == 1 and the RHS is dependent on the LHS,
            # then we have a (single-variate) posterior
            right = other.right
            left = other.left + self.left
            y_cv = self.covariance()
            x_cv = other.covariance()
            right_id = other.left.index(self.right)
            a_mat = torch.tensor([x.get_coef_wrt(self.right) for x in self.left]).view(1,-1)
        else:  # other.right in self.left in this case
            right = self.right
            left = self.left + other.left
            y_cv = other.covariance()
            x_cv = self.covariance()
            right_id = self.left.index(other.right)
            a_mat = torch.tensor([x.get_coef_wrt(other.right) for x in other.left]).view(1,-1)

        y_precision = torch.inverse(y_cv)
        x_precision = torch.inverse(x_cv)
        mod_y_precision = torch.zeros_like(x_precision)
        mod_y_precision[right_id, right_id] = torch.matmul(torch.matmul(a_mat, y_precision), a_mat.t())

        return self.create_distribution(x_precision, y_precision, mod_y_precision, a_mat, left, right, right_id)

    def mul_posterior(self, other):
        assert (self.right in other.left) ^ (other.right in self.left)  # ensures forms like p(x|y)p(y)
        print("WARNING: Calculating product of densities while assuming that precisely one term is a posterior and that, it conditions on a variable one step away in the graphical model.")

        if self.is_posterior:
            if self.right in other.left:
                right = other.right
                right_id = other.left.index(self.right)
                left = self.left + other.left

                y_precision = torch.inverse(self.right.conditional_variance(self.left[0]))
                x_precision = torch.inverse(self.left[0].prior().covariance())
                a_mat = torch.tensor(self.right.a).view(1,-1)
            else:
                right = self.right
                right_id = self.left.index(other.right)
                left = other.left + self.left

                y_precision = torch.inverse(other.covariance())
                x_precision = torch.inverse(self.covariance())
                a_mat = torch.tensor([x.get_coef_wrt(other.right) for x in other.left]).view(1,-1)
        else:
            # in this case, other.is_posterior
            if other.right in self.left:
                right = self.right
                right_id = self.left.index(other.right)
                left = other.left + self.left

                y_precision = torch.inverse(other.right.conditional_variance(other.left[0]))
                x_precision = torch.inverse(other.left[0].prior().covariance())
                a_mat = torch.tensor(other.right.a).view(1,-1)
            else:
                right = other.right
                right_id = other.left.index(self.right)
                left = self.left + other.left

                y_precision = torch.inverse(self.covariance())
                x_precision = torch.inverse(other.covariance())
                a_mat = torch.tensor([x.get_coef_wrt(self.right) for x in self.left]).view(1,-1)

        mod_y_precision = torch.matmul(torch.matmul(a_mat, y_precision), a_mat.t())

        return self.create_distribution(x_precision, y_precision, mod_y_precision, a_mat, left, right, right_id)

    # def mul_posterior(self, other):
    #     assert (self.right in other.left) ^ (other.right in self.left)  # ensures forms like p(x|y)p(y)
    #     print("WARNING: Calculating product of densities while assuming that precisely one term is a posterior and that, it conditions on a variable one step away in the graphical model.")

    #     if self.right in other.left:
    #         condition_dist = other
    #         right_id = other.left.index(self.right)
    #         left = self.left + other.left

    #         y_precision = torch.inverse(self.right.conditional_variance(None))
    #         x_precision = torch.inverse(self.left[0].prior().covariance())
    #         a_mat = torch.tensor(self.right.a).view(1,-1)
    #     else:
    #         condition_dist = self
    #         right_id = self.left.index(other.right)
    #         left = other.left + self.left

    #         y_precision = torch.inverse(other.right.conditional_variance(None))
    #         x_precision = torch.inverse(other.left[0].prior().covariance())
    #         a_mat = torch.tensor(other.right.a).view(1,-1)

    #     mod_y_precision = torch.matmul(torch.matmul(a_mat, y_precision), a_mat.t())

    #     return self.create_distribution(x_precision, y_precision, mod_y_precision, a_mat, left, condition_dist, right_id)

    @staticmethod
    def create_distribution(x_precision, y_precision, mod_y_precision,
                            a_mat, left, right, right_id):
        lambda_x_x = x_precision + mod_y_precision

        correlation_precision = -torch.matmul(a_mat, y_precision)
        if correlation_precision.nelement() > 1:
            lambda_x_y = correlation_precision
            lambda_y_x = correlation_precision.t()
        else:
            lambda_x_y = torch.zeros(1, lambda_x_x.shape[0])
            lambda_x_y[0, right_id] = correlation_precision
            lambda_y_x = torch.zeros(1, lambda_x_x.shape[0])
            lambda_y_x[0, right_id] = correlation_precision.t()

        # hackery
        if lambda_x_x.nelement() == 1:
            lambda_x_x = lambda_x_x.reshape(1,1)
        if lambda_x_y.nelement() == 1:
            lambda_x_y = lambda_x_y.reshape(1,1)
        if lambda_y_x.nelement() == 1:
            lambda_y_x = lambda_y_x.reshape(1,1)
        if y_precision.nelement() == 1:
            y_precision = y_precision.reshape(1,1)

        # create appropriate precision matrix
        try:
            precision = torch.cat([torch.cat([lambda_x_x, lambda_x_y.t()], dim=1), torch.cat([lambda_y_x, y_precision.t()], dim=1)], axis=0)
        except:
            precision = torch.cat([torch.cat([lambda_x_x, lambda_x_y.t()], dim=0).t(), torch.cat([lambda_y_x, y_precision.t()], dim=1)], axis=0)
        # print('precision: {}'.format(precision))
        # print('covavriance: {}'.format(torch.inverse(precision)))

        # set means
        if right is None:
            mean = torch.cat([x.mu.reshape(-1) for x in left])
            prob_dist = dist.MultivariateNormal(mean, torch.inverse(precision))
        else:
            pm = lambda_x_x * (x_precision * left[0].mu + torch.matmul(torch.matmul(a_mat, y_precision), torch.tensor(1.).reshape(1,-1)))
            # print('lambda_x_x: {}'.format(lambda_x_x))
            # print('a: {}'.format(a_mat))
            # print('y_precision: {}'.format(y_precision))
            # print('x_precision: {}'.format(x_precision))
            # print('x0.mu: {}'.format(left[0].mu))
            # print('x1.mu: {}'.format(left[1].mu))
            # print('posterior mean: {}'.format(pm))
            def condition_on(value):
                value = torch.tensor(value).reshape(1,-1)
                mean_list = []
                for x in left:
                    if x.is_dependent_on(right):
                        # This happens when right is an x, i.e. when p(.|xs)
                        mean_list.append((x.get_coef_wrt(right)*value).reshape(-1))
                    else:
                        # This happens when right is a y or a *subsequent* x, i.e. when p(.|ys) or p(x_{t-1}|x_t)
                        posterior_mean = lambda_x_x * (x_precision * x.mu + torch.matmul(torch.matmul(a_mat, y_precision), value))

                        mean_list.append(posterior_mean.reshape(-1))
                return dist.MultivariateNormal(torch.cat(mean_list), torch.inverse(precision))
            prob_dist = TorchDistributionInfo(condition_on)
        return GaussianDistribution(prob_dist, left=left, right=right)

    def marginalize_out_helper(self, var, left, mean_fun, covariance_fun, value=None):
        if isinstance(var, list):
            new_dist = self
            for v in var:
                new_dist, left = new_dist.marginalize_out_helper(v, left, mean_fun, covariance_fun, value)
            return new_dist, left
        else:
            assert var in left

            # remove var from self.left
            idx = left.index(var)
            del left[idx]

            # remove var from mean
            mu = mean_fun(value=value).clone()
            first_part = mu[:idx]
            second_part = mu[idx+1:] if idx+1 < len(mu) else torch.tensor([])
            mu = torch.cat([first_part, second_part])

            # remove var from covariance
            covariance = covariance_fun().clone()
            mask = torch.tensor([i for i in range(len(covariance)) if i != idx])
            covariance = torch.index_select(covariance, 0, mask)
            covariance = torch.index_select(covariance, 1, mask)

            # create normal dist
            if len(left) == 1:
                prob_dist = dist.Normal(mu, torch.sqrt(covariance))
            else:
                prob_dist = dist.MultivariateNormal(mu, covariance)
            return prob_dist, left

    def marginalize_out(self, var):
        left = self.left.copy()
        if isinstance(var, list):
            for v in var:
                idx = left.index(v)
                del left[idx]
        else:
            idx = left.index(var)
            del left[idx]

        if self.is_conditional:
            prob_dist = TorchDistributionInfo(lambda value: self.marginalize_out_helper(var, self.left.copy(), self.mean, self.covariance, value)[0])
        else:
            prob_dist, left = self.marginalize_out_helper(var, self.left.copy(), self.mean, self.covariance)

        return GaussianDistribution(prob_dist, left=left, right=self.right)

    def get_dist(self, **kwargs):
        if self.is_conditional:
            return self.dist(kwargs)
        return self.dist

    def sample(self, **kwargs):
        return self.get_dist(**kwargs).sample()

    def mean(self, **kwargs):
        return self.get_dist(**kwargs).mean

    def covariance(self, **kwargs):
        """
        Computes covariance of this distribution
        (Note that the variance does *not* depend on parameter values in linear gaussian setting)
        """
        dummy = 1.  # don't need a value in order to call covariance
        prob_dist = self.get_dist(value=dummy)
        if isinstance(prob_dist, dist.MultivariateNormal):
            return prob_dist.covariance_matrix
        return prob_dist.stddev**2

    def precision(self, **kwargs):
        return torch.inverse(self.covariance(self, **kwargs))

class GaussianRandomVariable:
    x_ids = 0
    y_ids = 1
    def __init__(self, mu, sigma, observed=False, name=""):
        self.mu = torch.tensor(mu)
        self.sigma = torch.tensor(sigma)
        self.observed = observed
        if name == "x":
            self.name = 'x{}'.format(GaussianRandomVariable.x_ids)
            GaussianRandomVariable.x_ids += 1
        elif name == "y":
            self.name = 'y{}'.format(GaussianRandomVariable.y_ids)
            GaussianRandomVariable.y_ids += 1
        else:
            self.name = 'intermediate'

    def __mul__(self, a):
        if isinstance(a, torch.Tensor):
            return GaussianRandomVariable(self.mu * a, torch.sqrt(self.sigma**2 * a * a))
        # if isinstance(a, GaussianRandomVariable):
        #     if a.observed:
        #         return self * a.mu
        #     # assuming independent summands
        #     print('WARNING: assuming independence')
        #     mu = self.mu * a.mu
        #     var = (self.sigma**2 + self.mu**2) * (a.sigma**2 + a.mu**2) - self.mu**2 * a.mu**2
        #     return GaussianRandomVariable(mu, torch.sqrt(var))
        raise NotImplementedError

    def __rmul__(self, a):
        return self * a

    def __add__(self, b):
        if isinstance(b, torch.Tensor):
            return GaussianRandomVariable(self.mu + b, self.sigma)
        if isinstance(b, GaussianRandomVariable):
            if b.observed:
                return self + b.mu
            # assuming independent summands
            print('WARNING: assuming independence')
            return GaussianRandomVariable(self.mu + b.mu, torch.sqrt(self.sigma**2 + b.sigma**2))
        raise NotImplementedError

    def __radd__(self, b):
        return self + b

    def observe(self, value):
        mu = torch.tensor(value)
        sigma = torch.tensor(0.)
        observed = True
        return GaussianRandomVariable(mu, sigma, observed)

    def prior(self):
        return GaussianDistribution(dist=dist.Normal(self.mu, self.sigma), left=[self])

    def conditional_variance(self, var=None):
        if var is None:
            return self.sigma**2
        raise NotImplementedError

    def is_dependent_on(self, var):
        return False

    def get_coef_wrt(self, var):
        if self == var:
            return torch.tensor(self == var, dtype=torch.float32).reshape(1,-1)
        raise NotImplementedError


class LinearGaussian(GaussianRandomVariable):
    def __init__(self, a, x: GaussianRandomVariable, b: GaussianRandomVariable, name):
        assert isinstance(x, GaussianRandomVariable)
        assert isinstance(b, GaussianRandomVariable)
        var = a * x + b
        super(LinearGaussian, self).__init__(var.mu, var.sigma, name=name)
        self.a = a
        self.x = x
        self.b = b

    def is_dependent_on(self, var: GaussianRandomVariable):
        if var == self.x or var == self:
            return True
        return self.x.is_dependent_on(var)

    def get_coef_wrt(self, var: GaussianRandomVariable):
        if var == self:
            return torch.tensor(1.)
        if var == self.x:
            return self.a
        return self.a * self.x.get_coef_wrt(var)

    def likelihood(self):
        """
        The likelihood function for this variable P(Y|X).
        Note that this function returns a distribution object which
        can be evaluated at a particular value of X=x but can also
        be used to compute joint distributions.
        """
        a = self.a
        x = self.x
        b = self.b
        def lik(value):
            var = a * x.observe(value) + b
            return dist.Normal(var.mu, var.sigma)
        prob_dist = GaussianDistribution(TorchDistributionInfo(lik), left=[self], right=x)
        return prob_dist

    def posterior(self):
        """
        The posterior function for this variable P(X|Y).
        Note that this function returns a distribution object which
        can be evaluated at a particular value of Y=y but can also
        be used to compute joint distributions.
        """
        if (self.x.sigma <= torch.tensor(0.)).all() or (self.b.sigma <= torch.tensor(0.)).all():
            raise Exception("Cannot compute posterior if either x or y are deterministic.")

        a = self.a
        x = self.x
        b = self.b.mu
        sigma = self.b.sigma

        inv_sigma_x = torch.inverse(x.sigma**2)
        inv_sigma_y = torch.inverse(sigma**2)

        inv_sigma_x_given_y = inv_sigma_x + a * inv_sigma_y * a
        sigma_x_given_y = torch.inverse(inv_sigma_x_given_y)
        sigma_term = torch.sqrt(sigma_x_given_y) if sigma_x_given_y.nelement() == 1 else sigma_x_given_y

        mu_x_given_y = sigma_x_given_y * (a * inv_sigma_y * (1. - b) + inv_sigma_x * x.mu)
        # print('sigma_x_given_y: {}'.format(sigma_x_given_y))
        # print('a: {}'.format(a))
        # print('inv_sigma_y: {}'.format(inv_sigma_y))
        # print('inv_sigma_x: {}'.format(inv_sigma_x))
        # print('x.mu: {}'.format(x.mu))
        # print('mu_x_given_y: {}'.format(mu_x_given_y))
        def post(value):
            mu_x_given_y = sigma_x_given_y * (a * inv_sigma_y * (value - b) + inv_sigma_x * x.mu)
            if sigma_x_given_y.nelement() == 1:
                prob_dist = dist.Normal(mu_x_given_y, sigma_term)
            else:
                prob_dist = dist.MultivariateNormal(mu_x_given_y, sigma_term)
            return prob_dist
        return GaussianDistribution(TorchDistributionInfo(post), left=[x], right=self, is_posterior=True)

    def conditional_variance(self, var):
        if self.x == var:
            return self.b.sigma**2
        if self.is_dependent_on(var) or var is None:
            return self.a**2 * self.x.conditional_variance(var) + self.b.sigma**2
        else:
            raise NotImplementedError


class LinearGaussianEnv(gym.Env):
    def __init__(self, ys, A, Q, C, R, mu_0, Q_0):
        # define action space
        self.action_space = gym.spaces.Box(low=-1., high=1., shape=(2,), dtype=float)

        # define observation sapce
        self.observation_space = gym.spaces.Box(low=-1., high=1., shape=(2,), dtype=float)

        # data
        self.ys = ys

        # current index into data and max index
        self.index = 0
        self.max_index = len(self.ys)

        # true parameters
        self.A = A
        self.Q = Q
        self.C = C
        self.R = R
        self.mu_0 = mu_0
        self.Q_0 = Q_0

        # store previous hidden state xt
        self.prev_xt = None

    def score_initial_state(self, xt):
        """ Scores xt against the prior N(mu_0, Q_0) """
        return dist.MultivariateNormal(self.mu_0, self.Q_0).log_prob(xt)

    def step(self, action):
        # get y test and increment index
        ytest = self.ys[self.index]
        self.index += 1

        # cast action to the appropriate torch.tensor and dtype
        xt = torch.tensor(action, dtype=ytest.dtype)

        # score next observation (ytest) against the likelihood distribution
        lik_reward = score_y(ytest, xt, self.C, self.R)
        # score next state against prior
        prior_reward = score_state_transition(xt, self.prev_xt, self.A, self.Q)
        reward = lik_reward.sum() + prior_reward.sum()

        # check done
        done = self.index >= self.max_index

        # add y to info but set prev_xt to current xt
        info = {}
        self.prev_xt = xt

        # return stuff
        return self.prev_xt, reward, done, info

    def reset(self):
        self.prev_xt = get_start_state(mu_0, Q_0)
        self.index = 1
        return self.prev_xt


class LinearGaussianPolicy(nn.Module):
    def __init__(self):
        pass

    def act(self):
        pass


class FixedNormal(torch.distributions.Normal):
    """
    Vectorize multivariate normal distribution (becuase pytorch natively has
    some issues vectorizing these / did when this code was originally made).
    This class inherits everything from .Normal and can use repareterization
    if desired.

    Attributes
    ----------
    ...

    Methods
    -------
    mode()
    returns mean of normal distribution with respect to some action. This is the
    mean action of a given policy using the feature space provided.

    log_probs()
    return the probability of an action conditional on the model of actions given
    states, and the standard deviation defined over each action index.

    entrop()
    return entropy of normal distribution, this should be averaged over a set
    of examples, as the indiviual one only returns this for one example per.

    std()
    returns std of normal distribution with respect to some action. This is the
    std action of a given policy using the feature space provided.
    """

    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean

    def std(self):
        return self.std


class DiagGaussian(nn.Module):
    """
    Something convert a generic feature space to a multivariation Diagnol
    Guassian distribution over actions. This class includes some tweaks that
    allow you to enforce specific types of initial distributions over actions
    like say a mean zero-standard deviation one action distribution (which is
    very much required to use most continuous control on policy methods).

    Attributes
    ----------
    num_inputs: int
    This defines the size of the flattened vector input which defines the output
    feature space from the previous layer.

    num_outputs: int
    Again, this distribution outputs a flat vector of actions, and this attrib
    defines the size of that vector. For our examples this is always size two,
    which is the size of the action space in carla.

    param_sd: [int,None]
    This will determine whether or not the standard deviation will be parameterized
    by some feature space in the same way that the mean is, or if we will not
    actually condition it as is done in many on policy methods like PPO/TRPO/A2C.

    zero_mean: [True,False]
    Indicates if we are to force the output of the linear layer to be mean zero
    standard deviation one. Again initialization is important in RL.

    Methods
    -------
    get_mean(x)
    returns mean of normal distribution with respect to some action. This is the
    mean action of a given policy using the feature space provided.

    get_std(sd_input, min_std)
    returns std of normal distribution with respect to some action. This is the
    std action of a given policy using the feature space provided.

    forward(x, sd_input)
    Returns a fixed normal distribution defined by the mean and sd as defined above.
    To reiterat, this takes a set of features and actually returns a parameterized
    distribution object.
    """

    def __init__(self, num_inputs, num_outputs, bounds=None, param_sd=None, zero_mean=True):
        super(DiagGaussian, self).__init__()
        # initializer
        init_ = lambda m: self._init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0))
        # set info
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        #
        if zero_mean:
            self.fc_mean.weight.data.mul_(0.0)
            self.fc_mean.bias.data.mul_(0.0)
        # # complex
        # self.fc_std = nn.Linear(param_sd, num_outputs, bias=True)
        # self.fc_std.weight.data.mul_(0.0)
        # self.fc_std.bias.data.mul_(0.0)
        # simple
        stdev_init = - 0.25 * torch.ones(num_outputs)
        self.logstd = torch.nn.Parameter(stdev_init)
        # should we bound input
        if bounds is not None:
            assert bounds[1] > bounds[0]
            self.bound = True
            self.action_scale = torch.FloatTensor(
                np.array((bounds[1] - bounds[0]) / 2.))
            self.action_bias = torch.FloatTensor(
                np.array((bounds[1] + bounds[0]) / 2.))
        else:
            self.bound = False

    @staticmethod
    def _init(module, weight_init, bias_init, gain=1):
        """ Used int mlp initialization. """
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module

    def get_mean(self, x):
        if not self.bound:
            action_mean = self.fc_mean(x)
        else:
            y_t = torch.tanh(self.fc_mean(x))
            action_mean = y_t * self.action_scale + self.action_bias
        return action_mean

    def get_std(self, sd_input, min_std=1e-12):
        if sd_input is not None:
            action_log_std = self.fc_std(sd_input)
            return action_log_std.exp() + min_std
        else:
            return self.logstd.exp() + min_std

    def forward(self, x, sd_input=None):
        return FixedNormal(self.get_mean(x),  self.get_std(sd_input))

    def to(self, device):
        if self.bound:
            self.action_scale = self.action_scale.to(device)
            self.action_bias = self.action_bias.to(device)
        return super(DiagGaussian, self).to(device)


# model name
MODEL = 'linear_gaussian_model'

# true params
ys = torch.rand(10, 2)
A = torch.eye(2)
Q = torch.eye(2)
C = torch.eye(2)
R = torch.eye(2)
mu_0 = torch.zeros_like(ys[0])
Q_0 = Q

def train():
    # create env
    env = LinearGaussianEnv(ys, A, Q, C, R, mu_0, Q_0)
    # network archictecture
    arch = [1024 for _ in range(3)]
    # create policy
    model = PPO('MlpPolicy', env, ent_coef=0.5, policy_kwargs=dict(net_arch=[dict(pi=arch, vf=arch)]))

    # train policy
    model.learn(total_timesteps=10000)

    # save model
    model.save(MODEL)


def evaluate():
    # load model
    model = torch.load(MODEL)
    # create env
    env = LinearGaussianEnv(ys, A, Q, C, R, mu_0, Q_0)
    # get first obs
    obs = env.reset()
    # collect joint p(x,y)
    joints = []
    # evidence estimate
    evidence_est = torch.tensor(0.)
    # evaluate N times
    N = 100
    for _ in range(N):
        # keep track of xs
        xs = []
        done = False
        while not done:
            xt = model.predict(obs, deterministic=True)[0]
            obs, reward, done, info = env.step(xt)
            xs.append(xt)
        log_num = log_joint(xs=xs, ys=ys, A=A, Q=Q, C=C, R=R, mu0=mu_0, Q0=Q_0)
        log_denom = model.evaluate_actions(obs=[torch.zeros_like(mu_0)]+xs[0:-1], actions=xs)
        evidence_est += torch.exp(log_num - log_denom)/N
    evidence_true = evidence()
    print('evidence estimate: {}'.format(evidence_est))
    print('true evidence: {}'.format(evidence_true))
    print('abs difference of evidence estimate and evidence: {}'.format(abs(evidence_true-evidence_est)))


def test():
    ys = torch.rand(10, 1)
    A = torch.tensor(2.).reshape(1,1)
    Q = torch.tensor(3.).reshape(1,1)
    C = torch.tensor(0.5).reshape(1,1)
    R = torch.tensor(2.).reshape(1,1)
    mu_0 = torch.ones_like(ys[0])
    Q_0 = Q

    num_transitions = 2
    xt = GaussianRandomVariable(mu_0, Q_0, name="x")
    w = GaussianRandomVariable(0., Q)
    v = GaussianRandomVariable(0., R)
    xs = [xt]
    ys = []
    for i in range(num_transitions):
        xt = LinearGaussian(A, xt, w, name="x")
        yt = LinearGaussian(C, xt, v, name="y")
        xs.append(xt)
        ys.append(yt)
        print('x name: {}'.format(xt.name))
        print('y name: {}'.format(yt.name))

    # x1 = Ax0 + w
    # x2 = Ax1 + w = A(Ax0 + w) + w = A^2x0 + Aw + w
    # E[x2] = A^2E[x0] = 4
    # Var[x2] = A^4Var[x0] + A^2Var[w] + Var[w] = 16*9 + 4*9 + 9 = 189
    # E[y2] = CE[x2] + v = CA^2E[x0] + v = 2
    #   Var[y2] = C^2Var[x2] + 4
    # Var[y2] = A^4Var[x0] + A^2Var[w] + Var[w] = 16*9 + 4*9 + 9
    print(xs[-2].mu)
    print('xs var: {}'.format(xs[-2].sigma**2))
    print(ys[-2].mu)
    print('ys var: {}'.format(ys[-2].sigma**2))
    # p(x0) * p(x1|x0) = p(x1,x0)

    x_dist = xs[0].prior() * xs[1].likelihood()
    # int p(x1,x0)dx0 = p(x1)
    x1 = x_dist.marginalize_out(xs[0])
    # E[x1] = AE[x0] + E[w] = 2*1 + 0 = 2
    # Var[x1] = A^2Var[x0] + Var[w] = 4*9 + 9 = 45
    print('x1 mean: {}'.format(x1.mean()))
    print('x1 cov: {}'.format(x1.covariance()))
    # p(y1|x1) * p(x1) = p(y1, x1)

    lik =ys[-2].likelihood()
    prob_dist = lik * x1
    print('p(y1).prior: {}'.format(ys[-2].prior().covariance()))
    print('p(x1,y1) mean: {}'.format(prob_dist.mean()))
    print('p(x1,y1) cov: {}'.format(prob_dist.covariance()))
    print('p(x1,y1) precision: {}'.format(torch.inverse(prob_dist.covariance())))

    # int p(y1, x1)dx1 = p(y1)
    # y1 = Cx1 + v
    # E[y1|x1] = Cx1 + 0 = Cx1
    # Var[y1|x1] = Var[v] = 4
    # E[y1] = CE[x1] + E[v] = C(AE[x0] + E[w]) + 0 = CA = 0.5(2*1 + 0) + 0 = 1
    # Var[y1] = C^2Var[x1] + Var[v] = C^2(A^2Var[x0] + Var[w]) + 4 = 45/4 + 4 = 15.25
    # C^2(A^2 9 + 9) + 4 = (1/4)*(4*9 + 9) + 4 = 45/4 + 4

    # p(y1)
    p1 = prob_dist.marginalize_out(x1.left)
    print('p(y1) mean: {}'.format(p1.mean()))
    print('p(y1) cov: {}'.format(p1.covariance()))

    # p(x1) * p(x2|x1) = p(x2,x1)
    x_dist = x1 * xs[2].likelihood()
    # int p(x2,x1)dx1 = p(x2)
    x2 = x_dist.marginalize_out(x1.left)
    print('p(x2) mean: {}'.format(x2.mean()))
    print('p(x2) cov: {}'.format(x2.covariance()))

    lik = ys[-1].likelihood()
    prob_dist = lik * x2
    print('p(y2).prior: {}'.format(ys[-1].prior().covariance()))
    print('p(x2,y2) mean: {}'.format(prob_dist.mean()))
    print('p(x2,y2) cov: {}'.format(prob_dist.covariance()))
    print('p(x2,y2) precision: {}'.format(torch.inverse(prob_dist.covariance())))

    # p(y2)
    # E[X2] = AE[X1] = 2*2=4
    # Var[X2] = A^2Var[X1] + Q^2 = 4*45 + 9 = 189
    # E[Y2] = CE[X2] + 0 = C*4 = 2
    # Var[Y2] = C^2Var[X2] + R^2 = 189/4 + 4 = 51.25
    p2 = prob_dist.marginalize_out(x2.left)
    print('p(y2) mean: {}'.format(p2.mean()))
    print('p(y2) cov: {}'.format(p2.covariance()))

    # p(Y1|X1) * p(X1|X0)
    # E[Y1|X0] = CAX0 = X0
    # Var[Y1|X0] = C^2Var[X1|X0] + R^2 = C^2Q^2 + R^2 = 9/4 + 4 = 6.25
    # E[X1|X0] = AX0 = 2X0
    # Var[X1|X0] = Q^2 = 9
    y1_given_x1 = ys[-2].likelihood()
    x1_given_x0 = xs[1].likelihood()

    x1_y1_given_x0 = y1_given_x1 * x1_given_x0
    print('p(x1,y1|x0) mean: {}'.format(x1_y1_given_x0.mean(value=2.)))
    print('p(x1,y1|x0) cov: {}'.format(x1_y1_given_x0.covariance()))

    # p(y1,x1|x0) * p(x0)
    x0_x1_y1 = x1_y1_given_x0 * xs[0].prior()
    print('p(x0,x1,y1) mean: {}'.format(x0_x1_y1.mean()))
    print('p(x0,x1,y1) cov: {}'.format(x0_x1_y1.covariance()))
    print('p(x0,x1,y1) precision: {}'.format(torch.inverse(x0_x1_y1.covariance())))
    cov = x0_x1_y1.covariance()

    # p(x0)p(x1|x0) = p(x1,x0)
    x1_x0 = xs[0].prior() * xs[1].likelihood()
    # p(x1,x0)p(y1|x1) = p(y1,x1,x0)
    another_x0_x1_y1 = y1_given_x1 * x1_x0
    print('another p(x0,x1,y1) mean: {}'.format(another_x0_x1_y1.mean()))
    print('another p(x0,x1,y1) cov: {}'.format(another_x0_x1_y1.covariance()))
    print('another p(x0,x1,y1) precision: {}'.format(torch.inverse(another_x0_x1_y1.covariance())))

    y1_given_x0 = x1_y1_given_x0.marginalize_out(x1_given_x0.left)
    print('p(y1|x0) mean: {}'.format(y1_given_x0.mean(value=2.)))
    print('p(y1|x0) cov: {}'.format(y1_given_x0.covariance()))

    # p(Y1|X1) * p(X1,X0) = p(Y1|X1) * p(X1|X0) * p(X0) = p(Y1,X1,X0)
    x1_x2_y1 = y1_given_x1 * x_dist
    print('p(x1,x2,y1) mean: {}'.format(x1_x2_y1.mean()))
    print('p(x1,x2,y1) cov: {}'.format(x1_x2_y1.covariance()))

    x1_x2 = x1_x2_y1.marginalize_out(y1_given_x1.left)
    print('p(x1,x2) mean {}'.format(x1_x2.mean()))
    print('p(x1,x2) covariance {}'.format(x1_x2.covariance()))

    # p(x1|y1)
    posterior = ys[-2].posterior()
    print('p(x1|y1=1.) mean: {}'.format(posterior.mean(value=2.)))
    print('p(x1|y1=1.) cov: {}'.format(posterior.covariance(value=2.)))

    joint = posterior * p1
    print('p(x1,y1) mean: {}'.format(joint.mean()))
    print('p(x1,y1) cov: {}'.format(joint.covariance()))
    print('p(x1,y1) precision: {}'.format(torch.inverse(joint.covariance())))


if __name__ == "__main__":
    test()
    ys = torch.rand(10, 1)
    A = torch.tensor(2.).reshape(1,1)
    Q = torch.tensor(3.).reshape(1,1)
    C = torch.tensor(0.5).reshape(1,1)
    R = torch.tensor(2.).reshape(1,1)
    mu_0 = torch.ones_like(ys[0])
    Q_0 = Q

    num_transitions = 2
    xt = GaussianRandomVariable(mu_0, Q_0, name="x")
    w = GaussianRandomVariable(0., Q)
    v = GaussianRandomVariable(0., R)
    p_xt_prev = xt.prior()
    xs = [xt]
    ys = []
    posterior_xt_prev_given_yt_prev = None
    for i in range(num_transitions):
        xt = LinearGaussian(A, xt, w, name="x")
        yt = LinearGaussian(C, xt, v, name="y")
        xs.append(xt)
        ys.append(yt)

        if posterior_xt_prev_given_yt_prev is not None:
            # p(x2|x1) * p(x1|y1)
            joint_xt_given_yt_prev = xt.likelihood() * posterior_xt_prev_given_yt_prev
            print('p(x1,x2|y1) mean: {}'.format(joint_xt_given_yt_prev.mean(value=2.)))
            print('p(x1,x2|y1) cov: {}'.format(joint_xt_given_yt_prev.covariance(value=2.)))
            print('p(x1,x2|y1) precision: {}'.format(torch.inverse(joint_xt_given_yt_prev.covariance(value=2.))))

            p_xt_given_yt_prev = joint_xt_given_yt_prev.marginalize_out(posterior_xt_prev_given_yt_prev.left[0])
            p_xt_given_yt_prev.is_posterior = True

            joint_xt_yt_given_yt_prev = yt.likelihood() * p_xt_given_yt_prev
            import pdb; pdb.set_trace()
            p_yt_given_yt_prev = joint_xt_yt_given_yt_prev.marginalize_out(p_xt_given_yt_prev.left[0])

            # print(' mmmm ean {}'.format(p_yt_given_yt_prev.mean(value=1.)))
            # print('covariance {}'.format(p_yt_given_yt_prev.covariance(value=1.)))
            joint_yt = p_yt_given_yt_prev * joint_yt

        else:
            # set up for the next iteration
            joint_xt_prev = xt.likelihood() * p_xt_prev
            p_xt_prev = joint_xt_prev.marginalize_out(xt.x)

            joint_yt_xt_prev = yt.likelihood() * p_xt_prev
            joint_yt = joint_yt_xt_prev.marginalize_out(p_xt_prev.left[0])

        posterior_xt_prev_given_yt_prev = yt.posterior()

    print("w: {}".format(joint_yt.mean()))
    print("p(y_1,...,y_n) covariance: {}".format(joint_yt.covariance()))


    # p(y1,y2) = p(y2|y1)p(y1)
    # p(y2|y1) = int p(y2|x2)p(x2|y1)dx2
    # p(x2|y1) = int p(x2|x1)p(x1|y1)dx1
    # p(y1) = int p(y1|x1)p(x1)dx1
    # p(x1) = int p(x1|x0)p(x0)dx0

    # train()
    # evaluate()
