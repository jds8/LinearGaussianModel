#!/usr/bin/env python3
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
        if self.is_y_given_y ^ other.is_y_given_y:
            return self.y_mul(other)

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
                left = self.left + other.left

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
                left = other.left + self.left

                y_precision = torch.inverse(self.covariance())
                x_precision = torch.inverse(other.covariance())
                a_mat = torch.tensor([x.get_coef_wrt(self.right) for x in self.left]).view(1,-1)

        mod_y_precision = torch.matmul(torch.matmul(a_mat, y_precision), a_mat.t())

        return self.create_distribution(x_precision, y_precision, mod_y_precision, a_mat, left, right, right_id)

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
        return torch.inverse(self.covariance(**kwargs))


class GaussianRandomVariable:
    x_ids = 0
    y_ids = 1
    def __init__(self, mu, sigma, observed=False, name=""):
        self.mu = torch.tensor(mu) if not isinstance(mu, torch.Tensor) else mu
        self.sigma = torch.tensor(sigma) if not isinstance(sigma, torch.Tensor) else sigma
        self.observed = observed
        if name == "x":
            self.name = 'x{}'.format(GaussianRandomVariable.x_ids)
            GaussianRandomVariable.x_ids += 1
        elif name == "y":
            self.name = 'y{}'.format(GaussianRandomVariable.y_ids)
            GaussianRandomVariable.y_ids += 1
        else:
            self.name = name + ' intermediate'

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
            print('WARNING: assuming independence between {} and {}'.format(self.name, b.name))
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
            return torch.tensor(self == var, dtype=torch.float32).reshape(1, -1)
        raise NotImplementedError

    def covariance(self, var):
        if self == var:
            return self.sigma**2
        try:
            return var.a * self.covariance(var.x) + self.covariance(var.b)
        except:
            return torch.tensor(0.).reshape(1, -1)

    def covariance_str(self, var):
        if self == var:
            return '{}'.format(self.sigma.item()**2)
        try:
            return '{} * ({}) + {}'.format(var.a.item(), self.covariance_str(var.x) ,self.covariance_str(var.b))
        except:
            return '{}'.format(torch.tensor(0.).reshape(1, -1).item())

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
        """"""
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

    def covariance(self, var):
        return self.a * self.x.covariance(var) + self.b.covariance(var)

    def covariance_str(self, var):
        return '{} * ({}) + {}'.format(self.a.item(), self.x.covariance_str(var), self.b.covariance_str(var))

def test_covariance(true_ys=None):
    num_transitions = len(true_ys)
    xt = GaussianRandomVariable(mu_0, Q_0, name="x")
    w = GaussianRandomVariable(0., Q, "w")
    v = GaussianRandomVariable(0., R, "v")
    p_xt_prev = xt.prior()
    xs = [xt]
    ys = []
    posterior_xt_prev_given_yt_prev = None
    for i in range(num_transitions):
        yt = LinearGaussian(C, xt, v, name="y")
        xt = LinearGaussian(A, xt, w, name="x")
        xs.append(xt)
        ys.append(yt)

    # print('x1 estimate: {}={}'.format(ys[0].x.covariance_str(ys[0].x),ys[0].x.covariance(ys[0].x)))
    # print('y1 estimate: {}={}'.format(ys[0].covariance_str(ys[0]),ys[0].covariance(ys[0])))
    # print('x2 estimate: {}={}'.format(ys[1].x.covariance_str(ys[1].x),ys[1].x.covariance(ys[1].x)))
    # print('y2 estimate: {}={}'.format(ys[1].covariance_str(ys[1]),ys[1].covariance(ys[1])))
    # print('x3 estimate: {}={}'.format(ys[2].x.covariance_str(ys[2].x),ys[2].x.covariance(ys[2].x)))
    # print('y3 estimate: {}={}'.format(ys[2].covariance_str(ys[2]), ys[2].covariance(ys[2])))
    # x1_var = Q**2
    # y1_var = C**2*x1_var + R**2
    # x2_var = A**2*x1_var + Q**2
    # x1_w_covar = 0.
    # y2_var = C**2*x2_var + R**2
    # x2_w_covar = A*x1_w_covar + Q**2
    # x3_var = A**2*x2_var + 2*A*x2_w_covar + Q**2
    # y3_var = C**2*x3_var + R**2
    # print('x1 var: {}'.format(x1_var))
    # print('y1 var: {}'.format(y1_var))
    # print('x2 var: {}'.format(x2_var))
    # print('y2 var: {}'.format(y2_var))
    # print('x3 var: {}'.format(x3_var))
    # print('y3 var: {}'.format(y3_var))

    # cov_y1_y2 = C**2*A*Q**2 + R**2
    # print('cov(y1, y2) estimate: {}={}'.format(ys[0].covariance_str(ys[1]), ys[0].covariance(ys[1])))
    # print('cov(y1, y2) actual: {}'.format(cov_y1_y2))
    # cov_y2_y3 = C**2*(A**3*Q**2+A*Q**2+Q**2) + R**2
    # print('cov(y2, y3) estimate: {}={}'.format(ys[1].covariance_str(ys[2]), ys[1].covariance(ys[2])))
    # print('cov(y2, y3) actual: {}'.format(cov_y2_y3))

    cov = compute_covariance(ys)
    print('cov {}'.format(cov))
    print('cov cond number {}'.format(torch.linalg.cond(cov)))
    d = dist.MultivariateNormal(torch.tensor([y.mu for y in ys]), cov)
    print('mean: {}'.format(d.mean))
    print('sample: {}'.format(d.sample()))
    print('true ys: {}'.format(true_ys))
    print('log_prob: {}'.format(d.log_prob(true_ys.reshape(d.mean.shape))))

def test():
    num_transitions = 2
    xt = GaussianRandomVariable(mu_0, Q_0, name="x")
    w = GaussianRandomVariable(0., Q, "w")
    v = GaussianRandomVariable(0., R, "v")
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

            p_xt_given_yt_prev = joint_xt_given_yt_prev.marginalize_out(posterior_xt_prev_given_yt_prev.left[0])
            p_xt_given_yt_prev.is_posterior = True

            joint_xt_yt_given_yt_prev = yt.likelihood() * p_xt_given_yt_prev
            p_yt_given_yt_prev = joint_xt_yt_given_yt_prev.marginalize_out(p_xt_given_yt_prev.left[0])

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


    # p(y1, y2, y3) = p(y3|y2,y1)*p(y2,y1)
    # p(yn|y1,...,y_{n-1}) is the normalization constant
    # p(xn|y1,...,yn) = int p(x{n-1}|y1,...,y{n-1})p(xn|x_{n-1})p(yn|xn) / p(yn|y1,...,y_{n-1}) x_{n-1}
    # p(x2|y1,y2) = p(x2,y2|y1)/p(y2)
    # p(x1|y1)
    # p(y1,y2) = p(y2|y1)p(y1)
    # p(y2|y1) = int p(y2|x2)p(x2|y1)dx2
    # p(x2|y1) = int p(x2|x1)p(x1|y1)dx1
    # p(y1) = int p(y1|x1)p(x1)dx1
    # p(x1) = int p(x1|x0)p(x0)dx0

    # train()
    # evaluate()

def compute_covariance(ys):
    cov = torch.zeros(len(ys), len(ys))
    for r in range(len(ys)):
        for c in range(len(ys)):
            cov[r, c] = ys[r].covariance(ys[c])
    return cov + torch.diag(torch.tensor([1e-6 for _ in range(cov.shape[0])]))

def analytical_score(true_ys, A, Q, C, R, mu0, Q0):
    num_transitions = len(true_ys)
    xt = GaussianRandomVariable(mu_0, Q_0, name="x")
    w = GaussianRandomVariable(0., Q, "w")
    v = GaussianRandomVariable(0., R, "v")
    p_xt_prev = xt.prior()
    xs = [xt]
    ys = []
    posterior_xt_prev_given_yt_prev = None
    for i in range(num_transitions):
        yt = LinearGaussian(C, xt, v, name="y")
        xt = LinearGaussian(A, xt, w, name="x")
        xs.append(xt)
        ys.append(yt)
    cov = compute_covariance(ys)
    d = dist.MultivariateNormal(torch.tensor([y.mu for y in ys]), cov)
    return d.log_prob(true_ys.reshape(d.mean.shape))
