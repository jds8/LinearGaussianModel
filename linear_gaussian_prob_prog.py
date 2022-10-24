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
from generative_model import generate_trajectory, \
    single_gen_A, single_gen_Q,\
    single_gen_C, single_gen_R,\
    single_gen_mu_0, single_gen_Q_0,\
    gen_A, gen_Q,\
    gen_C, gen_R,\
    gen_mu_0, gen_Q_0
from math_utils import band_matrix, kalman_filter
from dimension_table import create_dimension_table


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
        self.is_conditional = isinstance(self.dist, TorchDistributionInfo) and self.right is not None

    def __mul__(self, other):
        # if self.is_y_given_y ^ other.is_y_given_y:
        #     return self.y_mul(other)

        if self.is_posterior ^ other.is_posterior:
            return self.mul_posterior(other)

        if (self.right in other.left) ^ (other.right in self.left):  # ensures form like p(y|x)p(x)
            if self.right in other.left:
                # in the special case that len(left) == 1 and the RHS is dependent on the LHS,
                # then we have a (single-variate) posterior
                right = other.right
                left = other.left + self.left
                y_cv = self.covariance()
                x_cv = other.covariance()
                right_id = other.left.index(self.right)
                if self.left[0].x.mu.nelement() > 1: # if the dimension is greater than 1
                    a_mat_not = torch.cat([x.get_coef_wrt(self.right) for x in self.left], 0)
                    a_mat = torch.cat([x.get_coef_wrt(self.right) for x in self.left], 1)

                else: # if the dimension is equal to 1
                    a_mat = torch.tensor([x.get_coef_wrt(self.right) for x in self.left]).view(1, -1)
            else:  # other.right in self.left in this case
                right = self.right
                left = self.left + other.left
                y_cv = other.covariance()
                x_cv = self.covariance()
                right_id = self.left.index(other.right)
                if other.left[0].x.mu.nelement() > 1: # if the dimension is greater than 1
                    a_mat = torch.cat([x.get_coef_wrt(other.right) for x in other.left], 1)
                else: # if the dimension is equal to 1
                    a_mat = torch.tensor([x.get_coef_wrt(other.right) for x in other.left]).view(1, -1)
                # a_mat = torch.tensor([x.get_coef_wrt(other.right) for x in other.left])
                # a_mat = a_mat.view(a_mat.shape[1], -1)
                # a_mat = other.left[0].get_coef_wrt(other.right)
                # a_mat = a_mat.view(a_mat.shape[1], -1)

            y_precision = torch.inverse(y_cv)
            x_precision = torch.inverse(x_cv)

            # mod_y_precision = torch.zeros_like(x_precision)
            # mod_y_precision[right_id, right_id] = torch.matmul(torch.matmul(a_mat, y_precision), a_mat.t())
            a_mat = a_mat.reshape(-1, y_precision.shape[0])

            mod_y_precision = torch.matmul(torch.matmul(a_mat, y_precision), a_mat.t())

            return self.create_distribution(x_precision, y_precision, mod_y_precision, a_mat, left, right, right_id)
        else: # these distributions are independent, so multiply them
            covariance = torch.block_diag(self.covariance(), other.covariance())
            left = self.left + other.left
            if self.right is None:
                right = other.right
            elif other.right is None:
                right = self.right
            else:
                assert self.right == other.right
                right = self.right

            def mul(value):
                mu1 = self.mean(value=value).reshape(-1, 1)
                mu2 = other.mean(value=value).reshape(-1, 1)
                mean = torch.cat([mu1, mu2]).squeeze()
                return dist.MultivariateNormal(mean, covariance)

            if right is not None:  # if we are conditioning, then condition
                return GaussianDistribution(dist=TorchDistributionInfo(mul), left=left, right=right)

            else:  # compute everything directly
                return GaussianDistribution(dist=mul(value=None), left=left, right=right)

    def mul_posterior(self, other):
        assert (self.right in other.left) ^ (other.right in self.left)  # ensures forms like p(x|y)p(y)
        print("WARNING: Calculating product of densities while assuming that precisely one term is a posterior and that it conditions on a variable one step away in the graphical model.")

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

    def __pow__(self, n):
        if n < 1:
            raise NotImplementedError
        if n == 1:
            return self
        return (self**(n-1)) * self
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

            # get dimensionality of var
            d = var.mu.nelement()

            # index into mu and covariance depending on the dimensionality of the variables
            start_idx = 0
            for v in left:
                if v == var:
                    break
                start_idx += v.mu.nelement()

            # remove var from self.left
            idx = left.index(var)
            del left[idx]

            # remove var from mean
            mu = mean_fun(value=value).clone()
            first_part = mu[:start_idx]
            second_part = mu[start_idx+d:] if start_idx+d < len(mu) else torch.tensor([])
            mu = torch.cat([first_part, second_part])

            # remove var from covariance
            covariance = covariance_fun().clone()
            mask = torch.tensor([i for i in range(len(covariance)) if i < start_idx or i >= start_idx + d])
            covariance = torch.index_select(covariance, 0, mask)
            covariance = torch.index_select(covariance, 1, mask)

            # create normal dist
            if mu.nelement() == 1:
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

    def log_prob(self, xs, **kwargs):
        return self.get_dist(**kwargs).log_prob(xs)

    def mean(self, **kwargs):
        return self.get_dist(**kwargs).mean

    def covariance(self, **kwargs):
        """
        Computes covariance of this distribution
        (Note that the variance does *not* depend on parameter values in linear gaussian setting)
        """
        # don't need a value in order to call covariance
        dummy = None if self.right is None else self.right.mu
        prob_dist = self.get_dist(value=dummy)
        if isinstance(prob_dist, dist.MultivariateNormal):
            return prob_dist.covariance_matrix
        return prob_dist.stddev**2

    def precision(self, **kwargs):
        return torch.inverse(self.covariance(**kwargs))

    @staticmethod
    def _condition_helper(dd, x_inds, z_inds, all_vars, z_values):
        # find indices which correspond to x_inds ( dimensionality might be > 1 )
        x_dims = []
        if x_inds.nelement() > 0:
            x_dims = [len(var.mu) for var in np.array(all_vars)[x_inds.numpy()]]
        z_dims = [len(var.mu) for var in np.array(all_vars)[z_inds.numpy()]]

        start_inds = [0]
        for var in all_vars[0:-1]:
            start_inds += [start_inds[-1] + var.mu.nelement()]

        x_dim_inds = np.array([])
        for ind, dim in zip(x_inds, x_dims):
            start_ind = start_inds[ind]
            x_dim_inds = np.concatenate((x_dim_inds, np.arange(start_ind, start_ind+dim)))
        x_dim_inds = torch.tensor(x_dim_inds, dtype=torch.int32)

        z_dim_inds = np.array([])
        for ind, dim in zip(z_inds, z_dims):
            start_ind = start_inds[ind]
            z_dim_inds = np.concatenate((z_dim_inds, np.arange(start_ind, start_ind+dim)))
        z_dim_inds = torch.tensor(z_dim_inds, dtype=torch.int32)

        mu_x = torch.index_select(dd.mean, 0, x_dim_inds)
        mu_z = torch.index_select(dd.mean, 0, z_dim_inds)
        sigma_xx = torch.index_select(torch.index_select(dd.covariance_matrix, 0, x_dim_inds), 1, x_dim_inds)
        sigma_xz = torch.index_select(torch.index_select(dd.covariance_matrix, 0, x_dim_inds), 1, z_dim_inds)
        sigma_zx = torch.index_select(torch.index_select(dd.covariance_matrix, 0, z_dim_inds), 1, x_dim_inds)
        sigma_zz = torch.index_select(torch.index_select(dd.covariance_matrix, 0, z_dim_inds), 1, z_dim_inds)
        product = torch.mm(sigma_xz, torch.inverse(sigma_zz))

        mu_x_given_z = mu_x + torch.mm(product, (z_values - mu_z).reshape(product.shape[1], -1).to(product.dtype)).reshape(mu_x.shape)
        sigma_x_given_z = sigma_xx - torch.mm(sigma_xz, torch.mm(torch.inverse(sigma_zz), sigma_zx))

        # print('sigma_xx {}'.format(sigma_xx))
        # print('sigma_xz {}'.format(sigma_xz))
        # print('sigma_zz {}'.format(sigma_zz))
        # print('other {}'.format(torch.mm(torch.mm(sigma_xz, torch.inverse(sigma_zz)), sigma_zx)))

        return dist.MultivariateNormal(mu_x_given_z.reshape(-1), sigma_x_given_z)

    def condition(self, z_rvs):
        z_inds = [self.left.index(var.r_var) for var in z_rvs]
        all_vars = [var for var in self.left]
        z_values = torch.cat([var.value for var in z_rvs])
        x_inds = list(set(range(len(self.left))) - set(z_inds))
        z_inds = torch.tensor(z_inds)
        x_inds = torch.tensor(x_inds)
        if self.right is None:
            dd = self.get_dist()
            prob_dist = self._condition_helper(dd, x_inds, z_inds, all_vars, z_values)
        else:
            dd_dist = self.dist
            def inner_fun(value):
                dd = dd_dist.evaluate(value)
                return self._condition_helper(dd, x_inds, z_inds, all_vars, z_values)
            prob_dist = TorchDistributionInfo(inner_fun)
        return GaussianDistribution(prob_dist, left=self.left, right=self.right)


class RandomVariable:
    def __init__(self, r_var, value):
        self.r_var = r_var
        self.value = value


class JointVariables:
    def __init__(self, rvs, A, C):
        self.rvs = rvs
        self.A = A
        self.C = C
        self.dist = self._compute_joint_dist()

    def _compute_joint_dist(self):
        mu = torch.cat([rv.mu for rv in self.rvs], dim=0).squeeze()
        if mu.shape == torch.Size([]):
            mu = mu.reshape(1)
        cov = torch.block_diag(*[rv.sigma for rv in self.rvs])
        i = 0
        j = 0
        for i_rv in range(len(self.rvs)):
            x = self.rvs[i_rv]
            i_dim = x.mu.nelement()
            j = i
            for j_rv in range(i_rv, len(self.rvs)):
                y = self.rvs[j_rv]
                j_dim = y.mu.nelement()
                sigma = x.covariance(y)
                cov_block = cov[i:i+i_dim, j:j+j_dim]
                cov[i:i+i_dim, j:j+j_dim] = sigma.reshape(cov_block.shape)
                cov[j:j+j_dim, i:i+i_dim] = sigma.reshape(cov_block.t().shape)

                j += j_dim
            i += i_dim
        return GaussianDistribution(dist.MultivariateNormal(mu, cov), left=self.rvs, right=None)

    def condition(self, z_rvs):
        z_inds = [self.rvs.index(var) for var in z_rvs]
        x_inds = list(set(range(len(self.rvs))) - set(z_inds))
        z_inds = torch.tensor(z_inds)
        x_inds = torch.tensor(x_inds)
        left = [rv for rv in self.rvs if rv not in z_rvs]
        def condition_values(value):
            return GaussianDistribution._condition_helper(self.dist.get_dist(), x_inds, z_inds, self.rvs, value)
        return GaussianDistribution(TorchDistributionInfo(condition_values), left=left, right=left[0])


class GaussianRandomVariable:
    x_ids = 0
    y_ids = 0
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
            return GaussianRandomVariable(self.mu * a, torch.sqrt(self.sigma**2 * a * a), observed=self.observed)
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
            # print('WARNING: assuming independence between {} and {}'.format(self.name, b.name))
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
            return torch.tensor(self == var, dtype=torch.float32) * torch.eye(var.sigma.shape[0])
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
            return '{} * ({}) + {}'.format(var.a.item(), self.covariance_str(var.x), self.covariance_str(var.b))
        except:
            return '{}'.format(torch.tensor(0.).reshape(1, -1).item())

class LinearGaussian(GaussianRandomVariable):
    def __init__(self, a, x: GaussianRandomVariable, b: GaussianRandomVariable, name):
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
            return torch.eye(var.sigma.shape[0])
        if var == self.x:
            return self.a
        return torch.mm(self.a, self.x.get_coef_wrt(var))

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

def compute_joint():
    w = GaussianRandomVariable(mu=0., sigma=single_gen_Q, name="w")
    v = GaussianRandomVariable(mu=0., sigma=single_gen_R, name="v")
    xt = GaussianRandomVariable(mu=single_gen_mu_0, sigma=single_gen_Q_0, name="x")
    xs = [xt]
    ys = []
    posterior_xt_prev_given_yt_prev = None
    num_transitions = 2
    for i in range(num_transitions):
        yt = LinearGaussian(a=single_gen_C, x=xt, b=v, name="y")
        xt = LinearGaussian(a=single_gen_A, x=xt, b=w, name="x")
        xs.append(xt)
        ys.append(yt)

    joint = xs[0].prior() * ys[0].likelihood()
    for i in range(1, num_transitions):
        joint *= xs[i].likelihood()
        joint *= ys[i].likelihood()


class MultiGaussianRandomVariable:
    x_ids = 0
    y_ids = 0
    def __init__(self, mu, sigma, observed=False, name=""):
        """ Mu represents the mean and Sigma the covariance matrix of a Gaussian distribution """
        self.sigma = torch.tensor(sigma) if not isinstance(sigma, torch.Tensor) else sigma
        self.mu = torch.tensor(mu) if not isinstance(mu, torch.Tensor) else mu
        self.observed = observed
        if name == "x":
            self.name = 'x{}'.format(MultiGaussianRandomVariable.x_ids)
            MultiGaussianRandomVariable.x_ids += 1
        elif name == "y":
            self.name = 'y{}'.format(MultiGaussianRandomVariable.y_ids)
            MultiGaussianRandomVariable.y_ids += 1
        else:
            self.name = name + ' intermediate'

    @classmethod
    def reset_ids(cls):
        cls.x_ids = 0
        cls.y_ids = 0

    def __rmul__(self, a):
        if isinstance(a, torch.Tensor):
            mgrv = MultiGaussianRandomVariable(torch.mm(a, self.mu.reshape(a.shape[1], -1)).squeeze(), torch.mm(torch.mm(a, self.sigma), a.t()), observed=self.observed, name=self.name)
            return mgrv
        raise NotImplementedError

    def __add__(self, b):
        if isinstance(b, torch.Tensor):
            return MultiGaussianRandomVariable(self.mu + b, self.sigma)
        if isinstance(b, MultiGaussianRandomVariable):
            if b.observed:
                return self + b.mu
            # assuming independent summands
            # print('WARNING: assuming independence between {} and {}'.format(self.name, b.name))
            return MultiGaussianRandomVariable(self.mu + b.mu, self.sigma + b.sigma, name=self.name)
        raise NotImplementedError

    def __radd__(self, b):
        return self + b

    def observe(self, value: torch.Tensor):
        dim = value.nelement()
        mu = value
        sigma = torch.zeros(dim, dim)
        observed = True
        return MultiGaussianRandomVariable(mu, sigma, observed)

    def prior(self):
        return GaussianDistribution(dist=dist.MultivariateNormal(self.mu.reshape(self.sigma.shape[0]), self.sigma), left=[self])

    def conditional_variance(self, var=None):
        if var is None:
            return self.sigma
        raise NotImplementedError

    def is_dependent_on(self, var):
        return False

    def get_coef_wrt(self, var):
        if self == var:
            return torch.tensor(self == var, dtype=torch.float32) * torch.eye(var.sigma.shape[0])
        raise NotImplementedError

    def covariance(self, var):
        if var == None:
            var = self
        if self == var:
            return self.sigma
        try:
            left = torch.mm(var.a, self.covariance(var.x))
            return left + self.covariance(var.b)
        except:
            return torch.zeros(var.mu.shape[0], self.mu.shape[0])

    def covariance_str(self, var):
        if self == var:
            return '{}'.format(self.sigma.item())
        try:
            return '{} * ({}) + {}'.format(var.a.item(), self.covariance_str(var.x), self.covariance_str(var.b))
        except:
            return '{}'.format(torch.tensor(0.).reshape(1, -1).item())

    def marginal(self):
        return self.prior()

    def copy(self):
        return MultiGaussianRandomVariable(self.mu, self.sigma, self.observed)


class MultiLinearGaussian(MultiGaussianRandomVariable):
    def __init__(self, a, x: MultiGaussianRandomVariable, b: MultiGaussianRandomVariable, name):
        ax = a * x
        var = ax + b
        super(MultiLinearGaussian, self).__init__(var.mu, var.sigma, name=name)
        self.a = a
        self.x = x
        self.b = b

    def is_dependent_on(self, var: MultiGaussianRandomVariable):
        if var == self.x or var == self:
            return True
        return self.x.is_dependent_on(var)

    def get_coef_wrt(self, var: MultiGaussianRandomVariable):
        if var == self:
            return torch.eye(self.var.sigma.shape[0])
        if var == self.x:
            return self.a
        return torch.mm(self.a, self.x.get_coef_wrt(var))

    def likelihood(self):
        """"""
        a = self.a
        x = self.x
        b = self.b
        def lik(value):
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value).reshape(b.mu.shape)
            var = a * x.observe(value) + b
            return dist.MultivariateNormal(var.mu.reshape(var.sigma.shape[1]), var.sigma)
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
        sigma_term = torch.sqrt(sigma_x_given_y).reshape(1, 1) if sigma_x_given_y.nelement() == 1 else sigma_x_given_y

        mu_x_given_y = sigma_x_given_y * (a * inv_sigma_y * (1. - b) + inv_sigma_x * x.mu)

        def post(value):
            mu_x_given_y = sigma_x_given_y * (a * inv_sigma_y * (value - b) + inv_sigma_x * x.mu)
            prob_dist = dist.MultivariateNormal(mu_x_given_y, sigma_term)
            return prob_dist
        return GaussianDistribution(TorchDistributionInfo(post), left=[x], right=self, is_posterior=True)

    def conditional_variance(self, var):
        if self.x == var:
            return self.b.sigma**2
        if self.is_dependent_on(var) or var is None:
            return torch.mm(torch.mm(self.a.t(), self.x.conditional_variance(var)), self.a) + self.b.sigma
        else:
            raise NotImplementedError

    def covariance(self, var):
        if var == None:
            var = self
        if self == var:
            return self.sigma
        b_cov = self.b.covariance(var)
        ax = torch.mm(self.a, self.x.covariance(var).reshape(self.a.shape[1], -1)).reshape(b_cov.shape)
        return ax + b_cov

    def covariance_str(self, var):
        return '{} * ({}) + {}'.format(self.a.item(), self.x.covariance_str(var), self.b.covariance_str(var))

    def marginal(self):
        return (self.likelihood() * self.x.marginal()).marginalize_out(self.x)


class VecLinearGaussian:
    def __init__(self, a, x, b):
        self.a = a
        self.x = x
        self.b = b

    def posterior_vec(self):
        """
        The posterior function for this variable P(X|Y).
        Note that this function returns a distribution object which
        can be evaluated at a particular value of Y=y but can also
        be used to compute joint distributions.
        """
        d, _ = self.a.shape
        nd, _ = self.x.covariance().shape

        a = band_matrix(self.a, int(nd/d))
        x = self.x
        b = self.b

        inv_sigma_x = torch.inverse(x.covariance())
        inv_sigma_y = torch.inverse(b.covariance())

        inv_sigma_x_given_y = inv_sigma_x + torch.mm(torch.mm(a.t(), inv_sigma_y), a)
        sigma_x_given_y = torch.inverse(inv_sigma_x_given_y)

        def post(value):
            mu_x_given_y = torch.mm(sigma_x_given_y, torch.mm(torch.mm(a.t(), inv_sigma_y), (value - b.mean()).reshape(-1, 1)) + torch.mm(inv_sigma_x, x.mean().reshape(-1, 1)))
            if sigma_x_given_y.nelement() > 1:
                return dist.MultivariateNormal(mu_x_given_y.squeeze(-1), sigma_x_given_y)
            return dist.Normal(mu_x_given_y.squeeze(-1), torch.sqrt(sigma_x_given_y))
        return GaussianDistribution(TorchDistributionInfo(post), left=[x], right=self, is_posterior=True)


class LinearGaussianVariables:
    def __init__(self, xs, ys, w, v, table):
        self.xs = xs
        self.ys = ys
        self.w = w
        self.v = v
        self.table = table

        # the following lists are to be populated in
        # _compute_x_given_ys from compute_joint_ys
        self.joints = [self.ys[0].marginal()]
        self.ys_liks = [self.ys[0].likelihood()]
        # self._compute_joint_ys()

    def _compute_prev_ys_given_x(self, ind):
        posterior = self.xs[ind].posterior()
        ys_given_x = self.ys_liks[ind-1]
        return (ys_given_x * posterior).marginalize_out(posterior.left)

    def _compute_x_given_ys(self, ind):
        if ind == 0:
            return self.ys[0].posterior()
        y_lik = self.ys[ind].likelihood()
        prev_ys_lik = self._compute_prev_ys_given_x(ind)
        ys_lik = y_lik * prev_ys_lik
        self.ys_liks.append(ys_lik)
        x_marg = self.xs[ind].marginal()
        x_y_joint = ys_lik * x_marg
        return x_y_joint.condition(ys_lik.left)

    def _compute_next_x_given_ys(self, ind):
        lik = self.xs[ind].likelihood()
        x_given_ys = self._compute_x_given_ys(ind-1)
        return (lik * x_given_ys).marginalize_out(lik.right)

    def _compute_joint_ys(self):
        for i, y in enumerate(self.ys[1:]):
            lik = y.likelihood()
            next_x_given_ys = self._compute_next_x_given_ys(i+1)
            conditional = (lik * next_x_given_ys).marginalize_out(lik.right)
            # TODO: this product is not yet defined
            self.joints.append(conditional * self.joints[i])


def get_linear_gaussian_variables(dim, num_obs, table):
    MultiGaussianRandomVariable.reset_ids()
    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    mu_0 = table[dim]['mu_0']
    Q_0 = table[dim]['Q_0']

    w = MultiGaussianRandomVariable(mu=torch.zeros(dim), sigma=Q, name="w")
    v = MultiGaussianRandomVariable(mu=torch.zeros(1), sigma=R, name="v")
    xt = MultiGaussianRandomVariable(mu=mu_0, sigma=Q_0, name="x")
    xs = [xt]
    ys = []
    for i in range(num_obs):
        yt = MultiLinearGaussian(a=C, x=xt, b=v.copy(), name="y")
        ys.append(yt)
        xt = MultiLinearGaussian(a=A, x=xt, b=w.copy(), name="x")
        xs.append(xt)
    # remove the last transition as it's not part of the model
    del xs[-1]

    return LinearGaussianVariables(xs=xs, ys=ys, w=w, v=v, table=table)

def compute_block_posterior(dim, num_observations, table):
    lgv = get_linear_gaussian_variables(dim=dim, num_obs=num_observations, table=table)
    C = lgv.table[dim]['C']

    prior = lgv.xs[0].prior()
    for i in range(1, num_observations):
        lik = lgv.xs[i].likelihood()
        prior *= lik

    noise = lgv.v.prior()**num_observations

    # find full likelihood
    ys = VecLinearGaussian(a=C.t(), x=prior, b=noise)

    # compute posterior
    posterior = ys.posterior_vec()

def test_joint_vars():
    dim = 2
    num_obs = 2

    table = create_dimension_table(torch.tensor([dim]), random=False)
    lgv = get_linear_gaussian_variables(dim=dim, num_obs=num_obs, table=table)
    xs = lgv.xs
    ys = lgv.ys

    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    mu_0 = table[dim]['mu_0']
    Q_0 = table[dim]['Q_0']

    jvs = JointVariables([xs[0], xs[1], ys[0], ys[1]], A, C)
    print(jvs.dist.covariance())
    rhs_rvs = [xs[0], ys[0]]
    conditional = jvs.condition(rhs_rvs)

    # evaluate it
    ys = generate_trajectory(num_obs, A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0)[0]
    env = LinearGaussianEnv(A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0, ys=ys, sample=False)

def test_graphical_model():
    dim = 1
    num_obs = 10

    table = create_dimension_table(torch.tensor([dim]), random=True)
    lgv = get_linear_gaussian_variables(dim=dim, num_obs=num_obs, table=table)
    xs = lgv.xs
    ys = lgv.ys

    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    mu_0 = table[dim]['mu_0']
    Q_0 = table[dim]['Q_0']

    obs = generate_trajectory(num_obs, A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0)[0]

    # jvs = JointVariables([xs[0], xs[1], ys[0]], A, C)
    # print(jvs.dist.covariance())
    # rhs_rvs = [xs[0], ys[0]]
    # conditional_1 = jvs.condition(rhs_rvs)
    # print(conditional_1.covariance())

    # jvs3 = JointVariables([xs[0], xs[1], ys[1]], A, C)
    # rhs_rvs3 = [xs[0], ys[1]]
    # conditional_3 = jvs3.condition(rhs_rvs3)
    # print(conditional_3.covariance())

    # jvs2 = JointVariables([xs[0], xs[1]], A, C)
    # rhs_rvs2 = [xs[0]]
    # conditional_2 = jvs2.condition(rhs_rvs2)
    # print(conditional_2.covariance())

    # # p(x2|x1)
    # jvs = JointVariables([xs[1], xs[2]], A, C)
    # rhs_rvs = [xs[1]]
    # conditional_1 = jvs.condition(rhs_rvs)
    # print(conditional_1.covariance())

    # # p(x2|y2)
    # jvs = JointVariables([ys[2], xs[2]], A, C)
    # rhs_rvs = [ys[2]]
    # conditional_1 = jvs.condition(rhs_rvs)
    # print(conditional_1.covariance())

    # # p(x2|x1, y2)
    # jvs = JointVariables([xs[1], ys[2], xs[2]], A, C)
    # rhs_rvs = [ys[2], xs[1]]
    # conditional_1 = jvs.condition(rhs_rvs)
    # print(conditional_1.covariance())

    # # p(x2|x1, y0, y1, y2)
    # jvs = JointVariables([xs[1], ys[0], ys[1], ys[2], xs[2]], A, C)
    # rhs_rvs = [ys[0], ys[1], ys[2], xs[1]]
    # conditional_1 = jvs.condition(rhs_rvs)
    # print(conditional_1.covariance())

    # # p(x2|y1, y2)
    # jvs = JointVariables([ys[1], ys[2], xs[2]], A, C)
    # rhs_rvs = [ys[1], ys[2]]
    # conditional_1 = jvs.condition(rhs_rvs)
    # print(conditional_1.covariance())

    # # p(x2|y0, y2)
    # jvs = JointVariables([ys[0], ys[2], xs[2]], A, C)
    # rhs_rvs = [ys[0], ys[2]]
    # conditional_1 = jvs.condition(rhs_rvs)
    # print(conditional_1.covariance())


    # # p(x_{num_obs}|y{0:num_obs})
    jvs = JointVariables(ys[0:num_obs] + [xs[-1]], A, C)
    rhs_rvs = ys[0:num_obs]
    conditional_1 = jvs.condition(rhs_rvs)
    print(conditional_1.mean(value=obs))
    print(conditional_1.covariance(a=obs))

    # prior = dist.Normal(0, torch.sqrt(Q))
    # state_transition_variance = Q
    # likelihood_var = R
    # pred = kalman_filter(obs, prior, state_transition_variance, likelihood_var, A, C)
    # print(pred.mean)
    # print(pred.variance)

    # # p(x1|y0, y1, y2)
    # jvs = JointVariables([ys[0], ys[1], ys[2], xs[1]], A, C)
    # rhs_rvs = [ys[0], ys[1], ys[2]]
    # conditional_1 = jvs.condition(rhs_rvs)
    # print(conditional_1.covariance())

    # # p(x0|y0, y1, y2)
    # jvs = JointVariables([ys[0], ys[1], ys[2], xs[0]], A, C)
    # rhs_rvs = [ys[0], ys[1], ys[2]]
    # conditional_1 = jvs.condition(rhs_rvs)
    # print(conditional_1.covariance())

    # p(x0, x1, x2|y0, y1, y2)
    # jvs = JointVariables([xs[0], xs[1], xs[2], ys[0], ys[1], ys[2]], A, C)
    # rhs_rvs = [ys[0], ys[1], ys[2]]
    # conditional_1 = jvs.condition(rhs_rvs)
    # print(conditional_1.mean(value=torch.ones(len(rhs_rvs))))
    # print(conditional_1.covariance(value=torch.ones(len(rhs_rvs))))

if __name__ == "__main__":
    test_graphical_model()
