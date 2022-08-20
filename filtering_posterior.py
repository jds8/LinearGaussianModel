#!/usr/bin/env python3
import numpy as np
import torch
import torch.distributions as dist
from copy import deepcopy
from generative_model import generate_trajectory
from math_utils import logvarexp, importance_sampled_confidence_interval, log_effective_sample_size, log_max_weight_proportion, log_mean
from plot_utils import legend_without_duplicate_labels
from linear_gaussian_prob_prog import \
    MultiGaussianRandomVariable, GaussianRandomVariable, MultiLinearGaussian, \
    LinearGaussian, VecLinearGaussian, get_linear_gaussian_variables, JointVariables, \
    RandomVariable
from dimension_table import create_dimension_table


class FilteringPosterior:
    def __init__(self, numerator, left):
        self.numerator = numerator
        self.left = left

    def condition(self, y_values, x_value=None):
        if self.numerator.right is not None:
            joint = self.numerator * self.numerator.right.marginal()
        else:
            joint = self.numerator

        # names = [y.name for y in self.numerator.left if 'y' in y.name]
        # sorted_idx = np.argsort(names)
        # r_vars = np.array([y for y in self.numerator.left if 'y' in y.name])
        # r_vars = r_vars[sorted_idx]
        # r_vars = list(r_vars) + [self.numerator.right] if self.numerator.right is not None else list(r_vars)

        if x_value is not None:
            values = torch.cat([y_values, x_value])
            # values = torch.cat([y_values.reshape(x_value.shape[0], -1), x_value.reshape(-1, 1)], dim=1)
        else:
            values = y_values
        values = values.squeeze().reshape(-1)

        rvs = []
        # for r_var, val in zip(r_vars, values):
        current_idx = 0
        for r_var in self.left:
            dim = r_var.mu.nelement()
            rv = RandomVariable(r_var=r_var, value=values[current_idx:current_idx+dim])
            rvs.append(rv)
            current_idx += dim

        # # the next part is to ensure that the random variables
        # # in joint correspond to the y_values and x_values that
        # # are passed for conditioning
        # names = [y.name for y in self.numerator.left if 'y' in y.name]
        # r_vars = [y for y in self.numerator.left if 'y' in y.name]
        # sorted_idx = np.argsort(names)
        # vals = y_vals[sorted_idx]
        # if x_value is not None:
        #     # get index of x
        #     x_idx = joint.left.index(self.numerator.right)
        #     if x_idx == 0:
        #         vals = torch.cat([x_value, vals], dim=1)
        #     elif x_idx == len(joint.left)-1:
        #         vals = torch.cat([vals, x_value], dim=1)
        #     else:
        #         vals = torch.cat([vals[0:x_idx], x_value, vals[x_idx:]], dim=1)

        # return joint.condition(r_vars, vals)
        return joint.condition(rvs)

def compute_filtering_data_structures(dim, num_obs):
    lgv = get_linear_gaussian_variables(dim=dim, num_obs=num_obs)
    xs = lgv.xs
    ys = lgv.ys

    # compute denominator
    p_y_next_given_x = [None] * (num_obs-1)

    p_y_T_x_T_given_x_T_minus_1 = ys[-1].likelihood() * xs[-1].likelihood()
    p_y_next_given_x[-1] = p_y_T_x_T_given_x_T_minus_1.marginalize_out(xs[-1])

    current_index = -2

    for i in range(len(p_y_next_given_x)-2, -1, -1):
        x_dist = xs[current_index].likelihood()
        lik = ys[current_index].likelihood()
        p_ys_given_x_t = lik * p_y_next_given_x[current_index+1]
        p_ys_x_t_given_x_t_minus_1 = p_ys_given_x_t * x_dist
        p_y_next_given_x[current_index] = p_ys_x_t_given_x_t_minus_1.marginalize_out(xs[current_index])

        current_index -= 1
    return xs, ys, p_y_next_given_x

def old_compute_conditional_filtering_posterior(t, num_obs, xs, ys, p_y_next_given_x, A, C):
    if t == 0:
        lik = ys[0].likelihood()
        cond_ys = lik * p_y_next_given_x[0]
        prior = xs[0].prior()
        numerator = cond_ys * prior
        denominator = JointVariables(ys, A=A, C=C).dist
    else:
        lik = xs[t].likelihood()
        numerator = ys[t].likelihood()
        if t < num_obs - 1:
            numerator *= p_y_next_given_x[t]
        numerator *= lik
        denominator = p_y_next_given_x[t-1]

    return FilteringPosterior(numerator, denominator.left + [x for x in [denominator.right] if x is not None])

def compute_conditional_filtering_posterior(t, num_obs, xs, ys, A, C, m=1):
    rest_of_ys = ys[t+1:] if m == 0 else ys[t+1:t+m]
    rvars = [ys[t], xs[t]] if t < len(ys) else []
    rvars += rest_of_ys
    rvars += [xs[t-1]] if t > 0 else []
    jvs = JointVariables(rvars, A=A, C=C)
    condition_vars = [ys[t]] if t < len(ys) else []
    condition_vars += rest_of_ys
    condition_vars = condition_vars + [xs[t-1]] if t > 0 else condition_vars
    return FilteringPosterior(jvs.dist, condition_vars)

def compute_filtering_posterior(t, num_obs, xs, ys, A, C, m=1):
    rest_of_ys = ys[t+1:] if m == 0 else ys[t+1:t+m]
    rvars = [ys[t], xs[t]] if t < len(ys) else []
    rvars += rest_of_ys
    jvs = JointVariables(rvars, A=A, C=C)
    condition_vars = [ys[t]] if t < len(ys) else []
    condition_vars += rest_of_ys
    return FilteringPosterior(jvs.dist, condition_vars)

def old_compute_conditional_filtering_posteriors(table, num_obs, dim, ys=None):
    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    mu_0 = table[dim]['mu_0']
    Q_0 = table[dim]['Q_0']

    if ys is None:
        ys = generate_trajectory(num_obs, A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0)[0]

    assert(num_obs == len(ys))

    fp_xs, fp_ys, p_y_next_given_x = compute_filtering_data_structures(dim=dim, num_obs=num_obs)

    # true evidence
    jvs = JointVariables(fp_ys, A=A, C=C)
    # print('true evidence: ', jvs.dist.log_prob(ys).exp())

    fps = []
    for t in range(num_obs):
        filtering_posterior = old_compute_conditional_filtering_posterior(t, num_obs, fp_xs, fp_ys, p_y_next_given_x, A, C)
        fps.append(filtering_posterior)
    return fps, ys

def compute_conditional_filtering_posteriors(table, num_obs, dim, m=0, condition_on_x=True, ys=None):
    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    mu_0 = table[dim]['mu_0']
    Q_0 = table[dim]['Q_0']

    return _compute_conditional_filtering_posteriors(A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0,
                                                     num_obs=num_obs, dim=dim, m=m, condition_on_x=condition_on_x, ys=ys)

def _compute_conditional_filtering_posteriors(A, Q, C, R, mu_0, Q_0, num_obs, dim, m=0, condition_on_x=True, ys=None):
    lgv = get_linear_gaussian_variables(dim=dim, num_obs=num_obs)

    # true evidence
    jvs = JointVariables(lgv.ys, A=A, C=C)
    # print('true evidence: ', jvs.dist.log_prob(ys).exp())

    fps = []
    for t in range(num_obs):
        if condition_on_x:
            filtering_posterior = compute_conditional_filtering_posterior(t, num_obs, lgv.xs, lgv.ys, A, C, m=m)
        else:
            filtering_posterior = compute_filtering_posterior(t, num_obs, lgv.xs, lgv.ys, A, C, m=m)
        fps.append(filtering_posterior)
    return fps

def test_filtering_posterior():
    dim = 1
    t = 1
    num_obs = 5
    table = create_dimension_table(dimensions=[dim], random=False)
    A = table[dim]['A']
    C = table[dim]['C']

    xs, ys, p_y_next_given_x = compute_filtering_data_structures(dim=dim, num_obs=num_obs)
    fp1 = old_compute_conditional_filtering_posterior(t=0, num_obs=num_obs, xs=xs, ys=ys, p_y_next_given_x=p_y_next_given_x, A=A, C=C)
    fp2 = old_compute_conditional_filtering_posterior(t=1, num_obs=num_obs, xs=xs, ys=ys, p_y_next_given_x=p_y_next_given_x, A=A, C=C)
    fp3 = old_compute_conditional_filtering_posterior(t=2, num_obs=num_obs, xs=xs, ys=ys, p_y_next_given_x=p_y_next_given_x, A=A, C=C)
    fp4 = old_compute_conditional_filtering_posterior(t=3, num_obs=num_obs, xs=xs, ys=ys, p_y_next_given_x=p_y_next_given_x, A=A, C=C)

    mu = fp2.denominator.mean(value=torch.ones(1))
    fp_dist = fp2.condition(mu)
    fp_dist.mean(value=torch.tensor(1.))

def compare_truncated_posterior(table, num_obs, dim, condition_length):
    A = table[dim]['A']
    Q = table[dim]['Q']
    C = table[dim]['C']
    R = table[dim]['R']
    mu_0 = table[dim]['mu_0']
    Q_0 = table[dim]['Q_0']

    traj_ys, _ = generate_trajectory(num_obs, A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0)[0:2]

    fps = compute_conditional_filtering_posteriors(table, num_obs, dim, m=0, condition_on_x=False, ys=None)
    fps_m = compute_conditional_filtering_posteriors(table, num_obs, dim, m=condition_length, condition_on_x=False, ys=None)

    kls = []
    for i, (f, f_m) in enumerate(zip(fps, fps_m)):
        td = f.condition(y_values=traj_ys[i:])
        td_m = f_m.condition(y_values=traj_ys[i:condition_length+i])
        kls.append(dist.kl_divergence(td.get_dist(), td_m.get_dist()))
    return kls


if __name__ == "__main__":
    dim = 1
    num_obs = 20
    table = create_dimension_table(torch.tensor([dim]), random=False)
    for condition_length in np.arange(2, 15):
        kls = compare_truncated_posterior(table, num_obs, dim, condition_length=condition_length)
        print(kls)


    # A = table[dim]['A']
    # Q = table[dim]['Q']
    # C = table[dim]['C']
    # R = table[dim]['R']
    # mu_0 = table[dim]['mu_0']
    # Q_0 = table[dim]['Q_0']

    # from linear_gaussian_env import LinearGaussianEnv
    # env = LinearGaussianEnv(A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0, ys=ys, sample=True)

    # from evaluation import evaluate_filtering_posterior
    # evaluate_filtering_posterior(ys=ys, N=2, tds=fps, env=env)
