#!/usr/bin/env python3
import torch
import torch.distributions as dist
import scipy.stats

# logsumexp
def logdiffexp(a, b):
    # computes log(exp(a) - exp(b))
    # note that (2*(a==small).type(torch.int)-1) is 1 if a==small and -1 otherwise
    small = min(a, b)
    big = max(a, b)
    return small + torch.log((1 - torch.exp(big - small))*(2*(a==small).type(torch.int)-1))

def logvarexp(vec: torch.Tensor):
    # computes the variance of exp(vec) and returns its log:
    # returns torch.log(torch.exp(vec).var())
    # sum[(e_i-bar(e))^2] = sum[e_i^2] - 2*bar(e)*sum[e_i] + n*bar(e)^2
    # sum(exp(x_i) - exp(bar(x)))
    logn = torch.log(torch.tensor(vec.nelement(), dtype=torch.float32))
    sample_avg = torch.logsumexp(vec, -1) - logn
    n_times_sample_avg_sq = 2*sample_avg + logn
    sq_sum_of_samples = torch.logsumexp(2*vec, -1)
    sq_terms = torch.logsumexp(torch.tensor([sq_sum_of_samples, n_times_sample_avg_sq]), -1)
    lse2 = torch.logsumexp(vec + sample_avg, -1) + torch.log(torch.tensor(2.))
    return logdiffexp(sq_terms, lse2) - logn

# confidence intervals
def calc_CI_from_mu_sigma(mu: torch.Tensor, sigma:torch.Tensor, N, alpha=torch.tensor(0.05)):
    Z = -dist.Normal(0, 1).icdf(alpha/2)
    std_err = sigma / np.sqrt(N)
    margin = std_err * Z
    return (mu + margin, mu - margin)

def calc_CI(data: torch.Tensor, alpha=torch.tensor(0.05)):
    N = len(data.squeeze())
    mu = data.mean()
    sigma = torch.sqrt(data.var())
    return calc_CI_from_mu_sigma(mu, sigma, N, alpha)

# diagnostic statistics for Importance Sampling
def log_mean(log_vec: torch.Tensor):
    return torch.logsumexp(log_vec, -1) - torch.log(torch.tensor(log_vec.nelement(), dtype=torch.float32))

def max_weight_proportion(vec: torch.Tensor):
    return vec.max() / vec.sum()

def log_max_weight_proportion(log_vec: torch.Tensor):
    return log_vec.max() - torch.logsumexp(log_vec, -1)

def effective_sample_size(weights: torch.Tensor):
    return weights.sum()**2 / (weights**2).sum()

def log_effective_sample_size(log_weights: torch.Tensor):
    return 2*torch.logsumexp(log_weights, -1) - torch.logsumexp(2*log_weights, -1)

# confidence intervals for IS
def importance_sampled_confidence_interval(mu, sigma, sample_size, epsilon=torch.tensor(0.05)):
    t = scipy.stats.t.ppf(q=1-epsilon/2, df=sample_size-1)
    return (mu - t*sigma, mu + t*sigma)

def band_matrix(band, num_copies):
    d = band.nelement()
    assert band.shape == torch.Size([1, d]) or band.shape == torch.Size([d, 1])
    outputs = []
    for i in range(num_copies):
        start_index = i*d
        c = torch.zeros(num_copies * d)
        c[start_index:start_index+d] = band.squeeze()
        outputs.append(c)
    return torch.stack(outputs)

def empirical_kl(p, q, samples=8):
    saps = p.rsample([samples, 1])
    return (p.log_prob(saps) - q.log_prob(saps)).mean()

def kl_divergence(p, q):
    """ The call to abs is to avoid nans during calls to backward() """
    return q.scale.log() - p.scale.log() + (p.scale / q.scale).abs().pow(2)/2 + ((p.loc - q.loc) / q.scale).abs().pow(2)/2 - 0.5

def gaussian_posterior(prior: dist.Normal, likelihood_var: torch.tensor, ob: torch.tensor, C: torch.tensor) -> dist.Normal:
    mu = (prior.mean * likelihood_var + C * ob * prior.variance) / (C**2 * prior.variance + likelihood_var)
    sigma_sq = (prior.variance * likelihood_var) / (C**2 * prior.variance + likelihood_var)

    return dist.Normal(mu, torch.sqrt(sigma_sq))

def integrate(posterior, state_transition_variance, A):
    """
    Multiplies two prob. dists and integrates out the variable that posterior is over
    Input:
    a filtering posterior: p(z_t|y_{1:t})
    a state transition: p(z_{t+1}|z_t)

    Output:
    predictive posterior: p(z_{t+1}|y_{1:t})
    """
    P = posterior.variance
    R = state_transition_variance

    lambda_xx = 1/P + A**2/R
    lambda_xy = -A/R
    lambda_yy = 1/R

    Sigma_inv = torch.tensor([[lambda_xx, lambda_xy], [lambda_xy, lambda_yy]])
    Sigma = torch.inverse(Sigma_inv)

    return dist.Normal(A*posterior.mean, torch.sqrt(Sigma[-1, -1]).item())

def kalman_filter(obs, prior: dist.Normal, state_transition_variance: torch.tensor,
                  likelihood_var: torch.tensor, A: torch.tensor, C: torch.tensor):
    """
    Computes p(z_t|y_{1:t}) for latents {z_t} and observations {y_t} where
    prior: p(z_1)
    state_transition: p(z_t|z_{t-1}) = N(Az_{t-1}, Q)
    likelihood: p(y_t|z_t) = N(Cz_t, R)
    """
    prediction = prior  # p(z_1)
    for ob in obs:
        filtering_posterior = gaussian_posterior(prediction, likelihood_var, ob, C)  # p(z_t|y_{1:t})
        prediction = integrate(filtering_posterior, state_transition_variance, A)  # p(z_{t+1}|y_{1:t})
    return filtering_posterior

def test_kalman_filter(obs, A, Q, C, R):
    prior = dist.Normal(0, torch.sqrt(Q))
    state_transition_variance = Q
    likelihood_var = R
    pred = kalman_filter(obs, prior, state_transition_variance, likelihood_var, A, C)
    print(pred.mean)
    print(pred.variance)
