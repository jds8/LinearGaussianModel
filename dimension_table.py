#!/usr/bin/env python3
import torch

def create_dimension_table(dimensions, random=False):
    table = {}
    if random:
        for dim in dimensions:
            dimension = dim.item() if isinstance(dim, torch.Tensor) else dim
            table[dimension] = {}
            table[dimension]['A'] = torch.rand(dimension, dimension)
            table[dimension]['Q'] = gen_covariance_matrix(dimension)
            table[dimension]['C'] = torch.rand(1, dimension)
            table[dimension]['R'] = torch.rand(1, 1)
            table[dimension]['mu_0'] = torch.zeros(dimension)
            table[dimension]['Q_0'] = table[dimension]['Q']
    else:
        for dim in dimensions:
            dimension = dim.item() if isinstance(dim, torch.Tensor) else dim
            table[dimension] = {}
            table[dimension]['A'] = torch.eye(dimension, dimension)
            table[dimension]['Q'] = torch.eye(dimension)
            table[dimension]['C'] = torch.eye(1, dimension)
            table[dimension]['R'] = torch.eye(1, 1)
            table[dimension]['mu_0'] = torch.zeros(dimension)
            table[dimension]['Q_0'] = table[dimension]['Q']
    return table
