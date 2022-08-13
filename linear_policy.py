#!/usr/bin/env python3

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

# Copied from: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html

class LinearNetwork(nn.Module):
    """
    Linear network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_state_dimension: int,
        last_layer_dim_pi: int = 4,
        last_layer_dim_vf: int = 4,
    ):
        super(LinearNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.hidden_state_dimension = hidden_state_dimension
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Linear(feature_dim, last_layer_dim_pi)
        # Value network
        self.value_net = nn.Linear(feature_dim, last_layer_dim_vf)

    def _mask_features(self, features: th.Tensor):
        n, m = features.shape
        idx = features[:, self.hidden_state_dimension].to(th.int).reshape(-1, 1)

        # create index matrix.
        # Note that each element of idx represents the number of ys we want to ignore,
        # which is why we start indexing at -self.hidden_state_dimension
        I = th.arange(-self.hidden_state_dimension, m-self.hidden_state_dimension).repeat(n, 1)

        # create mask out of indices which are after indices in b
        M = I > idx

        # mask the features ys after indices in idx at each row
        return th.mul(features, M)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        in_features = self._mask_features(features)
        policy_net = self.policy_net(in_features)
        value_net = self.value_net(in_features)
        return policy_net, value_net

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        in_features = self._mask_features(features)
        return self.policy_net(in_features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        in_features = self._mask_features(features)
        return self.value_net(in_features)


class LinearActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        hidden_state_dimension: int = 1,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        self.hidden_state_dimension = hidden_state_dimension
        super(LinearActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = LinearNetwork(self.features_dim,
                                           self.hidden_state_dimension,
                                           last_layer_dim_pi=self.action_space.shape[0],
                                           last_layer_dim_vf=self.action_space.shape[0])
