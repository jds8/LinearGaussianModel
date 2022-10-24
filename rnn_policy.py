#!/usr/bin/env python3

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

# Copied from: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html

class RNNWrapper(nn.RNN):
    def forward(self, inpt, hx=None):
        return super().forward(reversed(inpt), hx)

class RNNNetwork(nn.Module):
    """
    RNN network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        latent_dim: int,
        last_layer_dim_pi: int = 4,
        last_layer_dim_vf: int = 4,
        obs_size: int = 4,
        num_rnn_layers: int = 1,
        bidirectional: bool = False,
    ):
        super(RNNNetwork, self).__init__()

        self.latent_dim = latent_dim
        self.obs_size = obs_size

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # RNN
        self.rnn = RNNWrapper(1, obs_size, num_rnn_layers, batch_first=True, bidirectional=bidirectional)

        # latent_transform model
        self.latent_transform = nn.Sequential(
            nn.Linear(self.latent_dim, self.obs_size),
            nn.Tanh()
        )

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(self.obs_size, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(self.obs_size, last_layer_dim_vf), nn.ReLU()
        )

    def _extract_features(self, features: th.Tensor) -> th.Tensor:
        """
        This assumes that features contains [prev_xt, future_ys], i.e.
        that *only* future ys are passed in obs, *not all* obs.
        This is the case for the EnsembleLinearGaussianEnv
        """
        # _obs = features.squeeze()
        _obs = features
        try:
            prev_xt = _obs[:, :self.latent_dim]
        except:
            import pdb; pdb.set_trace()
            prev_xt = _obs[:, :self.latent_dim]

        future_ys = _obs[:, self.latent_dim:].reshape(features.shape[0], -1, 1)
        obs_output, _ = self.rnn(future_ys)
        if self.rnn.bidirectional:
            b = obs_output[:, -1, :args.obs_size] + obs_output[:, -1, args.obs_size:]
            num_rnn_terms = 2
        else:
            b = obs_output[:, -1, :]
            num_rnn_terms = 1
        h_combined = (self.latent_transform(prev_xt) + b) / (num_rnn_terms+1)
        return h_combined

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        extracted_features = self._extract_features(features)
        policy_net = self.policy_net(extracted_features)
        value_net = self.value_net(extracted_features)
        return policy_net, value_net

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        extracted_features = self._extract_features(features)
        return self.policy_net(extracted_features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        extracted_features = self._extract_features(features)
        return self.value_net(extracted_features)


class RNNActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        latent_dim: int,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        self.latent_dim = latent_dim
        super(RNNActorCriticPolicy, self).__init__(
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
        self.mlp_extractor = RNNNetwork(self.features_dim,
                                        self.latent_dim,
                                        last_layer_dim_pi=self.action_space.shape[0],
                                        last_layer_dim_vf=self.action_space.shape[0])
