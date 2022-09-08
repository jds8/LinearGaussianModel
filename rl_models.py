# general imports
from stable_baselines3 import PPO, SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.callbacks import BaseCallback
import torch


PPO_STR = "PPO"
SAC_STR = "SAC"


def get_rl_type_from_str(rl_type):
    if rl_type == 'PPO':
        return PPO
    elif rl_type == 'SAC':
        return SAC
    else:
        raise NotImplementedError


def load_rl_model(model_name, device, env=None):
    """ Loads the RL model """
    while model_name.endswith('.zip'):
        model_name = model_name[0:-4]

    if PPO_STR in model_name:
        rl_type = PPO
    elif SAC_STR in model_name:
        rl_type = SAC
    else:
        raise BaseException("Cannot determine rl model type for: {}".format(model_name))

    try:
        model = rl_type.load(model_name, env=env, device=device)
        print('loaded model {}'.format(model_name))
    except:
        model = rl_type.load(model_name+'.zip', env=env, device=device)
        print('loaded model {}'.format(model_name)+'.zip')
    return model, model.policy.to(device)


def evaluate_actions(policy, obs, action):
    """ Evaluate/score action under model given observation """
    if isinstance(policy, SACPolicy):
        mean_actions, log_std, _ = policy.actor.get_action_dist_params(obs)
        sac_action = torch.tanh(action)
        return policy.actor.action_dist.proba_distribution(mean_actions, log_std).log_prob(sac_action)
    return policy.evaluate_actions(obs=obs, actions=action)[1].sum()
