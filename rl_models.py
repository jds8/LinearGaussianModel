# general imports
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


def load_rl_model(model_name, device):
    # load model
    while model_name.endswith('.zip'):
        model_name = model_name[0:-4]
    try:
        model = PPO.load(model_name)
        print('loaded model {}'.format(model_name))
    except:
        model = PPO.load(model_name+'.zip')
        print('loaded model {}'.format(model_name)+'.zip')
    policy = model.policy.to(device)
    return model, policy
