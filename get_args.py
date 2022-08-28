# general imports
import argparse
import torch

# main arg-parse
def get_args():

    #
    parser = argparse.ArgumentParser(description='Parser')

    # waypoint-follows
    parser.add_argument('--traj_length', type=int, default=10, help='trajectory length')
    parser.add_argument('--dim', type=int, default=1, help='dimension of hidden state')
    parser.add_argument('--loss_type', type=str, default='forward_kl', help='type of KL regularization')
    parser.add_argument('--ent_coef', type=float, default=0.1, help='coefficient on KL regularization term (loss_type)')
    parser.add_argument('--subroutine', type=str, default='train_agent', help='what subroutine to run (train_agent, evaluate_agent, ess_dim, ess_traj, load_ess_dim)')
    parser.add_argument('--filenames', type=str, default='', help='what file to load', nargs='*')
    parser.add_argument('--ess_dir', type=str, default='', help='what directory of ess files to load')
    parser.add_argument('--data_type', type=str, default='traj', help='what type of data (traj or dim) to load')
    parser.add_argument('--initial_idx', type=int, default=0, help='initial index for x-axis of ESS plot')
    parser.add_argument('--ess_dims', type=int, default=1, help='dimensions for ess', nargs='*')
    parser.add_argument('--ess_traj_lengths', type=int, default=1, help='traj_lengths for ess', nargs='*')
    parser.add_argument('--ess_condition_lengths', type=int, default=1, help='condition_lengths for ess', nargs='*')
    parser.add_argument('--delta', type=float, default=0.03, help='the largest deviation between the evidence estimate ratio and 1')
    parser.add_argument('--condition_length', type=int, default=0, help='the number of observations on which to condition')
    parser.add_argument('--continue_training', type=bool, default=False, help='whether to continue training the rl model')
    parser.add_argument('--use_mlp_policy', type=bool, default=False, help='whether to use an MLP policy')

    # hyperparameters
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='what file to load')
    parser.add_argument('--clip_range', type=float, default=0.1, help='what file to load')

    # save directory
    parser.add_argument('--save_dir', type=str, default='linear_gaussian_data/', help='top level directory in which to save files')

    # environment type
    parser.add_argument('--env_type', type=str, default='LinearGaussianEnv', help='the type of environment')

    # Number of samples to take
    parser.add_argument('--num_samples', type=int, default=100, help='the number of samples to take for ESS, state occupancy, etc.')
    parser.add_argument('--rl_timesteps', type=int, default=500000, help='the number of RL interactions with the environment')

    # Args that you should only use for validation
    parser.add_argument('--ignore_reward', type=bool, default=False, help='whether or not to ignore the reward in the loss')

    # parse
    args, _ = parser.parse_known_args()

    # return it all
    return args, parser
