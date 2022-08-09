# general imports
import argparse
import torch

# main arg-parse
def get_args():

    #
    parser = argparse.ArgumentParser(description='Parser')

    # waypoint-follows
    parser.add_argument('--traj_len', type=int, default=10, help='trajectory length')
    parser.add_argument('--dim', type=int, default=1, help='dimension of hidden state')
    parser.add_argument('--loss_type', type=str, default='forward_kl', help='type of KL regularization')
    parser.add_argument('--ent_coef', type=int, default=0.1, help='coefficient on KL regularization term (loss_type)')

    # parse
    args, _ = parser.parse_known_args()

    # return it all
    return args, parser
