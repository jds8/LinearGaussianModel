# general imports
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from plot_utils import legend_without_duplicate_labels
import torch
from typing import List
from data_loader import QuantileData, DataWithColumns
import wandb
import numpy as np

def plot_ess_quantile_data(dataset: List[QuantileData]):
    for data_obj in dataset:
        med = data_obj.med
        data_label = data_obj.data_label
        lower_ci = data_obj.lower_ci
        upper_ci = data_obj.upper_ci
        x_vals = torch.arange(1, len(med)+1)
        plt.plot(x_vals, med.squeeze(), label=data_label)
        plt.fill_between(x_vals, y1=lower_ci, y2=upper_ci, alpha=0.3)

def plot_ess_data(data_obj: DataWithColumns):
    data = data_obj.data
    columns = data_obj.columns
    data_label = data_obj.data_label

    quantiles = torch.tensor([0.05, 0.5, 0.95], dtype=data.dtype)
    lower_ci, med, upper_ci = torch.quantile(data, quantiles, dim=0)

    x_vals = torch.arange(1, len(med)+1)
    plt.plot(x_vals, med.squeeze(), label=data_label)
    plt.fill_between(x_vals, y1=lower_ci, y2=upper_ci, alpha=0.3)

def plot_state_occupancy(state_occupancies, quantiles, traj_length, ent_coef, loss_type):
    for state_occupancy, name in state_occupancies:
        quants = torch.tensor(quantiles, dtype=state_occupancy.dtype)
        lwr, med, upr = torch.quantile(state_occupancy, quants, dim=0)
        x_data = torch.arange(1, traj_length+1)
        plt.plot(x_data, med.squeeze(), label=name)
        plt.fill_between(x_data, y1=lwr, y2=upr, alpha=0.3)
    plt.xlabel('Trajectory Step (of {})'.format(traj_length))
    plt.ylabel('xt')
    plt.title('Values of state xt at each time step t\n(Loss Type: {} Coef: {})'.format(loss_type, ent_coef))
    plt.legend()
    plt.savefig('{}/State Occupancy traj_len: {} ent_coef: {} loss_type: {}.pdf'.format(TODAY, traj_length, ent_coef, loss_type))
    wandb.save('{}/State Occupancy traj_len: {} ent_coef: {} loss_type: {}.pdf'.format(TODAY, traj_length, ent_coef, loss_type))
    plt.close()

def plot_3d_state_occupancy(state_occupancy_dict, quantiles, traj_length, ent_coef, loss_type, today_dir):
    colors = iter(cm.rainbow(np.linspace(0, 1, len(state_occupancy_dict))))
    ax = plt.axes(projection='3d')
    for name, state_occupancy in state_occupancy_dict.items():
        color = next(colors)
        # quants = torch.tensor(quantiles, dtype=state_occupancy.dtype)
        # lwr, med, upr = torch.quantile(state_occupancy, quants, dim=0)
        for c in range(traj_length):
            ordered_hist_data, _ = state_occupancy[:, c].sort()
            freqs, bins = torch.histogram(ordered_hist_data)
            ax.plot3D(bins[0:-1], torch.tensor(c+1).repeat(bins.nelement()-1), freqs, label=name, color=color)
    plt.ylabel('Trajectory Step (of {})'.format(traj_length))
    plt.xlabel('xt')
    #plt.zlabel('Frequency')
    plt.title('Values of state xt at each time step t\n(Loss Type: {} Coef: {})'.format(loss_type, ent_coef))
    legend_without_duplicate_labels(plt.gca())
    plt.savefig('{}/3DState Occupancy traj_len: {} ent_coef: {} loss_type: {}.pdf'.format(today_dir, traj_length, ent_coef, loss_type))
    wandb.save('{}/3DState Occupancy traj_len: {} ent_coef: {} loss_type: {}.pdf'.format(today_dir, traj_length, ent_coef, loss_type))
    plt.close()
