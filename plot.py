# general imports
import matplotlib.pyplot as plt
import torch
from typing import List
from data_loader import QuantileData, DataWithColumns

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
