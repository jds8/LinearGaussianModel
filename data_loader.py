# general imports
from typing import List
import torch
import pandas as pd
from pathlib import Path


class QuantileData:
    def __init__(self, med, lower_ci, upper_ci, data_label):
        self.med = med
        self.lower_ci = lower_ci
        self.upper_ci = upper_ci
        self.data_label = data_label


class DataWithColumns:
    def __init__(self, data, columns, data_label, data_type):
        self.data = data
        self.columns = columns
        self.data_label =data_label
        self.data_type =data_type


def load_data(filename, qtiles):
    df = pd.read_csv(filename, index_col=0)

    # get all of the non index columns
    data = torch.tensor(df[[str(x) for x in df.columns]]).to_numpy()

    # get the desired quantiles
    quantiles = torch.tensor(qtiles, dtype=data.dtype)
    lower_ci, med, upper_ci = torch.quantile(data, quantiles, dim=0)

    data_label = filename.split('(')[0]

    return QuantileData(med=med, lower_ci=lower_ci, upper_ci=upper_ci, data_label=data_label)

def load_dataset(names, qtiles):
    dataset = []
    for name in names:
        dataset.append(load_data(name, qtiles))
    return dataset

def load_ess_data(filename):
    df = pd.read_csv(filename, index_col=0)

    # get all of the non index columns
    data = torch.tensor(df[[str(x) for x in df.columns]].to_numpy())

    data_label = Path(filename).stem.split('(')[0]

    if filename.endswith('dim.csv'):
        data_type = 'dim'
    elif filename.endswith('traj.csv'):
        data_type = 'traj'
    else:
        raise NotImplementedError

    return DataWithColumns(data, df.columns, data_label, data_type)

