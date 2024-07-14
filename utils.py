import torch
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val

def AddGausNoise(arr, std=1.0):
    add = np.random.normal(scale=std,size=arr.shape)
    return arr + add

def AddUnifNoise(arr, low=-1.0, high=1.0):
    return arr + np.random.uniform(low, high, arr.shape)

def RandomErase(arr, min_ratio=0.01, max_ratio=0.1):
    num_erase = np.random.randint(int(len(arr)*min_ratio),int(len(arr)*max_ratio))
    indices = np.random.choice(len(arr), replace=False, size=num_erase)
    erased = arr.copy()
    erased[indices] = 0
    return erased

def Normalize(arr, mean=0.0, std=1.0):
    return (arr - mean)/std

def plot_hist(arr):
    # start, stop = np.log10(min(arr)), np.log10(max(arr))
    # bins = 20 ** np.linspace(start, stop, 10)
    # plt.hist(arr, density=True, log=True, bins=bins)
    # plt.xscale('log')

    plt.hist(arr, density=True, bins=20, edgecolor='white')
    # sns.histplot(arr,stat='density',edgecolor='white', bins=20)
    # sns.kdeplot(arr, color='black',clip=(0,None))
    # plt.yscale('log')
    plt.ylabel('Density')
    plt.xlabel('Abs(diff)')
    