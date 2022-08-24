import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from typing import List
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 12})


def visualize_map(data: np.ndarray, dst: str, ticklabels: List[str], xlabel: str = None, ylabel: str = None):
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(data,
                     linewidth=1,
                     vmin=0,
                     vmax=100,
                     square=True,
                     annot=True,
                     cmap="summer",
                     xticklabels=ticklabels,
                     yticklabels=ticklabels)
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    if ylabel is not None:
        plt.ylabel(ylabel, fontdict={'size': 28})
    if xlabel is not None:
        plt.xlabel(xlabel, fontdict={'size': 28})
    plt.tight_layout()
    plt.savefig(dst)
    plt.close()


def visualize_data(x: np.ndarray, dst: str):
    plt.figure(figsize=(20, 5))
    sns.heatmap(x, linewidth=1, square=True, cmap="YlGnBu")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(dst)
    plt.close()


def visualize_pooling(data_list: List[np.ndarray], dst: str, vmin: float = None, vmax: float = None):
    dim, num = data_list[0].shape
    tmp = None
    for i in range(len(data_list)):
        if i == 0:
            tmp = data_list[i]
        else:
            tmp = np.concatenate((tmp, np.zeros((1, num)), data_list[i]), axis=0)
    mask = np.zeros_like(tmp)
    if len(data_list) > 1:
        for i in range(len(data_list) - 1):
            mask[dim + i * (dim + 1), :] = True

    plt.figure(figsize=(5, 6))
    if vmin is not None and vmax is not None:
        sns.heatmap(tmp, mask=mask, linewidth=1, square=True, cmap="YlGnBu", vmin=vmin, vmax=vmax)
    else:
        sns.heatmap(tmp, mask=mask, linewidth=1, square=True, cmap="YlGnBu")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(dst)
    plt.close()
