import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

def visualize_kspaces(kspaces):
    n = kspaces.shape[2]
    fig, axs = plt.subplots(n, 2, squeeze=False)

    for i in range(n):
        axs[i, 1].imshow(np.angle(kspaces[:, :, i]))
        axs[i, 0].imshow(np.abs(kspaces[:, :, i]), norm=colors.LogNorm())

def visualize_single_kspace(kspace):
    fig, axs = plt.subplots(1, 2)
    axs[1].imshow(np.angle(kspace))
    axs[0].imshow(np.abs(kspace), norm=colors.LogNorm())


def visualize_seperate_kspaces(kspaces):
    n = kspaces.shape[2]

    for i in range(n):
        fig, axs = plt.subplots(1, 2, squeeze=False)
        axs[0, 1].imshow(np.angle(kspaces[:, :, i]))
        axs[0, 0].imshow(np.abs(kspaces[:, :, i]), norm=colors.LogNorm())

