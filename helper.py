import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

def visualize_kspaces(kspaces):
    n = kspaces.shape[2]
    fig, axs = plt.subplots(n, 2, squeeze=False)

    for i in range(n):
        axs[i, 1].imshow(np.angle(kspaces[:, :, i]))
        axs[i, 0].imshow(np.abs(kspaces[:, :, i]), norm=colors.LogNorm())
    plt.show()

def visualize_single_kspace(kspace):
    fig, axs = plt.subplots(1, 2)
    axs[1].imshow(np.angle(kspace))
    axs[0].imshow(np.abs(kspace), norm=colors.LogNorm())
    plt.show()


def visualize_seperate_kspaces(kspaces):
    n = kspaces.shape[2]

    for i in range(n):
        fig, axs = plt.subplots(1, 2, squeeze=False)
        axs[0, 1].imshow(np.angle(kspaces[:, :, i]))
        axs[0, 0].imshow(np.abs(kspaces[:, :, i]), norm=colors.LogNorm())
    plt.show()

def zero_outside_circle(array, r):
    ret_array = np.copy(array)
    x = np.arange(array.shape[0])
    y = np.arange(array.shape[1])
    cx = array.shape[0] / 2
    cy = array.shape[1] / 2

    ret_array[(x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 >= r ** 2] = 0
    return ret_array

def ifft(ksapce, axes=None):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(ksapce, axes=axes), axes=axes), axes=axes)

def fft(image, axes=None):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(image, axes=axes), axes=axes), axes=axes)

def visualize_images(images):
    fig, axs = plt.subplots(1, images.shape[2])
    for i in range(images.shape[2]):
        axs[i].imshow(np.abs(images[:, :, i]))
    plt.show()

def imshow(image):
    plt.imshow(np.abs(image))