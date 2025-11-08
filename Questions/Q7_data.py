import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import copy

# Generate 2D data with 4 cluster centers
def generate_data(n_samples=500, std=0.8):
    centers = [(-2, -2), (2, -2), (-2, 2), (2, 2)]
    X, Y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=std, random_state=42)
    return X, Y

# Generate Data
X, Y = generate_data()

X_mixup, Y_mixup = mixup(X, Y)
X_cutmix, Y_cutmix = cutmix(X, Y)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Raw Data Visualisation
ax[0].scatter(X[:, 0], X[:, 1], c=Y, cmap='coolwarm', alpha=0.8)
ax[0].set_title("Original Data")

# MixUp Visualisation
ax[1].scatter(X_mixup[:, 0], X_mixup[:, 1], c=Y_mixup, cmap='coolwarm', alpha=0.8)
ax[1].set_title("MixUp Data")

# CutMix Visualisation
ax[2].scatter(X_cutmix[:, 0], X_cutmix[:, 1], c=Y_cutmix, cmap='coolwarm', alpha=0.8)
ax[2].set_title("CutMix Data")

plt.show()