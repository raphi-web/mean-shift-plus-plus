from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from mean_shift import mean_shift_pp, mean_shift_pp_spatial, mean_shift_spatial
from PIL import Image
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = datetime.now()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        end_time = datetime.now()
        delta = end_time - self.start_time
        total_seconds = delta.total_seconds()
        minutes, seconds = divmod(total_seconds, 60)
        seconds, milliseconds = divmod(seconds * 1000, 1000)
        print(f"{self.name}: {int(minutes):02d}:{int(seconds):02d}.{milliseconds}")


image = np.array(Image.open("./input_files/test-image.jpg")).astype(np.float64)
h, w, c = image.shape
data = image.reshape(h * w, -1).astype(np.float64)
print("Image Shape: ", image.shape)

color_radius = 10
window_size = 7
max_iter = 30


# Comparing the different mean-shifts
with Timer("Mean-Shift-Spatial"):
    ms_sp_out = mean_shift_spatial(
        image,
        win_size=window_size,
        color_radius=color_radius,
        max_iter=max_iter,
        threshold=1,
    )

with Timer("Mean-Shift++"):
    ms_pp_out = mean_shift_pp(
        data, band_width=color_radius, threshold=1, max_iter=max_iter
    )


with Timer("Mean-Shift++-Spatial"):
    ms_pp_sp_out = mean_shift_pp_spatial(
        image,
        win_size=window_size,
        color_radius=color_radius,
        max_iter=max_iter,
        threshold=1,
    )

img_a = Image.fromarray(ms_sp_out.astype(np.uint8))
img_b = Image.fromarray(ms_pp_out.reshape(h, w, c).astype(np.uint8))
img_c = Image.fromarray(ms_pp_sp_out.astype(np.uint8))

fig, axs = plt.subplots(1, 3, figsize=(14, 5))
[ax.axis("off") for ax in axs]
axs[0].imshow(img_a)
axs[0].set_title("Spatial Mean-Shift")
axs[1].imshow(img_b)
axs[1].set_title("Mean-Shift++")
axs[2].imshow(img_c)
axs[2].set_title("Spatial Mean-Shift++")
plt.tight_layout()
plt.savefig("./output_files/result-1.png", dpi=200)
plt.show()

# activate if you want to see the clustering in feature space
X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)

result = mean_shift_pp(X, band_width=1, threshold=1, max_iter=100)
cluster_centers = np.unique(np.round(result, 0), axis=1)
nbrs = NearestNeighbors(n_neighbors=1).fit(cluster_centers)
distances, indices = nbrs.kneighbors(X)
labels = indices.flatten()

plt.figure(figsize=(12, 12))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
plt.scatter(
    cluster_centers[:, 0],
    cluster_centers[:, 1],
    s=300,
    c="red",
    marker="x",
)
plt.title("MeanShift++ Clustering")
plt.savefig("./output_files/result-2.png", dpi=200)
plt.show()
