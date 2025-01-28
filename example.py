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
        print(f"{self.name}: {int(minutes)}:{int(seconds)}.{milliseconds}")


# h, w, c = (50, 50, 3)
# image = np.random.randint(0, 255, (h, w, c)).astype(np.float64)

image = np.array(Image.open("./input_files/test-image.jpg")).astype(np.float64)
h, w, c = image.shape
data = image.reshape(h * w, -1).astype(np.float64)

# Comparing them both is not wise, since the spacial Ms works on grid data
with Timer("Mean-Shift-Spatial"):
    ms_sp_out = mean_shift_spatial(
        image, win_size=7, color_radius=10, max_iter=100, threshold=1
    )

with Timer("Mean-Shift++"):
    ms_pp_out = mean_shift_pp(data, band_width=10, threshold=1, max_iter=100)

print("Image Shape: ", image.shape)

with Timer("Mean-Shift++-Spatial"):
    ms_pp_sp_out = mean_shift_pp_spatial(
        image, win_size=7, color_radius=10, max_iter=100, threshold=1
    )

img = Image.fromarray(ms_sp_out.astype(np.uint8))
img.save("./output_files/Mean-Shift-Spatial.jpg")
img = Image.fromarray(ms_pp_out.reshape(h, w, c).astype(np.uint8))
img.save("./output_files/Mean-Shift_pp.jpg")
img = Image.fromarray(ms_pp_sp_out.astype(np.uint8))
img.save("./output_files/Mean-Shift-pp-Spatial.jpg")


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
plt.show()
