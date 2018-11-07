import numpy as np
import os
from imageio import imread
from os.path import join, split, splitext

batch_size = 128
epochs = 5
rows, cols = 128, 64


def load_iam_data(path, file_iter, size):
    """Parses images into numpy arrays."""
    x = np.ndarray(shape=(size, cols, rows))
    y = np.ndarray(shape=(size), dtype=object)

    for i in range(0, size):
        fi = next(file_iter)
        x[i] = imread(join(path, fi))
        y[i] = str(fi)[7:-4]

    return x, y

src = input("Source dir: ")
files = os.listdir(src)
file_iter = iter(files)

train_x, train_y = load_iam_data(src, file_iter, batch_size)
print(train_x.shape)
print(train_y.shape)
