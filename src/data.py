
import numpy as np
from tensorflow.keras.datasets import cifar10

from src.config import VAL_SIZE, SEED


def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train, y_test = y_train.flatten(), y_test.flatten()

    # Shuffle, then hold out part of the training set for validation
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(x_train))
    x_train, y_train = x_train[idx], y_train[idx]

    x_val, y_val = x_train[:VAL_SIZE], y_train[:VAL_SIZE]
    x_train, y_train = x_train[VAL_SIZE:], y_train[VAL_SIZE:]

    # Pixel scaling (0-1) happens inside the model, so raw 0-255 images work too.
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
