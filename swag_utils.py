import os
import numpy as np
import tensorflow as tf


class SWAGDiagonalCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, path=None):
        self.model = model
        self.path = path
        self.mean = [tf.zeros_like(w) for w in model.trainable_variables]
        self.squared_sum = [tf.zeros_like(w) for w in model.trainable_variables]
        self.deviation_vectors = []
        
    def on_epoch_end(self, epoch, logs=None):
        self.mean = [m * epoch / (epoch + 1) for m in self.mean]
        self.mean = [m + w / (epoch + 1) for m, w in zip(self.mean, self.model.trainable_variables)]
        self.squared_sum = [s * epoch / (epoch + 1) for s in self.squared_sum]
        self.squared_sum = [s + tf.square(w) / (epoch + 1) for s, w in zip(self.squared_sum, self.model.trainable_variables)]
        self.deviation_vectors.append([w - m for w, m in zip(self.model.trainable_variables, self.mean)])
        
    def on_train_end(self, logs=None):
        np.savez(os.path.join(self.path, 'swag_diagonal_mean.npz'), *self.mean)
        np.savez(os.path.join(self.path, 'swag_diagonal_squared_sum.npz'), *self.squared_sum)
        np.savez(os.path.join(self.path, 'swag_deviation_vectors.npz'), *self.deviation_vectors)

def assign_weights(model, weights):
    for w, i in zip(weights, range(len(weights))):
        model.trainable_variables[i].assign(w)

def read_weights(path):
    with np.load(path) as f:
        weights = [f[key] for key in f.keys()]
    return weights
