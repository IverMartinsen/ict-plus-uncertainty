import os
import random
import numpy as np
import tensorflow as tf
from utils.swag_utils import assign_weights, read_weights
from utils.optimizer import StochasticGradientLangevinDynamics
from utils.schedule import PolynomialDecay

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

path_to_weights = "./models/20240819_144329.keras"
source = "./models/20240828_110956"

model = tf.keras.models.load_model(path_to_weights, custom_objects={
    'StochasticGradientLangevinDynamics': StochasticGradientLangevinDynamics,
    'PolynomialDecay': PolynomialDecay})

mean_weights = read_weights(os.path.join(source, "swag_diagonal_mean.npz"))
mean_squared = read_weights(os.path.join(source, "swag_diagonal_squared_sum.npz"))

variances = [s - m**2 for s, m in zip(mean_squared, mean_weights)]
variances = [tf.clip_by_value(v, 0, 1e5) for v in variances]

for i in range(30):
    delta = []
    for j, w in enumerate(variances):
        delta.append(tf.random.stateless_normal(shape=w.shape, stddev=tf.sqrt(w), seed=[i, j]))
    theta = [m + d for m, d in zip(mean_weights, delta)]
    assign_weights(model, theta)
        
    model.save(os.path.join(source, f"model_{i}.keras"))
