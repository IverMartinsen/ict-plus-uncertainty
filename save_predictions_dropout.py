import os
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils import lab_to_int, make_dataset, load_data
from optimizer import StochasticGradientLangevinDynamics
from schedule import PolynomialDecay

# hyperparameters
path_to_model = './models/20240819_144329.keras'
destination = './stats/dropout_stats/'
image_size = [224, 224]
batch_size = 32
num_samples = 30

os.makedirs(destination, exist_ok=True)

# load the model
model = tf.keras.models.load_model(path_to_model, custom_objects={
    'StochasticGradientLangevinDynamics': StochasticGradientLangevinDynamics,
    'PolynomialDecay': PolynomialDecay,
    })
base_model = model.layers[-2]
classification_head = model.layers[-1]

# Set the batch normalization layers.trainable to False.
# That way, BN statistics are constant even when calling the model with training=True.
for i, layer in enumerate(base_model.layers):
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        print(layer)
        layer.trainable = False

def dropout_call(x):
    x = base_model(x, training=True)
    x = model.layers[-1](x, training=True)
    return x

# load the validation data
path_to_val_data = './data/Man vs machine_Iver_cropped/'
X_val, y_val = load_data(path_to_val_data)
ds_val = make_dataset(X_val, y_val, image_size, batch_size, seed=1)

# get the predictions
Y_pred = np.empty((len(y_val), num_samples, len(lab_to_int)))

tf.random.set_seed(1)
np.random.seed(1)
random.seed(1)

for i in tqdm(range(num_samples)):
    predictions = []
    for batch in ds_val:
        predictions.append(dropout_call(batch[0]))
    predictions = np.concatenate(predictions, axis=0)
    Y_pred[:, i, :] = predictions

np.save(os.path.join(destination, 'predictions.npy'), Y_pred)
np.save(os.path.join(destination, 'filenames.npy'), X_val)
np.save(os.path.join(destination, 'labels.npy'), y_val)
