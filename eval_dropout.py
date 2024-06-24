import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import classification_report
from utils import (
    lab_to_int, 
    lab_to_long, 
    make_dataset, 
    load_data, 
    compute_predictive_variance, 
    plot_images, 
    store_predictions, 
    store_confusion_matrix, 
    store_summary_stats,
    plot_uncertainty,
    make_calibration_plots,
    make_ordered_calibration_plot,
    )
from optimizer import StochasticGradientLangevinDynamics

# hyperparameters
path_to_model = './ensemble/ensemble/20240606_122403.keras'
destination = './stats/dropout_stats/'
image_size = [224, 224]
batch_size = 32
num_samples = 10

os.makedirs(destination, exist_ok=True)

# load the model
model = tf.keras.models.load_model(path_to_model, custom_objects={'pSGLangevinDynamics': StochasticGradientLangevinDynamics})
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
path_to_val_data = './data/Training_Dataset_Cropped_Split/val/'
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

