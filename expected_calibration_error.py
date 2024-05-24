import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import make_dataset, load_data, lab_to_int, compute_calibration_stats, make_calibration_plots

image_size = [224, 224]
batch_size = 32
path_to_val_data = './data/Training_Dataset_Cropped_Split/val/'

# load the model
model = tf.keras.models.load_model('./ensemble/20240312_155425.keras')

# load the validation data
X_val, y_val = load_data(path_to_val_data)
ds_val = make_dataset(X_val, y_val, image_size, batch_size, shuffle=False)

# get the predictions
Y_pred = model.predict(ds_val)

make_calibration_plots(Y_pred, y_val, './ensemble_stats/', num_bins=9)

calibration_stats = compute_calibration_stats(Y_pred, y_val, num_bins=9)

preds = calibration_stats['preds_per_bin']
freqs = calibration_stats['freqs_per_bin']

for layer in model.layers[-2].layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        # print parameters
        vars = layer.get_weights()
        break

for batch in ds_val:
    X_batch, y_batch = batch
    model(X_batch, training=True)


model.evaluate(ds_val)