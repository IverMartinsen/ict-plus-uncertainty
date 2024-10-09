import os
import numpy as np
import tensorflow as tf
from utils import make_dataset, load_data, lab_to_int
from optimizer import StochasticGradientLangevinDynamics
from schedule import PolynomialDecay

path_to_model =  "./models/20240905_080209.keras"
path_to_val_data = './data/Man vs machine_Iver_cropped/'
image_size = [224, 224]
batch_size = 32
destination = './results/aleatoric_results/'
os.makedirs(destination, exist_ok=True)

model = tf.keras.models.load_model(path_to_model, custom_objects={
    'StochasticGradientLangevinDynamics': StochasticGradientLangevinDynamics,
    'PolynomialDecay': PolynomialDecay
    })

logit = tf.keras.Model(inputs=model.input, outputs=model.layers[-4].output)
var = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)

X_val, y_val = load_data(path_to_val_data)
ds_val = make_dataset(X_val, y_val, image_size, batch_size, seed=1)


preds = []
vars = []
logits = []
for batch in ds_val:
    X_batch, y_batch = batch
    z = logit(X_batch)
    y = tf.nn.softmax(z, axis=-1)
    v = var(X_batch)
    preds.append(y)
    vars.append(v)
    logits.append(z)
preds = np.concatenate(preds, axis=0)
vars = np.concatenate(vars, axis=0)
logits = np.concatenate(logits, axis=0)
    
Y_pred = np.empty((len(y_val), 1, len(lab_to_int)))
Y_var = np.empty((len(y_val)))
Y_logit = np.empty((len(y_val), 1, len(lab_to_int)))
Y_pred[:, 0, :] = preds
Y_var = vars
Y_logit[:, 0, :] = logits

np.save(os.path.join(destination, 'predictions.npy'), Y_pred)
np.save(os.path.join(destination, 'variances.npy'), Y_var)
np.save(os.path.join(destination, 'labels.npy'), y_val)
np.save(os.path.join(destination, 'filenames.npy'), X_val)
np.save(os.path.join(destination, 'logits.npy'), Y_logit)