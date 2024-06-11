import os
import random
import numpy as np
import tensorflow as tf
from swag_utils import assign_weights, read_weights
from optimizer import StochasticGradientLangevinDynamics
from utils import make_dataset, load_data

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

path_to_weights = "./ensemble/ensemble/20240606_122403.keras"
path_to_val_data = './data/Training_Dataset_Cropped_Split/val/'
image_size = [224, 224]
batch_size = 32
destination = './swag_ensemble_135353'
update_bn = False
X_train, y_train = load_data('./data/Training_Dataset_Cropped_Split/train/')

ds_train = make_dataset(X_train, y_train, image_size, batch_size, shuffle=True, seed=1)




model = tf.keras.models.load_model(path_to_weights, custom_objects={'pSGLangevinDynamics': StochasticGradientLangevinDynamics})

mean_weights = read_weights("./swag/20240607_135353/swag_diagonal_mean.npz")
mean_squared = read_weights("./swag/20240607_135353/swag_diagonal_squared_sum.npz")

variances = [s - m**2 for s, m in zip(mean_squared, mean_weights)]
variances = [tf.clip_by_value(v, 0, 1e5) for v in variances]

for i in range(10):
    delta = []
    for j, w in enumerate(variances):
        delta.append(tf.random.stateless_normal(shape=w.shape, stddev=tf.sqrt(w), seed=[i, j]))
    print(delta[100][:10])
    theta = [m + d for m, d in zip(mean_weights, delta)]
    assign_weights(model, theta)
    
    ### if update batch norm layers
    if update_bn:
        for batch in ds_train:
            X_batch, y_batch = batch
            model(X_batch, training=True)
    ###
    
    X_val, y_val = load_data(path_to_val_data)
    ds_val = make_dataset(X_val, y_val, image_size, batch_size, shuffle=False, seed=1)
    model.evaluate(ds_val)
    
    model.save(os.path.join(destination, f"model_{i}.keras"))
