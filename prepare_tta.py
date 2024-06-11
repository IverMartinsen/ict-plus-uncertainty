import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import make_dataset, load_data
from optimizer import StochasticGradientLangevinDynamics

data_path="./data/Training_Dataset_Cropped_Split/"
image_size = [224, 224]
batch_size = 64
random_seed = 1
path_to_model = '/Users/ima029/Desktop/IKT+ Uncertainty/Repository/tta/20240610_135338.keras'

X_train, y_train = load_data(data_path + '/train/')
X_val, y_val = load_data(data_path + '/val/')

ds_train = make_dataset(X_train, y_train, image_size, batch_size, shuffle=True, seed=random_seed)
ds_val = make_dataset(X_val, y_val, image_size, batch_size, shuffle=False, seed=random_seed)

model = tf.keras.models.load_model(path_to_model, custom_objects={'StochasticGradientLangevinDynamics': StochasticGradientLangevinDynamics})

layers = []
layers.append(tf.keras.layers.RandomFlip(seed=random_seed))
layers.append(tf.keras.layers.RandomRotation(0.2, seed=random_seed))
layers.append(tf.keras.layers.RandomTranslation(0.2, 0.2, seed=random_seed))
layers.append(tf.keras.layers.RandomZoom((-0.2, 0.0), seed=random_seed))
layers.append(tf.keras.layers.RandomBrightness(0.2, seed=random_seed))
layers.append(tf.keras.layers.RandomContrast(0.4, seed=random_seed))

augmentation = tf.keras.Sequential(layers)


y_pred = np.empty((len(y_val), 10, 4))
labs = np.concatenate([y for _, y in ds_val], axis=0)

for i in range(10):
    preds = []
    for batch in ds_val:
        X_batch, y_batch = batch
        images = augmentation(X_batch, training=True)
        preds.append(model(images, training=False))
    preds = np.concatenate(preds, axis=0)
    y_pred[:, i, :] = preds
        


acc = np.mean(np.argmax(preds, axis=1) == labs)
print(f"Accuracy: {acc}")

np.mean(y_pred.mean(axis=1).argmax(axis=1) == labs)


for batch in ds_train:
    X_batch, y_batch = batch
    images = augmentation(X_batch, training=True)
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(4):
        axs[0, i].imshow(X_batch[i] / 255.)
        axs[0, i].set_title(f"Original {i + 1}")
        axs[0, i].axis('off')
        axs[1, i].imshow(images[i] / 255.)
        axs[1, i].set_title(f"Augmented {i + 1}")
        axs[1, i].axis('off')
        axs[2, i].imshow(X_batch[i + 4] / 255.)
        axs[2, i].set_title(f"Original {i + 5}")
        axs[2, i].axis('off')
        axs[3, i].imshow(images[i + 4] / 255.)
        axs[3, i].set_title(f"Augmented {i + 5}")
        axs[3, i].axis('off')
    plt.show()
    break

import pandas as pd

np.save('y_pred_2.npy', y_pred)
np.save('labs_2.npy', labs)
np.save('x_val_2.npy', X_val)