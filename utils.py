import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

lab_to_int = {'A': 0, 'B': 1, 'S': 2, 'P': 3}
int_to_lab = {v: k for k, v in lab_to_int.items()}
lab_to_long = {'A': 'Benthic agglutinated', 'B': 'Benthic calcareous', 'S': 'Sediment', 'P': 'Planktic'}

def map_fn(filename, label, image_size=[224, 224]):
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, image_size)
    return image, label

def load_data(path_to_data):
    X = glob.glob(path_to_data + '**/*.png', recursive=True)
    y = [os.path.basename(os.path.dirname(f)) for f in X]
    y = [lab_to_int[l] for l in y]
    return X, y

def make_dataset(X, y, image_size, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.map(lambda x, y: map_fn(x, y, image_size))
    if shuffle:
        ds = ds.shuffle(len(X))
    ds = ds.batch(batch_size)
    return ds

def compute_predictive_variance(Y_pred):
    """
    Compute the predictive variance
    """
    n = Y_pred.shape[0] # number of samples
    k = Y_pred.shape[-1] # number of classes
    # epistemic term
    ep_cov = np.matmul(Y_pred[:,:, :, np.newaxis], Y_pred[:, :, np.newaxis, :])
    ep_cov[:, :, np.arange(k), np.arange(k)] = (Y_pred * (1 - Y_pred))
    ep_cov = ep_cov.mean(axis=1)
    # aleatoric term
    al_cov = np.zeros((n, k, k))
    for i in range(n):
        al_cov[i] = np.cov(Y_pred[i].T)
    # total uncertainty
    tot_cov = ep_cov + al_cov
    return tot_cov

def plot_images(group, cols, plotname, destination, image_size=[224, 224]):
    """
    Plot the images from the group
    """
    _, axs = plt.subplots(cols, cols, figsize=(10, 10))

    for i, ax in enumerate(axs.flatten()):
        try:
            filename = group.iloc[i, 0]
            label = lab_to_long[int_to_lab[group['label'].iloc[i]]]
            pred = lab_to_long[int_to_lab[group['pred_mode'].iloc[i]]]
            ax.imshow(Image.open(filename).resize(image_size))
            ax.set_title(f'Label: {label}\nPred: {pred}', fontsize=6, fontweight='bold')
        except IndexError:
            pass
        ax.axis('off')
    plt.savefig(os.path.join(destination, plotname), bbox_inches='tight', dpi=300)
    plt.close()
