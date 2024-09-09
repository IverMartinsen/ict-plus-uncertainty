import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.stats
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, cohen_kappa_score

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
    X.sort()
    y = [os.path.basename(os.path.dirname(f)) for f in X]
    y = [lab_to_int[l] for l in y]
    return X, y

def make_dataset(X, y, image_size, batch_size, seed, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.map(lambda x, y: map_fn(x, y, image_size))
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(len(X), seed=seed)
    ds = ds.batch(batch_size)
    return ds

def compute_predictive_variance(Y_pred):
    """
    Compute the predictive variance
    """
    n = Y_pred.shape[0] # number of samples
    k = Y_pred.shape[-1] # number of classes
    # epistemic term
    al_cov = -np.matmul(Y_pred[:,:, :, np.newaxis], Y_pred[:, :, np.newaxis, :])
    al_cov[:, :, np.arange(k), np.arange(k)] = (Y_pred * (1 - Y_pred))
    al_cov = al_cov.mean(axis=1)
    # aleatoric term
    ep_cov = np.zeros((n, k, k))
    for i in range(n):
        ep_cov[i] = np.cov(Y_pred[i].T)
    # total uncertainty
    return ep_cov, al_cov

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
            ax.set_title(f'Label: {label}\nPred: {pred}\nFile: {os.path.basename(filename)}', fontsize=6, fontweight='bold')
        except IndexError:
            pass
        ax.axis('off')
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(os.path.join(destination, plotname), bbox_inches='tight', dpi=300)
    plt.close()


def store_predictions(y_pred, y, filenames, destination, y_var=None, y_logits=None):

    n, num_samples, _ = y_pred.shape
    df = pd.DataFrame()
    df['filename'] = filenames
    df['label'] = y
    for i in range(num_samples):
        df[f'model_{i}'] = y_pred[:, i, :].argmax(axis=1)
    df['agree'] = np.prod(df.iloc[:, 2:] == df.iloc[:, 2].values[:, None], axis=1).astype(bool) # check if all models agree
    keys = [f'model_{i}' for i in range(num_samples)]
    df['pred_mode'] = df[keys].mode(axis=1)[0] # majority vote
    df['percentage_agree'] = df[keys].apply(lambda x: x.value_counts().max() / x.value_counts().sum(), axis=1)
    df['pred_mean'] = y_pred.mean(axis=1).argmax(axis=1) # mean prediction
    df['loss'] = -np.log(y_pred.mean(axis=1)[np.arange(len(y)), y])
    df['conf_mean'] = y_pred.mean(axis=1).max(axis=1) # mean confidence
    conf_median = np.median(y_pred, axis=1)
    conf_median /= conf_median.sum(axis=1)[:, None]
    df['conf_median'] = conf_median.max(axis=1)
    df['pred_median'] = conf_median.argmax(axis=1)
    conf_min = np.min(y_pred, axis=1)
    conf_min = conf_min[np.arange(len(y)), df['pred_mean']]
    df['conf_min'] = conf_min
    df['var_mean'] = (df['conf_mean'] * (1 - df['conf_mean'])) # mean variance
    iqr = scipy.stats.iqr(y_pred, axis=1)
    iqr = iqr[np.arange(len(y)), df['pred_mean']]
    df['iqr'] = iqr
    var = np.var(y_pred, axis=1)
    var = var[np.arange(len(y)), df['pred_mean']]
    df['var'] = var
    ep_cov, al_cov = compute_predictive_variance(y_pred)
    cov = ep_cov + al_cov
    cov = cov[:, np.arange(4), np.arange(4)]
    df['predictive_variance'] = cov[np.arange(n), np.array(df['pred_mean']).flatten()]
    #eigvals = np.sort(np.linalg.eigvals(cov), axis=1)[:, ::-1]
    df['total_variance'] = cov.sum(axis=1)
    df['epistemic_variance'] = ep_cov[:, np.arange(4), np.arange(4)].sum(axis=1)
    df['aleatoric_variance'] = al_cov[:, np.arange(4), np.arange(4)].sum(axis=1)
    if y_var is not None:
        df['modeled_variance'] = y_var
    if y_logits is not None:
        df['logits'] = y_logits[np.arange(len(y)), 0, df['pred_mean']]
    else:
        y_logits = np.log(y_pred)
        df['log_var'] = y_logits.var(axis=1)[np.arange(len(y)), df['pred_mean']]
    df.to_csv(os.path.join(destination, 'predictions.csv'), index=False)
    
    return df

def store_confusion_matrix(labels, predictions, destination):
    confusion = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion, display_labels=list(lab_to_long.values()))
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    plt.savefig(os.path.join(destination, 'confusion_matrix.png'), bbox_inches='tight', dpi=300)


def plot_uncertainty(x, loss, c, x_label, title, filename, destination, q_val):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, loss, c=c)
    ax.set_ylabel('Loss')
    ax.set_xlabel(x_label)
    ax.axvline(x=np.quantile(x, 0.50), color='black', linestyle='--')
    ax.axvline(x=np.quantile(x, q_val), color='black', linestyle='--')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(['correct', '0.50 quantile', f'{q_val} quantile'])
    plt.savefig(os.path.join(destination, filename), bbox_inches='tight', dpi=300)
