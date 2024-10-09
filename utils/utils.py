import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.stats
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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


def store_predictions(y_pred, y, filenames, destination, y_var=None, y_logits=None):

    n, num_samples, _ = y_pred.shape
    df = pd.DataFrame()
    df['filename'] = filenames
    df['label'] = y
    for i in range(num_samples):
        df[f'model_{i}'] = y_pred[:, i, :].argmax(axis=1)

    keys = [f'model_{i}' for i in range(num_samples)]
    var = np.var(y_pred, axis=1)
    iqr = scipy.stats.iqr(y_pred, axis=1)
    ep_cov, al_cov = compute_predictive_variance(y_pred)
    cov = ep_cov + al_cov
    cov = cov[:, np.arange(4), np.arange(4)]

    # Percentage agreement
    df['agree'] = np.prod(df.iloc[:, 2:] == df.iloc[:, 2].values[:, None], axis=1).astype(bool) # check if all models agree
    df['percentage_agree'] = df[keys].apply(lambda x: x.value_counts().max() / x.value_counts().sum(), axis=1)
    # Mean prediction
    df['pred_mean'] = y_pred.mean(axis=1).argmax(axis=1) # mean prediction
    df['loss_mean'] = -np.log(y_pred.mean(axis=1)[np.arange(len(y)), y])
    df['conf_mean'] = y_pred.mean(axis=1).max(axis=1) # mean confidence
    df['var_mean'] = var[np.arange(len(y)), df['pred_mean']]
    df['iqr_mean'] = iqr[np.arange(len(y)), df['pred_mean']]
    df['post_pred_var_mean'] = cov[np.arange(n), np.array(df['pred_mean']).flatten()]
    df['entropy_mean'] = -np.sum(y_pred.mean(axis=1) * np.log(y_pred.mean(axis=1)), axis=1)
    # Mode prediction
    df['pred_mode'] = df[keys].mode(axis=1)[0] # majority vote
    # Median prediction
    conf_median = np.median(y_pred, axis=1)
    conf_median /= conf_median.sum(axis=1)[:, None]
    df['pred_median'] = conf_median.argmax(axis=1)
    df['loss_median'] = -np.log(conf_median[np.arange(len(y)), y])
    df['conf_median'] = conf_median.max(axis=1)
    df['var_median'] = var[np.arange(len(y)), df['pred_median']]
    df['iqr_median'] = iqr[np.arange(len(y)), df['pred_median']]
    df['post_pred_var_median'] = cov[np.arange(n), np.array(df['pred_median']).flatten()]
    df['entropy_median'] = -np.sum(conf_median * np.log(conf_median), axis=1)
    # Epistemic uncertainty
    df['tot_epi_var'] = ep_cov[:, np.arange(4), np.arange(4)].sum(axis=1)
    df['tot_ale_var'] = al_cov[:, np.arange(4), np.arange(4)].sum(axis=1)
    df['tot_var'] = cov.sum(axis=1)
    if y_var is not None:
        df['modeled_var'] = y_var
    if y_logits is not None:
        df['logits'] = y_logits[np.arange(len(y)), 0, df['pred_mean']]
    else:
        y_logits = np.log(y_pred)
        df['log_var_mean'] = y_logits.var(axis=1)[np.arange(len(y)), df['pred_mean']]
    
        weighted_preds = np.zeros(len(df['label']))
    df['unique_preds'] = np.unique(y_pred.argmax(axis=2), axis=1).shape[1]
    for i in range(y_pred.shape[1]):
        pred_i = y_pred[:, i, :].argmax(axis=1)
        idx = np.where(pred_i == df['pred_mean'])[0]
        weighted_preds[idx] += y_pred[idx, i, :].max(axis=1)
        for j in range(2, 10):
            idx = np.where((pred_i != df['pred_mean']) & (df['unique_preds'] == j))[0]
            weighted_preds[idx] -= y_pred[idx, i, :].max(axis=1) / (j - 1)

    df['weighted_confidence'] = weighted_preds / y_pred.shape[1]
    
    df.to_csv(os.path.join(destination, 'predictions.csv'), index=False)
    
    return df

def store_confusion_matrix(labels, predictions, destination):
    confusion = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion, display_labels=list(lab_to_long.values()))
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    plt.savefig(os.path.join(destination, 'confusion_matrix.png'), bbox_inches='tight', dpi=300)
    plt.close()
