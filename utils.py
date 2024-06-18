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

def compute_calibration_stats(y_pred, y_true, num_bins=9):
    """
    Compute calibration statistics for a set of predictions.
    Args:
        y_pred: array of shape (n, m) where n is the number of samples and m is the number of classes
        num_bins: number of bins for the calibration plot
    Returns:
        Dictionary with calibration statistics
    """
    n, m = y_pred.shape
    
    bins = np.linspace(0, 1, num_bins + 1)
    low = bins[:-1]
    upp = bins[1:]
    
    p_all = [] # average confidence per bin
    f_all = [] # frequency of positive class per bin
    o_all = [] # number of observations per bin
    error = 0  # average calibration error

    for k in range(m): # one vs all for each class

        # one vs all
        y_prob = y_pred[:, k]
        y_true_ = np.array([1 if y == k else 0 for y in y_true])

        p = np.zeros(len(low))
        freqs = np.zeros(len(low))
        observed = np.zeros(len(low))

        for i in range(len(low)):
            idx = np.where((y_prob >= low[i]) * (y_prob < upp[i]))
            assert idx[0].shape[0] > 0, f"Bin {i} for class {k} is empty"
            # average confidence for bin i
            p[i] = y_prob[idx].mean()
            # frequency of positive class in bin i
            freqs[i] = y_true_[idx].mean()
            # total number of observations in bin i
            observed[i] = len(idx[0])
        
        error += ((np.abs(freqs - p)) * observed / n).sum()

        p_all.append(p)
        f_all.append(freqs)
        o_all.append(observed)

    p = np.array(p_all).mean(axis=0)
    f = np.array(f_all).mean(axis=0)
    o = np.array(o_all).sum(axis=0)

    return {
        "bins": bins,
        "low": low,
        "upp": upp,
        "preds_per_bin": p,
        "freqs_per_bin": f,
        "ssize_per_bin": o,
        "total_error": error
    }

def make_calibration_plots(y_pred, y_true, destination, num_bins=9):
    """
    Make calibration plots for a set of predictions. Save the plots to the destination folder.
    Args:
        y_pred: array of shape (n, m) where n is the number of samples and m is the number of classes
        y_true: array of shape (n,) where n is the number of samples
        destination: path to save the plots
        num_bins: number of bins for the calibration plot
    """
    calibration_stats = compute_calibration_stats(y_pred, y_true, num_bins=num_bins)

    bins = calibration_stats['bins']
    low = calibration_stats['low']
    upp = calibration_stats['upp']
    preds = calibration_stats['preds_per_bin']
    freqs = calibration_stats['freqs_per_bin']
    obs = calibration_stats['ssize_per_bin']
    error = calibration_stats['total_error']

    step = bins[1] - bins[0]

    plt.figure(figsize=(10, 5))
    plt.bar((low + upp) / 2, obs, width=step*0.95, color="b")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observations per bin")
    plt.savefig(os.path.join(destination, "observations.png"), dpi=300)
    
    plt.figure(figsize=(10, 5))
    plt.bar((low + upp) / 2, freqs, width=step*0.95, color="b")
    plt.step(bins, np.concatenate([[0], preds]), where="pre", color="k", linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration error: {error:.3f}")
    plt.savefig(os.path.join(destination, "calibration1.png"), dpi=300)

    #plt.figure(figsize=(10, 5))
    #plt.bar((low + upp) / 2, freqs, width=step*0.95, color="b")
    ##plt.plot(preds, preds - freqs)
    ##plt.scatter(preds, preds - freqs)
    #plt.xlabel("Predicted probability")
    #plt.ylabel("Confidence error")
    #plt.title(f"Calibration error: {error:.3f}")
    #plt.savefig(os.path.join(destination, "calibration2.png"), dpi=300)

def make_ordered_calibration_plot(y_pred, y_true, destination, num_bins=20):
    """
    Make a calibration plot with the bins ordered by frequency
    """
    n, _ = y_pred.shape

    y_pred_ = np.argmax(y_pred, axis=1)
    y_conf = np.max(y_pred, axis=1)
    y_true = np.array(y_true)

    sorted_indices = np.argsort(y_conf)

    y_pred_sorted = y_pred_[sorted_indices]
    y_conf_sorted = y_conf[sorted_indices]
    y_true_sorted = y_true[sorted_indices]

    samples_per_bin = n // num_bins
    upper = y_conf_sorted[samples_per_bin::samples_per_bin]
    upper[-1] = 1.0
    lower = np.array([0.0] + list(upper[:-1]))

    num_bins = len(upper)
    
    acc = np.zeros(num_bins)
    conf = np.zeros(num_bins)
    obs = np.zeros(num_bins)

    for i in range(num_bins):
        mask = (y_conf_sorted >= lower[i]) & (y_conf_sorted <= upper[i])

        obs[i] = np.sum(mask)
        acc[i] = np.mean(y_pred_sorted[mask] == y_true_sorted[mask])
        conf[i] = np.mean(y_conf_sorted[mask])

    error = (np.abs(conf - acc) * obs / n).sum()

    plt.figure(figsize=(10, 5))
    plt.plot(conf, conf - acc, marker='o')
    plt.xlabel('Confidence')
    plt.ylabel('Confidence - Accuracy')
    plt.title(f'Reliability Diagram\nExpected Calibration Error: {error:.4f}')
    plt.grid()
    plt.savefig(os.path.join(destination, 'calibration_plot_ordered.png'), bbox_inches='tight', dpi=300)
    plt.close()

def store_predictions(y_pred, y, filenames, destination):

    n, num_samples, _ = y_pred.shape
    df = pd.DataFrame()
    df['filename'] = filenames
    df['label'] = y
    for i in range(num_samples):
        df[f'model_{i}'] = y_pred[:, i, :].argmax(axis=1)
    df['agree'] = np.prod(df.iloc[:, 2:] == df.iloc[:, 2].values[:, None], axis=1).astype(bool) # check if all models agree
    df['pred_mode'] = df.iloc[:, 2:].mode(axis=1)[0] # majority vote
    df['percentage_agree'] = df.iloc[:, 2:12].apply(lambda x: x.value_counts().max() / x.value_counts().sum(), axis=1)
    df['pred_mean'] = y_pred.mean(axis=1).argmax(axis=1) # mean prediction
    df['loss'] = -np.log(y_pred.mean(axis=1)[np.arange(len(y)), y])
    df['conf_mean'] = y_pred.mean(axis=1).max(axis=1) # mean confidence
    df['var_mean'] = (df['conf_mean'] * (1 - df['conf_mean'])) # mean variance
    iqr = scipy.stats.iqr(y_pred, axis=1)
    iqr = iqr[np.arange(len(y)), df['pred_mean']]
    df['iqr'] = iqr
    ep_cov, al_cov = compute_predictive_variance(y_pred)
    cov = ep_cov + al_cov
    cov = cov[:, np.arange(4), np.arange(4)]
    df['predictive_variance'] = cov[np.arange(n), np.array(df['pred_mean']).flatten()]
    #eigvals = np.sort(np.linalg.eigvals(cov), axis=1)[:, ::-1]
    df['total_variance'] = cov.sum(axis=1)
    df['epistemic_variance'] = ep_cov[:, np.arange(4), np.arange(4)].sum(axis=1)
    df['aleatoric_variance'] = al_cov[:, np.arange(4), np.arange(4)].sum(axis=1)
    df.to_csv(os.path.join(destination, 'predictions.csv'), index=False)
    
    return df

def store_confusion_matrix(labels, predictions, destination):
    confusion = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion, display_labels=list(lab_to_long.values()))
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    plt.savefig(os.path.join(destination, 'confusion_matrix.png'), bbox_inches='tight', dpi=300)

def store_summary_stats(df, destination, num_models=10):

    # total accuracy
    summary = pd.DataFrame(index=['accuracy'])
    for i in range(num_models):
        summary[f'model_{i}'] = (df['label'] == df[f'model_{i}']).mean()
    summary['majority_vote'] = (df['label'] == df['pred_mode']).mean()
    summary['mean_vote'] = (df['label'] == df['pred_mean']).mean()

    # compute cohens kappa between the models
    kappa = np.zeros((num_models, num_models))
    for i in range(num_models):
        for j in range(i+1, num_models):
            kappa[i, j] = cohen_kappa_score(df[f'model_{i}'], df[f'model_{j}'])
    # upper triangular matrix
    kappa = np.triu(kappa, k=1)

    summary['kappa'] = kappa.sum() / np.count_nonzero(kappa)
    summary['loss'] = df['loss'].mean()
    summary = summary.T
    summary.to_csv(os.path.join(destination, 'summary.csv'))

def plot_uncertainty(x, loss, c, x_label, title, filename, destination):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, loss, c=c)
    ax.set_ylabel('Loss')
    ax.set_xlabel(x_label)
    ax.axvline(x=np.quantile(x, 0.50), color='black', linestyle='--')
    ax.axvline(x=np.quantile(x, 0.75), color='black', linestyle='--')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(['correct', '0.50 quantile', '0.75 quantile'])
    plt.savefig(os.path.join(destination, filename), bbox_inches='tight', dpi=300)
