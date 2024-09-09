import os
import numpy as np
import matplotlib.pyplot as plt

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
            if idx[0].shape[0] == 0:
                print(f"Bin {i} for class {k} is empty")
                continue
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

    nonzeros = np.count_nonzero(o_all, axis=0)
    p = np.array(p_all).sum(axis=0) / nonzeros
    f = np.array(f_all).sum(axis=0) / nonzeros
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
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.bar((low + upp) / 2, freqs, width=step*0.95, color="b")
    plt.step(bins, np.concatenate([[0], preds]), where="pre", color="k", linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration error: {error:.3f}")
    plt.savefig(os.path.join(destination, f"calibration_{num_bins}_bins.png"), dpi=300)
    plt.close()

    #plt.figure(figsize=(10, 5))
    #plt.bar((low + upp) / 2, freqs, width=step*0.95, color="b")
    ##plt.plot(preds, preds - freqs)
    ##plt.scatter(preds, preds - freqs)
    #plt.xlabel("Predicted probability")
    #plt.ylabel("Confidence error")
    #plt.title(f"Calibration error: {error:.3f}")
    #plt.savefig(os.path.join(destination, "calibration2.png"), dpi=300)
    return error

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
    plt.plot(np.linspace(0, 1, num_bins), conf - acc, marker='o')
    plt.xticks(np.linspace(0, 1, num_bins), conf.round(2), rotation=45)
    plt.ylim(-0.3, 0.3)
    plt.xlabel('Confidence')
    plt.ylabel('Confidence - Accuracy')
    plt.title(f'Reliability Diagram\nExpected Calibration Error: {error:.4f}')
    plt.grid()
    plt.savefig(os.path.join(destination, f'reliability_plot_{num_bins}_bins.png'), bbox_inches='tight', dpi=300)
    plt.close()

    return error