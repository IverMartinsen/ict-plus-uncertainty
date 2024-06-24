import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


destination = 'stats/ensemble_stats'
y_pred = np.load(os.path.join(destination, 'predictions.npy'))
df = pd.read_csv(os.path.join(destination, 'predictions.csv'))

n, m, k = y_pred.shape

num_bins = 20

y_pred_ = df['pred_mean'].values
y_true = df['label'].values
y_conf = df['total_variance'].values
y_conf = np.log(y_conf + 1e-8)

sorted_indices = np.argsort(y_conf)

y_pred_sorted = y_pred_[sorted_indices]
y_conf_sorted = y_conf[sorted_indices]
y_true_sorted = y_true[sorted_indices]

samples_per_bin = n // num_bins
upper = y_conf_sorted[samples_per_bin::samples_per_bin]
if len(upper) < num_bins:
    upper = np.append(upper, [np.inf])
else:
    upper[-1] = np.inf

lower = np.array([0] + list(upper[:-1]))

assert len(upper) == num_bins

acc = np.zeros(num_bins)
conf = np.zeros(num_bins)
obs = np.zeros(num_bins)

for i in range(num_bins):
    #mask = (y_conf_sorted >= lower[i]) & (y_conf_sorted <= upper[i])
    mask = (y_conf_sorted > lower[i]) 
    obs[i] = np.sum(mask)
    acc[i] = np.mean(y_pred_sorted[mask] == y_true_sorted[mask])
    #conf[i] = np.mean(y_conf_sorted[mask])
    conf[i] = lower[i]

plt.figure(figsize=(10, 5))
plt.plot(conf, acc, marker='o')
plt.xlabel('Uncertainty')
plt.ylabel('Accuracy')
plt.ylim([0.70, 1.03])
plt.title(f'Total Uncertainty Calibration Plot ({num_bins} bins)')
plt.grid()
plt.savefig(os.path.join(destination, 'total_uncertainty_calibration_plot.png'), bbox_inches='tight', dpi=300)
plt.close()
