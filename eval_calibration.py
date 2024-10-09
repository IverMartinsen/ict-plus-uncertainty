import os
import argparse
import numpy as np
import pandas as pd
from utils.calibration import make_calibration_plots, make_ordered_calibration_plot

parser = argparse.ArgumentParser()
parser.add_argument("--destination", type=str, default='./results/ensemble_results/')
parser.add_argument("--key", type=str, default='mean')
args = parser.parse_args()

if __name__ == '__main__':

    Y_pred = np.load(os.path.join(args.destination, 'predictions.npy'))

    df = pd.read_csv(os.path.join(args.destination, 'predictions.csv'))

    cal_error = []
    rel_error = []
    
    for b in [5, 10, 15, 20, 25]:
        if args.key == 'mean':
            x = np.mean(Y_pred, axis=1)
        elif args.key == 'median':
            x = np.median(Y_pred, axis=1)
            x /= x.sum(axis=1)[:, None]
        e = make_calibration_plots(x, df['label'], args.destination, num_bins=b)
        cal_error.append(e)
        e = make_ordered_calibration_plot(x, df['label'], args.destination, num_bins=b)
        rel_error.append(e)
        # also make plot across all models
        x = []
        y = []
        for i in range(Y_pred.shape[1]):
            x.append(Y_pred[:, i, :])
            y.append(df['label'])
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y)
        e = make_calibration_plots(x, y, args.destination, num_bins=b, fname=f'calibration_{b}_bins_all_models.png')
        e = make_ordered_calibration_plot(x, y, args.destination, num_bins=b, fname=f'reliability_{b}_bins_all_models.png')

    error_summary = pd.DataFrame({
        'calibration_error': np.mean(cal_error),
        'reliability_error': np.mean(rel_error),
    }, index=[args.key])
    
    error_summary.to_csv(os.path.join(args.destination, 'calibration_error.csv'))