import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from utils import (
    store_predictions, 
    store_confusion_matrix, 
    store_summary_stats, 
    lab_to_long,
    make_calibration_plots,
    make_ordered_calibration_plot
)

parser = argparse.ArgumentParser()
parser.add_argument("--destination", type=str, default='./stats/ensemble_stats/')
args = parser.parse_args()

if __name__ == '__main__':

    Y_pred = np.load(os.path.join(args.destination, 'predictions.npy'))
    filenames = np.load(os.path.join(args.destination, 'filenames.npy'))
    labels = np.load(os.path.join(args.destination, 'labels.npy'))

    df = store_predictions(Y_pred, labels, filenames, args.destination)

    store_confusion_matrix(df['label'], df['pred_mean'], args.destination)

    store_summary_stats(df, args.destination)

    class_wise_accuracy = classification_report(df['label'], df['pred_mean'], target_names=list(lab_to_long.values()), output_dict=True)
    class_wise_df = pd.DataFrame(class_wise_accuracy).T
    class_wise_df.to_csv(os.path.join(args.destination, 'class_wise_accuracy.csv'))

    make_calibration_plots(Y_pred.mean(axis=1), df['label'], args.destination, num_bins=8)

    make_ordered_calibration_plot(Y_pred.mean(axis=1), df['label'], args.destination, num_bins=20)
