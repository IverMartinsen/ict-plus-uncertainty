# Description: Produces the following files:
# - predictions.csv
# - confusion_matrix.png
# - summary.csv
# - class_wise_accuracy.csv

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score
from utils.utils import (
    store_predictions, 
    store_confusion_matrix, 
    lab_to_long
)

parser = argparse.ArgumentParser()
parser.add_argument("--destination", type=str, default='./results/ensemble_results/')
parser.add_argument("--key", type=str, default='conf_mean')
args = parser.parse_args()


key_pred, key_uncertainty = {'mean': ('pred_mean', 'conf_mean'), 'median': ('pred_median', 'conf_median')}[args.key]


Y_pred = np.load(os.path.join(args.destination, 'predictions.npy'))
filenames = np.load(os.path.join(args.destination, 'filenames.npy'))
labels = np.load(os.path.join(args.destination, 'labels.npy'))
try:
    Y_var = np.load(os.path.join(args.destination, 'variances.npy'))
except FileNotFoundError:
    Y_var = None
try:
    Y_logit = np.load(os.path.join(args.destination, 'logits.npy'))
except FileNotFoundError:
    Y_logit = None


def store_summary_stats(df, destination, num_models=10, fname='summary.csv'):

    # total accuracy
    summary = pd.DataFrame(index=['accuracy'])
    for i in range(num_models):
        summary[f'model_{i}'] = (df['label'] == df[f'model_{i}']).mean()
    summary['average_accuracy'] = summary.mean(axis=1)

    # compute cohens kappa between the models
    kappa = np.zeros((num_models, num_models))
    for i in range(num_models):
        for j in range(i+1, num_models):
            kappa[i, j] = cohen_kappa_score(df[f'model_{i}'], df[f'model_{j}'])
    # upper triangular matrix
    kappa = np.triu(kappa, k=1)

    summary['cohens_kappa'] = kappa.sum() / np.count_nonzero(kappa)

    summary['acc_by_majority_vote'] = (df['label'] == df['pred_mode']).mean()
    summary['acc_by_mean_vote'] = (df['label'] == df['pred_mean']).mean()
    summary['acc_by_median_vote'] = (df['label'] == df['pred_median']).mean()
    summary['loss_by_mean'] = df['loss_mean'].mean()
    summary['loss_by_median'] = df['loss_median'].mean()
    
    x = df[key_uncertainty]
    z = df['label'] == df[key_pred]
    summary['mean_confidence'] = np.mean(x)
    summary['mean_confidence_correct'] = np.mean(x[z])
    summary['mean_confidence_incorrect'] = np.mean(x[~z])
    summary['std_confidence'] = np.std(x)
    summary['std_confidence_correct'] = np.std(x[z])
    summary['std_confidence_incorrect'] = np.std(x[~z])
    summary['100_recall_confidence'] = np.max(x[~z])
    summary['100_recall_sum'] = np.sum(x < np.max(x[~z]))
    summary['50_recall_confidence'] = np.median(x[~z])
    summary['50_recall_sum'] = np.sum(x < np.median(x[~z]))
    # num mistakes in total
    summary['tot_num_mistakes'] = np.sum(~z)
    # 32 most uncertain predictions
    idx = x.sort_values().index[:32]
    summary['32_most_uncertain_recall'] = (df['label'] != df[key_pred]).loc[idx].sum()
    # eval 133 most certain predictions
    summary['133_most_certain_accuracy'] = (df['label'] == df[key_pred]).loc[df[key_uncertainty].sort_values().index[-133:]].sum()
    
    summary = summary.T
    summary.to_csv(os.path.join(destination, fname))

if __name__ == '__main__':

    df = store_predictions(Y_pred, labels, filenames, args.destination, Y_var, Y_logit)

    store_confusion_matrix(df['label'], df[key_pred], args.destination)

    store_summary_stats(df, args.destination, num_models=Y_pred.shape[1])

    class_wise_accuracy = classification_report(df['label'], df[key_pred], target_names=list(lab_to_long.values()), output_dict=True)
    class_wise_df = pd.DataFrame(class_wise_accuracy).T
    class_wise_df.to_csv(os.path.join(args.destination, 'class_wise_accuracy.csv'))
