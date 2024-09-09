import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score
from utils import (
    store_predictions, 
    store_confusion_matrix, 
    lab_to_long
)

parser = argparse.ArgumentParser()
parser.add_argument("--destination", type=str, default='./results/sgld_5e5_results/')
parser.add_argument("--key", type=str, default='conf_mean')
parser.add_argument("--key_pred", type=str, default='pred_mean')
parser.add_argument("--key_uncertainty", type=str, default='conf_mean')
args = parser.parse_args()

if __name__ == '__main__':

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

    df = store_predictions(Y_pred, labels, filenames, args.destination, Y_var, Y_logit)

    store_confusion_matrix(df['label'], df[args.key_pred], args.destination)

    def store_summary_stats(df, destination, num_models=10):

        # total accuracy
        summary = pd.DataFrame(index=['accuracy'])
        for i in range(num_models):
            summary[f'model_{i}'] = (df['label'] == df[f'model_{i}']).mean()
        summary['average'] = summary.mean(axis=1)
        summary['majority_vote'] = (df['label'] == df['pred_mode']).mean()
        summary['mean_vote'] = (df['label'] == df['pred_mean']).mean()
        summary['median_vote'] = (df['label'] == df['pred_median']).mean()

        # compute cohens kappa between the models
        kappa = np.zeros((num_models, num_models))
        for i in range(num_models):
            for j in range(i+1, num_models):
                kappa[i, j] = cohen_kappa_score(df[f'model_{i}'], df[f'model_{j}'])
        # upper triangular matrix
        kappa = np.triu(kappa, k=1)

        summary['kappa'] = kappa.sum() / np.count_nonzero(kappa)
        summary['loss'] = df['loss'].mean()
        
        x = df[args.key_uncertainty]
        z = df['label'] == df[args.key_pred]
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
        # eval 133 most certain predictions
        acc = (df['label'] == df[args.key_pred]).loc[df[args.key_uncertainty].sort_values().index[-133:]].mean()
        summary['133_most_certain'] = acc
        
        summary = summary.T
        summary.to_csv(os.path.join(destination, 'summary.csv'))

    store_summary_stats(df, args.destination, num_models=Y_pred.shape[1])

    class_wise_accuracy = classification_report(df['label'], df[args.key_pred], target_names=list(lab_to_long.values()), output_dict=True)
    class_wise_df = pd.DataFrame(class_wise_accuracy).T
    class_wise_df.to_csv(os.path.join(args.destination, 'class_wise_accuracy.csv'))
