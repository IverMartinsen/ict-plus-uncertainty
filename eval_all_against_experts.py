import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

parser = argparse.ArgumentParser()
parser.add_argument("--destination", type=str, default='./results/sgld_5e5_results/')
parser.add_argument("--key", type=str, default='conf')
args = parser.parse_args()

Y_pred = np.load(os.path.join(args.destination, 'predictions.npy'))
filenames = np.load(os.path.join(args.destination, 'filenames.npy'))
labels = np.load(os.path.join(args.destination, 'labels.npy'))

path_to_df = os.path.join(args.destination, 'predictions.csv')
path_to_expert = './results/expert_results/predictions.csv'
image_size = [224, 224]

df_expert = pd.read_csv(path_to_expert)

num_models = Y_pred.shape[1]

summary = {
    'kappa': [],
    'spearman_r': [],
    'spearman_p': [],
    'certain_intersection': [],
    'uncertain_intersection': [],
    'mistakes_intersection': [],
    'easy_mistakes_intersection': []
}


for i in range(num_models):
    df_model = pd.DataFrame({
        'filename': filenames,
        'label': labels,
        'y_pred': Y_pred[:, i].argmax(axis=1),
        'conf': Y_pred[:, i].max(axis=1)
    })

    df_model['filename'] = df_model['filename'].apply(lambda x: x.split('/')[-1])
    df_model = df_model.set_index('filename').loc[df_expert['filename']].reset_index()

    key_pred_expert = 'pred_weighted'
    key_uncertainty_expert = 'weighted_confidence'

    # compute kappa score
    kappa = cohen_kappa_score(df_expert[key_pred_expert], df_model['y_pred'])
    spearman_r, spearman_p = spearmanr(df_expert[key_uncertainty_expert], df_model[args.key], alternative='two-sided')
    certain_intersection = np.intersect1d(
        df_expert[key_uncertainty_expert].sort_values().index[-133:], 
        df_model[args.key].sort_values().index[-133:]
        ).shape[0]
    uncertain_intersection = np.intersect1d(
        df_expert[key_uncertainty_expert].sort_values().index[:32],
        df_model[args.key].sort_values().index[:32]
        ).shape[0]
    mistakes_intersection = np.intersect1d(
        df_expert[df_expert[key_pred_expert] != df_expert['label']].index,
        df_model[df_model['y_pred'] != df_model['label']].index
        ).shape[0]
    easy_mistakes_intersection = np.intersect1d(
        df_expert[(df_expert[key_uncertainty_expert] == 1.0)].index,
        df_model[(df_model['y_pred'] != df_model['label'])].index
    ).shape[0]

    summary['kappa'].append(kappa)
    summary['spearman_r'].append(spearman_r)
    summary['spearman_p'].append(spearman_p)
    summary['certain_intersection'].append(certain_intersection)
    summary['uncertain_intersection'].append(uncertain_intersection)
    summary['mistakes_intersection'].append(mistakes_intersection)
    summary['easy_mistakes_intersection'].append(easy_mistakes_intersection)

for key in summary.keys():
    summary[key] = np.mean(summary[key])

summary = pd.DataFrame(summary, index=['values'])
summary = summary.T
summary.to_csv(os.path.join(args.destination, 'expert_summary_mean_of_all_models.csv'))