# Produce expert_summary.csv file
import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

parser = argparse.ArgumentParser()
parser.add_argument("--destination", type=str, default='./results/ensemble_results/')
parser.add_argument("--key", type=str, default='mean')
args = parser.parse_args()

df_expert = pd.read_csv('./results/expert_results/predictions.csv')
df_model = pd.read_csv(os.path.join(args.destination, 'predictions.csv'))
df_model['filename'] = df_model['filename'].apply(lambda x: x.split('/')[-1])
df_model = df_model.set_index('filename').loc[df_expert['filename']].reset_index()


key_pred, key_uncertainty = {'mean': ('pred_mean', 'conf_mean'), 'median': ('pred_median', 'conf_median')}[args.key]


# compute kappa score
kappa = cohen_kappa_score(df_expert['pred_weighted'], df_model[key_pred])
spearman_r, spearman_p = spearmanr(df_expert['weighted_confidence'], df_model[key_uncertainty], alternative='two-sided')
certain_intersection = np.intersect1d(
    df_expert['weighted_confidence'].sort_values().index[-133:], 
    df_model[key_uncertainty].sort_values().index[-133:]
    ).shape[0]
uncertain_intersection = np.intersect1d(
    df_expert['weighted_confidence'].sort_values().index[:32],
    df_model[key_uncertainty].sort_values().index[:32]
    ).shape[0]
mistakes_intersection = np.intersect1d(
    df_expert[df_expert['pred_weighted'] != df_expert['label']].index,
    df_model[df_model[key_pred] != df_model['label']].index
    ).shape[0]
easy_mistakes_intersection = np.intersect1d(
    df_expert[(df_expert['weighted_confidence'] == 1.0)].index,
    df_model[(df_model[key_pred] != df_model['label'])].index
).shape[0]

summary = {
    'kappa': kappa,
    'spearman_r': spearman_r,
    'spearman_p': spearman_p,
    'certain_intersection': certain_intersection,
    'uncertain_intersection': uncertain_intersection,
    'mistakes_intersection': mistakes_intersection,
    'easy_mistakes_intersection': easy_mistakes_intersection
}

summary = pd.DataFrame(summary, index=['values'])
summary = summary.T
summary.to_csv(os.path.join(args.destination, 'expert_summary.csv'))