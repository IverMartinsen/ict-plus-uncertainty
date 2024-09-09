import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

destination = 'results/sgld_5e5_results/'
path_to_df = os.path.join(destination, 'predictions.csv')
path_to_expert = './results/expert_results/predictions.csv'
image_size = [224, 224]

df_expert = pd.read_csv(path_to_expert)
df_model = pd.read_csv(path_to_df)
df_model['filename'] = df_model['filename'].apply(lambda x: x.split('/')[-1])
df_model = df_model.set_index('filename').loc[df_expert['filename']].reset_index()

p = 0
# hypergeometric cdf
from scipy.stats import hypergeom

1 - hypergeom.cdf(6, 260, 32, 32)

key_pred = 'pred_mean'
key = 'conf_mean'

# compute kappa score
kappa = cohen_kappa_score(df_expert['pred_mode'], df_model[key_pred])
spearman_r, spearman_p = spearmanr(df_expert['mode_weights'], df_model[key], alternative='two-sided')
certain_intersection = np.intersect1d(
    df_expert['mode_weights'].sort_values().index[-133:], 
    df_model[key].sort_values().index[-133:]
    ).shape[0]
uncertain_intersection = np.intersect1d(
    df_expert['mode_weights'].sort_values().index[:32],
    df_model[key].sort_values().index[:32]
    ).shape[0]
mistakes_intersection = np.intersect1d(
    df_expert[df_expert['pred_mode'] != df_expert['label']].index,
    df_model[df_model[key_pred] != df_model['label']].index
    ).shape[0]
easy_mistakes_intersection = np.intersect1d(
    df_expert[(df_expert['mode_weights'] == 1.0)].index,
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
summary.to_csv(os.path.join(destination, 'expert_summary.csv'))