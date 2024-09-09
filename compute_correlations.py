import os
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

path1 = "./stats/dropout_stats"
path2 = "./results/swag_5e5_results"

df1 = pd.read_csv(os.path.join(path1, 'predictions.csv'))
df2 = pd.read_csv(os.path.join(path2, 'predictions.csv'))


# replace // with /
df1['filename'] = df1['filename'].apply(lambda x: os.path.basename(x))
df2['filename'] = df2['filename'].apply(lambda x: os.path.basename(x))

f1 = df1['filename']
df2 = df2.set_index('filename').loc[f1].reset_index()

assert (df1['filename'] == df2['filename']).all()

y1 = df1['pred_mean']
y2 = df2['pred_mean']

kappa = cohen_kappa_score(y1, y2)
print(f"Cohen's Kappa: {kappa}")

p1 = df1['conf_mean']
p2 = df2['conf_mean']

spearman = spearmanr(p1, p2)
print(f"Spearman's Rho: {spearman.correlation}")
