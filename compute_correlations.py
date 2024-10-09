import os
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--path1", type=str, default='./results/ensemble_results')
parser.add_argument("--path2", type=str, default='./results/tta_results_s1')
args = parser.parse_args()

path1 = args.path1
path2 = args.path2

df1 = pd.read_csv(os.path.join(path1, 'predictions.csv'))
df2 = pd.read_csv(os.path.join(path2, 'predictions.csv'))

df1['filename'] = df1['filename'].apply(lambda x: os.path.basename(x))
df2['filename'] = df2['filename'].apply(lambda x: os.path.basename(x))

f1 = df1['filename']
df2 = df2.set_index('filename').loc[f1].reset_index()

assert (df1['filename'] == df2['filename']).all()

y1 = df1['pred_mean']
y2 = df2['pred_mean']

kappa = cohen_kappa_score(y1, y2)
print(f"Cohen's Kappa: {kappa}")

p1 = df1['conf_median']
p2 = df2['conf_mean']

spearman = spearmanr(p1, p2)
print(f"Spearman's Rho: {spearman.correlation}")

# print filenames where mistakes are the same
print("Same mistakes:")
print(df1[(y1 != df1['label']) & (y2 != df2['label'])]['filename'].values)
