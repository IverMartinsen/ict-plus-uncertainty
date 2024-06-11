import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

path1 = "/Users/ima029/Desktop/IKT+ Uncertainty/Repository/tta_stats/predictions.csv"
path2 = "/Users/ima029/Desktop/IKT+ Uncertainty/Repository/confidence_stats/predictions.csv"

df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)


# replace // with /
df1['filename'] = df1['filename'].apply(lambda x: x.replace('//', '/'))


f1 = df1['filename']
df2 = df2.set_index('filename').loc[f1].reset_index()

assert (df1['filename'] == df2['filename']).all()


y1 = df1['pred_mean']
y2 = df2['pred']

kappa = cohen_kappa_score(y1, y2)

print(f"Cohen's Kappa: {kappa}")

p1 = df1['var_mean']
p2 = df2['variance']

spearman = spearmanr(p1, p2)

print(f"Spearman's Rho: {spearman.correlation}")
