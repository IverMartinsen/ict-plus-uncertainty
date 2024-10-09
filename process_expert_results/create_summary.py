import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from utils.utils import store_confusion_matrix

destination = "./results/expert_results/"

os.makedirs(destination, exist_ok=True)

df = pd.read_csv(os.path.join(destination, 'predictions.csv'))

experts = ['Tine', 'Marit-Solveig', 'Kasia', 'Morten', 'Steffen', 'Eirik '][:4] # exclude Eirik and Steffen

# ==============================
# SUMMARY STATISTICS
# ==============================
summary = pd.DataFrame(index=['accuracy'])

for expert in experts:
    summary[expert] = (df['label'] == df[expert]).mean()

summary['majority_vote'] = (df['label'] == df['pred_mode']).mean()
summary['weighted_vote'] = (df['label'] == df['pred_weighted']).mean()

# compute cohens kappa between the experts
kappa = np.zeros((4, 4))
for i, expert in enumerate(experts):
    for j, expert2 in enumerate(experts):
        if expert == expert2:
            continue
        kappa[i, j] = cohen_kappa_score(df[expert], df[expert2])
kappa = np.triu(kappa, k=1)
summary['kappa'] = kappa.sum() / np.count_nonzero(kappa)

# compute spearmans rank correlation between the experts
correlation = np.zeros((4, 4))
for i, expert1 in enumerate(experts):
    for j, expert2 in enumerate(experts):
        if expert1 == expert2:
            continue
        correlation[i, j] = spearmanr(df[expert1 + '_uncertainty'], df[expert2 + '_uncertainty'])[0]
correlation = np.triu(correlation, k=1)
summary['rank_correlation'] = correlation.sum() / np.count_nonzero(correlation)

summary.T.to_csv(os.path.join(destination, 'summary.csv'))

store_confusion_matrix(df['label'], df['pred_weighted'], destination)
