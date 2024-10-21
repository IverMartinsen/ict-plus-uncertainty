import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import int_to_lab, lab_to_long

destination = "./results/expert_results/"

os.makedirs(destination, exist_ok=True)

df = pd.read_csv(os.path.join(destination, 'predictions.csv'))

experts = ['Tine', 'Marit-Solveig', 'Kasia', 'Morten', 'Steffen', 'Eirik '][:4] # exclude Eirik and Steffen

# ==============================
# SCATTER PLOT OF UNCERTAINTY
# ==============================

for i, expert1 in enumerate(experts):
    for j, expert2 in enumerate(experts):
        if expert1 == expert2:
            continue
        plt.figure(figsize=(10, 5))
        
        x = df[expert1 + '_uncertainty'] + np.random.normal(0, 1, len(df))
        y = df[expert2 + '_uncertainty'] + np.random.normal(0, 1, len(df))
        plt.scatter(x, y, s=10)
        plt.xlabel(f'{expert1} uncertainty')
        plt.ylabel(f'{expert2} uncertainty')
        plt.title(f'{expert1} vs {expert2}')
        plt.savefig(os.path.join(destination, f'{expert1}_vs_{expert2}.pdf'), dpi=300)
        plt.close()

# ==============================
# PLOT NUMBER OF MISTAKES VS UNCERTAINTY
# ==============================

markers = ['o', 'x', 's', '>']
x = df['num_mistakes'] + np.random.normal(0, 0.1, len(df))
y = df['mean_confidence'] + np.random.normal(0, 0.01, len(df))

plt.figure(figsize=(10, 5))
for i, label in enumerate(df['label'].unique()):
    idx = df['label'] == label
    plt.scatter(x[idx], y[idx], s=15, label=lab_to_long[int_to_lab[label]], marker=markers[i])
plt.xlabel('Number of mistakes')
plt.ylabel('Mean confidence')
plt.legend()
plt.savefig(os.path.join(destination, 'num_mistakes_vs_mean_confidence.pdf'), dpi=300)
plt.close()

markers = ['o', 'x', 's', '>']
x = df['num_mistakes'] + np.random.normal(0, 0.1, len(df))
y = df['weighted_confidence'] + np.random.normal(0, 0.01, len(df))

plt.figure(figsize=(10, 5))
for i, label in enumerate(df['label'].unique()):
    idx = df['label'] == label
    plt.scatter(x[idx], y[idx], s=15, label=lab_to_long[int_to_lab[label]], marker=markers[i])
plt.xlabel('Number of mistakes')
plt.ylabel('Weighted confidence')
plt.legend()
plt.savefig(os.path.join(destination, 'num_mistakes_vs_weighted_confidence.pdf'), dpi=300)
plt.close()