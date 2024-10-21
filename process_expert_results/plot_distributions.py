import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

destination = "./results/expert_results/"

os.makedirs(destination, exist_ok=True)

df = pd.read_csv(os.path.join(destination, 'predictions.csv'))

experts = ['Tine', 'Marit-Solveig', 'Kasia', 'Morten', 'Steffen', 'Eirik '][:4] # exclude Eirik and Steffen

# ==============================
# PLOT TIME DISTRIBUTION
# ==============================
plt.figure(figsize=(10, 5))

for expert in experts:
    time = df[expert + '_time']
    x = np.linspace(time.min(), time.max(), 100)
    kde = gaussian_kde(time)
    plt.fill_between(x, kde(x) * time.size, alpha=0.85, label=expert)

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Density')
plt.title('Original time distribution')
plt.savefig(os.path.join(destination, 'time_distribution.pdf'), dpi=300)
plt.close()

# ==============================
# PLOT STANDARDIZED TIME DISTRIBUTION
# ==============================
t = []
r = []
y = []
c = []
for expert in experts:
    time = df[expert + '_time']
    # normalize time
    time = time - np.mean(time)
    time = time / np.std(time)
    t.append(time)
    r.append(df[expert])
    y.append(df['label'])
    c.append(df[expert + '_uncertainty'])
t = np.array(t).flatten()
r = np.array(r).flatten()
y = np.array(y).flatten()
c = np.array(c).flatten()

plt.figure(figsize=(10, 5))
x = np.linspace(t.min(), t.max(), 100)
kde = gaussian_kde(t[r == y])
plt.fill_between(x, kde(x) * t.size, alpha=0.85, label='Correct prediction')
kde = gaussian_kde(t[r != y])
plt.fill_between(x, kde(x) * t.size, alpha=0.85, label='Incorrect prediction')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Density')
plt.title('Standardized time distribution')
plt.savefig(os.path.join(destination, 'standardized_time_distribution.pdf'), dpi=300)
plt.close()

# ==============================
# PLOT CONFIDENCE DISTRIBUTION
# ==============================
plt.figure(figsize=(10, 5))
bins = np.unique(df['mean_confidence'])
#bins = np.concatenate([[0.4], bins])
bins -= 0.03
bins = np.concatenate([bins, [1.03]])
x1 = df['mean_confidence'][df['label'] == df['pred_weighted']]
x2 = df['mean_confidence'][df['label'] != df['pred_weighted']]
plt.hist([x1, x2], bins=bins, alpha=1.00, label=['Correct prediction', 'Incorrect prediction'], density=True, histtype='bar')
#plt.hist(df['mean_confidence'][df['label'] == df['pred_weighted']], bins=bins, alpha=1.0, label='Correct prediction', density=True)
#plt.hist(df['mean_confidence'][df['label'] != df['pred_weighted']], bins=bins, alpha=0.4, label='Incorrect prediction', density=True)
#plt.hist(df['mean_confidence'][df['label'] == df['pred_mode']], bins=bin2, alpha=1.0, label='Correct prediction', density=True)
#plt.hist(df['mean_confidence'], bins=bins, alpha=1.0, label='All predictions', density=True)
plt.legend(fontsize=14)
plt.xlabel('Confidence', fontsize=14)
plt.ylabel('Density', fontsize=15)
plt.xticks(bins[1:] - 0.03, bins[:-1] + 0.03, fontsize=14)
plt.yticks([])
plt.tight_layout()
plt.savefig(os.path.join(destination, 'confidence_distribution.pdf'), dpi=300)
plt.close()

# ==============================
# PLOT AGREEMENT DISTRIBUTION
# ==============================
plt.figure(figsize=(10, 5))
bins = np.linspace(0.5, 1, 4)
plt.hist(df['percentage_agree'][df['label'] != df['pred_mode']], bins=bins, alpha=1.0, label='Incorrect prediction', density=True, color="C1")
plt.hist(df['percentage_agree'][df['label'] == df['pred_mode']], bins=bins, alpha=1.0, label='Correct prediction', density=True, color="C0")
plt.legend(fontsize=14)
plt.xlabel('Agreement', fontsize=14)
plt.ylabel('Density', fontsize=15)
plt.xticks([0.58, 0.75, 0.92], [0.50, 0.75, 1.00], fontsize=14)
plt.yticks([])
plt.tight_layout()
plt.savefig(os.path.join(destination, 'agreement_distribution.pdf'), dpi=300)
plt.close()

# ==============================
# PLOT WEIGHTED CONFIDENCE DISTRIBUTION
# ==============================
plt.figure(figsize=(10, 5))
bins = np.arange(0.00, 1.125, 0.0625)
bins -= 0.03
plt.hist(df['weighted_confidence'][df['label'] != df['pred_weighted']], bins=bins, alpha=1.0, label='Incorrect prediction', density=True, color="C1")
plt.hist(df['weighted_confidence'][df['label'] == df['pred_weighted']], bins=bins, alpha=1.0, label='Correct prediction', density=True, color="C0")
plt.legend(fontsize=14)
plt.xlabel('Weighted confidence', fontsize=14)
plt.ylabel('Density', fontsize=15)
plt.xticks((bins[1:] - 0.03)[::2], (bins[:-1] + 0.03)[::2], fontsize=14)
plt.yticks([])
plt.tight_layout()
plt.savefig(os.path.join(destination, 'weighted_confidence_distribution.pdf'), dpi=300)
plt.close()
