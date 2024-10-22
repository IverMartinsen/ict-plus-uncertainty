import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import int_to_lab, lab_to_long

df_experts = pd.read_csv("./results/expert_results/predictions.csv")
df_model = pd.read_csv('./results/ensemble_results/predictions.csv')

# ==============================
# SCATTER PLOT OF CONFIDENCE IN ASCENDING ORDER
# ==============================

fig, ax = plt.subplots(figsize=(10, 5))

y_e = df_experts['weighted_confidence'].values
z = df_experts['label'].values
order = np.argsort(y_e)
y_e = y_e[order]
z = z[order]
x = np.arange(len(order))

markers = ['o', 'x', 's', '>']

for i, label in enumerate(df_experts['label'].unique()):
    idx = z == label
    ax.scatter(x[idx], y_e[idx] + np.random.normal(0, 0.04, len(y_e[idx])), label=lab_to_long[int_to_lab[label]], s=20, marker=markers[i])
ax.set_xlabel('Sample', fontsize=12)
ax.set_ylabel('Weighted confidence', fontsize=12)
ax.legend(fontsize=12)
plt.savefig('expert_confidence_ascending.pdf', dpi=300)
plt.close()





fig, ax = plt.subplots(figsize=(10, 5))

y = df_model['conf_mean'].values
z = df_model['label']
#x = np.argsort(y)
y = y[order]
z = z[order]
x = np.arange(len(order))

markers = ['o', 'x', 's', '>']

for i, label in enumerate(df_model['label'].unique()):
    idx = z == label
    ax.scatter(x[idx], y[idx] + np.random.normal(0, 0.03, len(y[idx])), label=lab_to_long[int_to_lab[label]], s=20, marker=markers[i])
ax.set_xlabel('Sample', fontsize=12)
ax.set_ylabel('Confidence', fontsize=12)
ax.legend(fontsize=12)
plt.savefig('ensemble_confidence_ascending.pdf', dpi=300)
plt.close()






fig, ax = plt.subplots(figsize=(10, 5))

y = df_model['conf_mean'].values - y_e
z = df_model['label']
x = np.argsort(y)
y = y[order]
z = z[order]
x = np.arange(len(order))

markers = ['o', 'x', 's', '>']

for i, label in enumerate(df_model['label'].unique()):
    idx = z == label
    ax.scatter(x[idx], y[idx] + np.random.normal(0, 0.03, len(y[idx])), label=lab_to_long[int_to_lab[label]], s=20, marker=markers[i])
ax.set_xlabel('Sample', fontsize=12)
ax.set_ylabel('Ensemble confidence - expert confidence', fontsize=12)
ax.legend(fontsize=12)
plt.savefig('model_minus_expert_confidence_ascending.pdf', dpi=300)
plt.close()


