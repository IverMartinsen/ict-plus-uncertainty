import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
