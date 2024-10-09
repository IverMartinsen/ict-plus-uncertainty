import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from utils.utils import int_to_lab, lab_to_long


destination = "/Users/ima029/Desktop/IKT+ Uncertainty/Repository/results/expert_results/"
path_to_files = "/Users/ima029/Desktop/IKT+ Uncertainty/Repository/data/Man vs machine_Iver_cropped"

os.makedirs(destination, exist_ok=True)


df = pd.read_csv(os.path.join(destination, 'predictions.csv'))

experts = ['Tine', 'Marit-Solveig', 'Kasia', 'Morten', 'Steffen', 'Eirik '][:4] # exclude Eirik and Steffen

# ==============================
# PLOT UNCERTAINT PREDICTIONS
# ==============================
df_ = df[(df['weighted_confidence'] < 0.5625)]
df_.to_csv(os.path.join(destination, 'disagreement.csv'))

# ==============================
# PLOT TIME VS MODE WEIGHT
# ==============================
plt.figure(figsize=(10, 5))
plt.scatter(df['weighted_confidence'], df['time_standard'], c=(df['label'] == df['pred_mode']), s=10)
plt.xlabel('Mode weight')
plt.ylabel('Time (s)')
plt.title('Time vs mode weight')
plt.savefig(os.path.join(destination, 'time_vs_mode_weight.pdf'), dpi=300)

# ==============================
# DISPLAY UNCERTAIN IMAGES
# ==============================

fig, ax = plt.subplots(6, 6, figsize=(14, 14))
for i, ax_ in enumerate(ax.flatten()):
    try:
        path = os.path.join(path_to_files, df_['filename'].iloc[i][0], df_['filename'].iloc[i])
        label = lab_to_long[int_to_lab[df_['label'].iloc[i]]]
        pred = lab_to_long[int_to_lab[df_['pred_mode'].iloc[i]]]
        ax_.imshow(Image.open(path).resize((224, 224)))
        ax_.set_title(f'Label: {label}\nPred: {pred}\nFile: {os.path.basename(path)}', fontsize=6, fontweight='bold')
    except:
        pass
    ax_.axis('off')
plt.subplots_adjust(hspace=0.3)
plt.savefig(os.path.join(destination, 'disagreement_images.pdf'), dpi=300, bbox_inches='tight')

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
