import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from utils.utils import int_to_lab, lab_to_long

destination = "./results/expert_results/"
path_to_files = "./data/Man vs machine_Iver_cropped"
display_filenames = False

os.makedirs(destination, exist_ok=True)

df = pd.read_csv(os.path.join(destination, 'predictions.csv'))

# ==============================
# DISPLAY UNCERTAIN IMAGES
# ==============================

df_ = df[(df['weighted_confidence'] < 0.5625)]
df_.to_csv(os.path.join(destination, 'uncertaint_images.csv'))

lab_to_long = {'A': 'B. Agglutinated', 'B': 'B. Calcareous', 'S': 'Sediment', 'P': 'Planktic'}

fig, ax = plt.subplots(4, 8, figsize=(20, 10))
for i, ax_ in enumerate(ax.flatten()):
    try:
        path = os.path.join(path_to_files, df_['filename'].iloc[i][0], df_['filename'].iloc[i])
        label = lab_to_long[int_to_lab[df_['label'].iloc[i]]]
        pred = lab_to_long[int_to_lab[df_['pred_mode'].iloc[i]]]
        ax_.imshow(Image.open(path).resize((224, 224)))
        if label == pred:
            color = 'green'
        else:
            color = 'red'
        if display_filenames:
            ax_.set_title(f'Label: {label}\nPred: {pred}\nFile: {os.path.basename(path)}', fontsize=10, fontweight='bold', color=color)
        else:
            ax_.set_title(f'Label: {label}\nPred: {pred}', fontsize=10, fontweight='bold', color=color)
    except:
        pass
    ax_.axis('off')
plt.subplots_adjust(hspace=0.3)
plt.tight_layout()
plt.savefig(os.path.join(destination, 'uncertain_images.pdf'), dpi=300, bbox_inches='tight')
plt.close()
