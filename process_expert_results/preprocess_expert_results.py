import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import cohen_kappa_score
from scipy.stats import gaussian_kde, spearmanr
from utils.utils import lab_to_int, int_to_lab, lab_to_long, store_confusion_matrix


destination = "/Users/ima029/Desktop/IKT+ Uncertainty/Repository/results/expert_results/"
path_to_files = "/Users/ima029/Desktop/IKT+ Uncertainty/Repository/data/Man vs machine_Iver_cropped"

os.makedirs(destination, exist_ok=True)

# import csv from Steffen
expert_results = "./data/Expert results/all_questionnaire_results_with error estimate_separated_plot2.xls"
expert_results = pd.read_excel(expert_results, sheet_name=None)

experts = ['Tine', 'Marit-Solveig', 'Kasia', 'Morten', 'Steffen', 'Eirik ']

experts = experts[:4] # exclude Eirik and Steffen

# create new dataframe with condensed information
df = pd.DataFrame()
df['filename'] = expert_results[experts[0]]['image'].iloc[:-1]
df['label'] = expert_results[experts[0]]['true class'].iloc[:-1].apply(lambda x: lab_to_int[x])

for expert in experts:
    tmp = expert_results[expert]
    tmp = tmp.set_index('image').loc[df['filename']].reset_index()
    print(tmp['image'].iloc[:10])
    y_pred = tmp['response'].apply(lambda x: lab_to_int[x])
    df[expert] = y_pred
    try:
        df[expert + '_uncertainty'] = tmp['certainty']
    except KeyError:
        df[expert + '_uncertainty'] = tmp['certainity']
    df[expert + '_time'] = tmp['time']

t = np.zeros(len(df))
for expert in experts:
    time = df[expert + '_time']
    # normalize time
    time = time - np.mean(time)
    time = time / np.std(time)
    t += time
df['time_standard'] = t / len(experts)


df['agree'] = np.prod(df.iloc[:, 2:6] == df.iloc[:, 2].values[:, None], axis=1).astype(bool) # check if all models agree
df['percentage_agree'] = df[experts].apply(lambda x: x.value_counts().max() / x.value_counts().sum(), axis=1)
df['unique_preds'] = df[experts].apply(lambda x: len(np.unique(x)), axis=1)
df['confidence'] = (df[[expert + '_uncertainty' for expert in experts]].mean(axis=1)) / 100
df['uncertainty'] = 1 - (df['percentage_agree'])*(df['confidence'])
df['pred_mode'] = df[experts].mode(axis=1)[0] # majority vote
df['rank'] = df['uncertainty'].rank()
# handle special cases
idx = df[(df['percentage_agree'] == 0.5) & (df['unique_preds'] == 2)].index

for i in idx:
    for expert in experts:
        print(expert, df.loc[i, expert], df.loc[i, expert + '_uncertainty'])
    print('=====================')

preds = [0, 0, 1, 0, 3, 1, 2, 2]

df.loc[idx, 'pred_mode'] = preds

pred_weighted = np.zeros((len(df), 4))
for expert in experts:
    pred_weighted[np.arange(len(df)), df[expert]] += df[expert + '_uncertainty'] / 100
pred_weighted /= np.sum(pred_weighted, axis=1)[:, None]
df['pred_weighted'] = np.argmax(pred_weighted, axis=1)

# handle special cases
idx = np.where(np.sort(pred_weighted, axis=1)[:, -2] == np.sort(pred_weighted, axis=1)[:, -1])[0]
df.iloc[21]['pred_weighted'] = 1
df.iloc[55]['pred_weighted'] = 0

df['conf_mean'] = pred_weighted.max(axis=1)


mode_weights = np.zeros(len(df))
for expert in experts:
    idx = np.where(df[expert] == df['pred_weighted'])[0]
    mode_weights[idx] += df[expert + '_uncertainty'][idx] / 100
    idx = np.where((df[expert] != df['pred_weighted']) & (df['unique_preds'] == 2))[0]
    mode_weights[idx] -= df[expert + '_uncertainty'][idx] / 100
    idx = np.where((df[expert] != df['pred_weighted']) & (df['unique_preds'] == 3))[0]
    mode_weights[idx] -= df[expert + '_uncertainty'][idx] / 200

df['weighted_confidence'] = mode_weights / len(experts)

df.to_csv(os.path.join(destination, 'predictions.csv'))

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

# ==============================
# PLOT CONFIDENCE DISTRIBUTION
# ==============================
plt.figure(figsize=(10, 5))
bins = np.unique(df['confidence'])
#bins = np.concatenate([[0.4], bins])
bins -= 0.03
bins = np.concatenate([bins, [1.03]])
plt.hist(df['confidence'][df['label'] == df['pred_mode']], bins=bins, alpha=1.0, label='Correct prediction', density=True)
plt.hist(df['confidence'][df['label'] != df['pred_mode']], bins=bins, alpha=0.4, label='Incorrect prediction', density=True)
plt.legend()
plt.xlabel('Confidence')
plt.ylabel('Density')
plt.xticks(bins[1:] - 0.03, bins[:-1] + 0.03)
plt.title('Confidence distribution')
plt.savefig(os.path.join(destination, 'confidence_distribution.pdf'), dpi=300)

# ==============================
# PLOT AGREEMENT DISTRIBUTION
# ==============================
plt.figure(figsize=(10, 5))
bins = np.linspace(0.5, 1, 4)
plt.hist(df['percentage_agree'][df['label'] == df['pred_mode']], bins=bins, alpha=1.0, label='Correct prediction', density=True)
plt.hist(df['percentage_agree'][df['label'] != df['pred_mode']], bins=bins, alpha=0.4, label='Incorrect prediction', density=True)
plt.legend()
plt.xlabel('Agreement')
plt.ylabel('Density')
plt.title('Agreement distribution')
plt.savefig(os.path.join(destination, 'agreement_distribution.pdf'), dpi=300)

# ==============================
# PLOT UNCERTAINT PREDICTIONS
# ==============================
df_ = df[(df['weighted_confidence'] < 0.5625)]
df_.to_csv(os.path.join(destination, 'disagreement.csv'))

# ==============================
# PLOT MODE WEIGHT DISTRIBUTION
# ==============================
plt.figure(figsize=(10, 5))
bins = np.arange(0.00, 1.125, 0.0625)
bins -= 0.03
plt.hist(df['weighted_confidence'][df['label'] == df['pred_weighted']], bins=bins, alpha=1.0, label='Correct prediction', density=True)
plt.hist(df['weighted_confidence'][df['label'] != df['pred_weighted']], bins=bins, alpha=0.4, label='Incorrect prediction', density=True)
plt.legend()
plt.xlabel('Weighted confidence')
plt.ylabel('Density')
plt.xticks((bins[1:] - 0.03), (bins[:-1] + 0.03))
plt.title('Weighted confidence distribution')
plt.savefig(os.path.join(destination, 'weighted_confidence_distribution.pdf'), dpi=300)

# ==============================
# PLOT MEAN CONFIDENCE DISTRIBUTION
# ==============================
plt.figure(figsize=(10, 5))
steps = 0.03
bins = np.arange(np.min(df['conf_mean']), np.max(df['conf_mean']) + steps, steps)
plt.hist(df['conf_mean'][df['label'] == df['pred_weighted']], bins=bins, alpha=1.0, label='Correct prediction', density=True)
plt.hist(df['conf_mean'][df['label'] != df['pred_weighted']], bins=bins, alpha=0.4, label='Incorrect prediction', density=True)
plt.legend()
plt.xlabel('Mean confidence')
plt.ylabel('Density')
plt.title('Mean confidence distribution')
plt.savefig(os.path.join(destination, 'mean_confidence_distribution.pdf'), dpi=300)
plt.close()

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
    


# ==============================
# SUMMARY STATISTICS
# ==============================
summary = pd.DataFrame(index=['accuracy'])

for expert in experts:
    summary[expert] = (df['label'] == df[expert]).mean()

summary['majority_vote'] = (df['label'] == df['pred_mode']).mean()

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