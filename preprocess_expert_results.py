import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from utils import lab_to_int

destination = "/Users/ima029/Desktop/IKT+ Uncertainty/Repository/results/expert_results/"
os.makedirs(destination, exist_ok=True)

path_to_files = "/Users/ima029/Desktop/IKT+ Uncertainty/Repository/data/Man vs machine_Iver_cropped"


# import csv from Steffen
expert_results = "./data/Expert results/all_questionnaire_results_with error estimate_separated_plot2.xls"
expert_results = pd.read_excel(expert_results, sheet_name=None)

experts = ['Tine', 'Marit-Solveig', 'Kasia', 'Morten', 'Steffen', 'Eirik ']

keys = experts[:4] # exclude Eirik and Steffen

# create new dataframe with condensed information
df = pd.DataFrame()
df['filename'] = expert_results[keys[0]]['image'].iloc[:-1]
df['label'] = expert_results[keys[0]]['true class'].iloc[:-1].apply(lambda x: lab_to_int[x])

for key in keys:
    tmp = expert_results[key]
    tmp = tmp.set_index('image').loc[df['filename']].reset_index()
    print(tmp['image'].iloc[:10])
    y_pred = tmp['response'].apply(lambda x: lab_to_int[x])
    df[key] = y_pred
    df[key + '_uncertainty'] = tmp['certainity']
    df[key + '_time'] = tmp['time']

t = np.zeros(len(df))
for key in keys:
    time = df[key + '_time']
    # normalize time
    time = time - np.mean(time)
    time = time / np.std(time)
    t += time
df['time_standard'] = t / len(keys)


df['agree'] = np.prod(df.iloc[:, 2:6] == df.iloc[:, 2].values[:, None], axis=1).astype(bool) # check if all models agree
df['percentage_agree'] = df[keys].apply(lambda x: x.value_counts().max() / x.value_counts().sum(), axis=1)
df['unique_preds'] = df[keys].apply(lambda x: len(np.unique(x)), axis=1)
df['confidence'] = (df[[key + '_uncertainty' for key in keys]].mean(axis=1)) / 100
df['uncertainty'] = 1 - (df['percentage_agree'])*(df['confidence'])
df['pred_mode'] = df[keys].mode(axis=1)[0] # majority vote
df['rank'] = df['uncertainty'].rank()
df.to_csv(os.path.join(destination, 'predictions.csv'))

idx = df[(df['percentage_agree'] == 0.5) & (df['unique_preds'] == 2)].index

for i in idx:
    for key in keys:
        print(key, df.loc[i, key], df.loc[i, key + '_uncertainty'])
    print('=====================')

preds = [0, 0, 1, 0, 3, 1, 2, 2]

df.loc[idx, 'pred_mode'] = preds


test = []

m = 3
c = [0, 1, 2, 3]
k = len(c)
p = [0,0.25, 0.5, 0.75, 1.0]

for j in range(100000):
    v = np.random.choice(c, m, replace=True)
    #print(v)
    # turn v into 1-hot matrix
    v_ = np.zeros((m, k))
    #print(v_)
    v_[np.arange(m), v] = 1
    #print(v_)
    p_ = np.random.choice(p, m, replace=True)
    #print(p_)
    p_ = np.repeat(p_, k).reshape(m, k)
    #print(p_)
    v_ = v_ * p_
    #print(v_)
    v_ = v_.sum(axis=0)
    #print(v_)
    n = np.unique(v).size
    if n == 1:
        x = np.sum(v_)
    else:
        x = np.max(v_) - np.sum(v_) / (n - 1) + np.max(v_) / (n - 1)
    #print(x)
    x /= m
    #print(x)
    if np.isnan(x):
        continue
    if x > 0 and x < 0.0001:
        continue
    test.append(x)


print(np.unique(np.round(test, 10)).size)


# plot time distribution
from scipy.stats import gaussian_kde
plt.figure(figsize=(10, 5))
for key in keys:
    time = df[key + '_time']
    # remove outliers
    #time = time[time < 30]
    # normalize time
    #time = time - np.mean
    #time = time / np.std(time)
    #plt.hist(time, bins=40, alpha=0.5, label=key)
    # add distribtuion curve
    x = np.linspace(time.min(), time.max(), 100)
    kde = gaussian_kde(time)
    #plt.plot(x, kde(x) * time.size, label=key + ' KDE')
    # add fill
    plt.fill_between(x, kde(x) * time.size, alpha=0.85, label=key)
    
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Density')
plt.title('Original time distribution')
plt.savefig(os.path.join(destination, 'time_distribution.pdf'), dpi=300)


# plot time distribution
t = []
r = []
y = []
c = []
for key in keys:
    time = df[key + '_time']
    # normalize time
    time = time - np.mean(time)
    time = time / np.std(time)
    t.append(time)
    r.append(df[key])
    y.append(df['label'])
    c.append(df[key + '_uncertainty'])
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

# plot confidence distribution
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

# plot agreement distribution
plt.figure(figsize=(10, 5))
bins = np.linspace(0.5, 1, 4)
plt.hist(df['percentage_agree'][df['label'] == df['pred_mode']], bins=bins, alpha=1.0, label='Correct prediction', density=True)
plt.hist(df['percentage_agree'][df['label'] != df['pred_mode']], bins=bins, alpha=0.4, label='Incorrect prediction', density=True)
plt.legend()
plt.xlabel('Agreement')
plt.ylabel('Density')
plt.title('Agreement distribution')
plt.savefig(os.path.join(destination, 'agreement_distribution.pdf'), dpi=300)




df_[df_['percentage_agree'] == 0.75]['mode_weights']


mode_weights = np.zeros(len(df))

for key in keys:
    idx = np.where(df[key] == df['pred_mode'])[0]
    mode_weights[idx] += df[key + '_uncertainty'][idx] / 100
    idx = np.where((df[key] != df['pred_mode']) & (df['unique_preds'] == 2))[0]
    mode_weights[idx] -= df[key + '_uncertainty'][idx] / 100
    idx = np.where((df[key] != df['pred_mode']) & (df['unique_preds'] == 3))[0]
    mode_weights[idx] -= df[key + '_uncertainty'][idx] / 200

df['mode_weights'] = mode_weights / len(keys)

# plots given disagreement
df_ = df[(df['mode_weights'] < 0.5625)]
df_


df_.to_csv(os.path.join(destination, 'disagreement.csv'))

# mode weight distribution
plt.figure(figsize=(10, 5))
#bins = np.unique(df['mode_weights'])
#bins -= 0.03
#bins = np.concatenate([bins, [1.03]])
bins = np.arange(0.00, 1.125, 0.0625)
bins -= 0.03
plt.hist(df['mode_weights'][df['label'] == df['pred_mode']], bins=bins, alpha=1.0, label='Correct prediction', density=True)
plt.hist(df['mode_weights'][df['label'] != df['pred_mode']], bins=bins, alpha=0.4, label='Incorrect prediction', density=True)
plt.legend()
plt.xlabel('Mode weight')
plt.ylabel('Density')
plt.xticks((bins[1:] - 0.03), (bins[:-1] + 0.03))
plt.title('Mode weight distribution')
plt.savefig(os.path.join(destination, 'mode_weight_distribution_disagreement.pdf'), dpi=300)

plt.figure(figsize=(10, 5))
plt.scatter(df['mode_weights'], df['time_standard'], c=(df['label'] == df['pred_mode']), s=10)
plt.xlabel('Mode weight')
plt.ylabel('Time (s)')
plt.title('Time vs mode weight')
plt.savefig(os.path.join(destination, 'time_vs_mode_weight_disagreement.pdf'), dpi=300)



from PIL import Image
from utils import int_to_lab, lab_to_long

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


df__ = df[(df['percentage_agree'] == 1.0) & (df['confidence'] == 1.0)]





plt.figure(figsize=(10, 5))
bins = np.unique(df_['confidence'])
bins -= 0.03
bins = np.concatenate([bins, [1.03]])
plt.hist(df_['confidence'][df_['label'] == df_['pred_mode']], bins=bins, alpha=1.0, label='Correct prediction', density=True)
plt.hist(df_['confidence'][df_['label'] != df_['pred_mode']], bins=bins, alpha=0.4, label='Incorrect prediction', density=True)
plt.legend()
plt.xlabel('Confidence')
plt.xticks(bins[1:] - 0.03, bins[:-1] + 0.03)
plt.ylabel('Density')
plt.title('Confidence distribution')
plt.savefig(os.path.join(destination, 'confidence_distribution_disagreement.pdf'), dpi=300)

plt.figure(figsize=(10, 5))
x = np.linspace(df_['time_standard'].min(), df_['time_standard'].max(), 100)
kde = gaussian_kde(df_['time_standard'][df_['label'] == df_['pred_mode']])
plt.fill_between(x, kde(x) * df_['time_standard'].size, alpha=0.85, label='Correct prediction')
kde = gaussian_kde(df_['time_standard'][df_['label'] != df_['pred_mode']])
plt.fill_between(x, kde(x) * df_['time_standard'].size, alpha=0.85, label='Incorrect prediction')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Density')
plt.title('Standardized time distribution')
plt.savefig(os.path.join(destination, 'standardized_time_distribution_disagreement.pdf'), dpi=300)


x = df_['confidence'] + np.random.normal(0, 0.01, len(df_))
y = df_['time_standard'] + np.random.normal(0, 0.01, len(df_))

plt.figure(figsize=(10, 5))
plt.scatter(x, y, c=(df_['label'] == df_['pred_mode']), s=10)
plt.xlabel('Confidence')
plt.ylabel('Time (s)')
plt.title('Time vs confidence')
plt.savefig(os.path.join(destination, 'time_vs_confidence_disagreement.pdf'), dpi=300)



plt.figure(figsize=(10, 5))
plt.scatter(c + np.random.normal(0, 1, len(df)*4), t, c=(r == y), s=10)
plt.show()


t_ = np.repeat(df['time_standard'], 1)
c_ = c.reshape(-1, 4).mean(axis=1)

plt.figure(figsize=(10, 5))
for i in np.unique(c_):
    plt.boxplot(t_[c_ == i], positions=[i], widths=5)
plt.xlabel('Confidence')
plt.ylabel('Time (s)')
plt.title('Time vs confidence')
plt.show()

plt.scatter(c_ + np.random.normal(0, 1, len(df)), t_, c=(df['label'] == df['pred_mode']), s=10)
plt.xlabel('Confidence')
plt.ylabel('Time (s)')
plt.title('Time vs confidence')
plt.show()



idx = df[(df['percentage_agree'] < 1.0)].index

# make a 3d plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
x = df['percentage_agree'].iloc[idx] + np.random.normal(0, 0.01, len(idx))
y = df['time_standard'].iloc[idx]
z = df['confidence'].iloc[idx] + np.random.normal(0, 0.01, len(idx))


ax.scatter(x, y, z, c=(df['label'].iloc[idx] == df['pred_mode'].iloc[idx]), s=10)
ax.set_xlabel('Percentage agree')
ax.set_ylabel('Time (s)')
ax.set_zlabel('Confidence')
plt.title('Time vs confidence vs correct prediction')
plt.show()

df['uncertainty'] = 1 - (df['percentage_agree']) + 1 - (df['confidence']) + df['time_standard']

# plot uncertainty vs expert prediction
plt.figure(figsize=(10, 5))
plt.scatter(
    np.arange(len(df)), 
    np.sort(df['uncertainty']) + np.random.normal(0, 0.01, len(df)), 
    c=(df['pred_mode'] == df['label']), 
    label='Expert prediction wrong',
    s=10,
)
plt.legend()
plt.xlabel('Sample index (ordered by uncertainty)')
plt.ylabel('Expert uncertainty')
plt.savefig(os.path.join(destination, 'uncertainty_vs_expert_prediction.pdf'), dpi=300)

# Summary statistics
summary = pd.DataFrame(index=['accuracy'])
for key in keys:
    summary[key] = (df['label'] == df[key]).mean()
summary['majority_vote'] = (df['label'] == df['pred_mode']).mean()

# compute cohens kappa between the models
kappa = np.zeros((4, 4))
for i, key in enumerate(keys):
    for j, key2 in enumerate(keys):
        if key == key2:
            continue
        kappa[i, j] = cohen_kappa_score(df[key], df[key2])
# upper triangular matrix
kappa = np.triu(kappa, k=1)

summary['kappa'] = kappa.sum() / np.count_nonzero(kappa)
summary = summary.T

summary.to_csv(os.path.join(destination, 'summary.csv'))
