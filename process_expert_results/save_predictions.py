import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from utils.utils import lab_to_int


destination = "./results/expert_results/"
path_to_files = "./data/Man vs machine_Iver_cropped"

os.makedirs(destination, exist_ok=True)

# import csv
expert_results = "./data/Expert results/all_questionnaire_results_with error estimate_separated_plot2.xls"
expert_results = pd.read_excel(expert_results, sheet_name=None)

experts = ['Tine', 'Marit-Solveig', 'Kasia', 'Morten', 'Steffen', 'Eirik '][:4] # exclude Eirik and Steffen

# create new dataframe with condensed information
df = pd.DataFrame()
df['filename'] = expert_results[experts[0]]['image'].iloc[:-1]
df['label'] = expert_results[experts[0]]['true class'].iloc[:-1].apply(lambda x: lab_to_int[x])

t = np.zeros(len(df))
pred_weighted = np.zeros((len(df), 4))

df['num_mistakes'] = np.zeros(len(df))

for expert in experts:
    tmp = expert_results[expert]
    tmp = tmp.set_index('image').loc[df['filename']].reset_index()
    df[expert] = tmp['response'].apply(lambda x: lab_to_int[x])
    df['num_mistakes'] += (df[expert].values != df['label']).astype(int)
    try:
        df[expert + '_uncertainty'] = tmp['certainty']
    except KeyError:
        df[expert + '_uncertainty'] = tmp['certainity']
    time = tmp['time']
    df[expert + '_time'] = time
    # normalize time
    time = time - np.mean(time)
    time = time / np.std(time)
    t += time
    # weighted predictions
    pred_weighted[np.arange(len(df)), df[expert]] += df[expert + '_uncertainty'] / 100

pred_weighted /= np.sum(pred_weighted, axis=1)[:, None]

df['pred_weighted'] = np.argmax(pred_weighted, axis=1)
df['time_standardized'] = t / len(experts)
df['percentage_agree'] = df[experts].apply(lambda x: x.value_counts().max() / x.value_counts().sum(), axis=1)
df['num_unique_preds'] = df[experts].apply(lambda x: len(np.unique(x)), axis=1)
df['agree'] = df['num_unique_preds'] == 1
df['mean_confidence'] = (df[[expert + '_uncertainty' for expert in experts]].mean(axis=1)) / 100
df['pred_mode'] = df[experts].mode(axis=1)[0] # majority vote

# handle special cases for mode prediction
idx = df[(df['percentage_agree'] == 0.5) & (df['num_unique_preds'] == 2)].index

print('These special cases need to be handled manually:')
for i in idx:
    print(f'Index: {i}')
    for expert in experts:
        print(expert, df.loc[i, expert], df.loc[i, expert + '_uncertainty'])
    print('=====================')

# hardcode the special cases
df.loc[idx, 'pred_mode'] = [0, 0, 1, 0, 3, 1, 2, 2]

# handle and hardcode special cases for weighted prediction
idx = np.where(np.sort(pred_weighted, axis=1)[:, -2] == np.sort(pred_weighted, axis=1)[:, -1])[0]
df.iloc[21]['pred_weighted'] = 1
df.iloc[55]['pred_weighted'] = 0

# compute weighted confidence
weighted_confidence = np.zeros(len(df))
for expert in experts:
    idx = np.where(df[expert] == df['pred_weighted'])[0]
    weighted_confidence[idx] += df[expert + '_uncertainty'][idx] / 100
    idx = np.where((df[expert] != df['pred_weighted']) & (df['num_unique_preds'] == 2))[0]
    weighted_confidence[idx] -= df[expert + '_uncertainty'][idx] / 100
    idx = np.where((df[expert] != df['pred_weighted']) & (df['num_unique_preds'] == 3))[0]
    weighted_confidence[idx] -= df[expert + '_uncertainty'][idx] / 200

df['weighted_confidence'] = weighted_confidence / len(experts)

df.to_csv(os.path.join(destination, 'predictions.csv'))
