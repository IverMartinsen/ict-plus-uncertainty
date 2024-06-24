import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr
from utils import lab_to_int

expert_results = "./data/Expert results/all_questionnaire_results_with error estimate_separated_plot2.xls"
expert_results = pd.read_excel(expert_results, sheet_name=None)

experts = ['Tine', 'Marit-Solveig', 'Kasia', 'Morten', 'Steffen', 'Eirik ']
kappas = []

for expert in experts:
    expert_results_df = expert_results[expert]
    expert_results_df = expert_results_df.iloc[:-1]

    machine_results = "/Users/ima029/Desktop/IKT+ Uncertainty/Repository/stats/ensemble_stats/predictions.csv"
    machine_results = pd.read_csv(machine_results)
    machine_results['basename'] = machine_results['filename'].apply(lambda x: x.split('/')[-1])
    machine_results = machine_results.set_index('basename').loc[expert_results_df['image']].reset_index()

    assert (expert_results_df['image'] == machine_results['basename']).all()

    y1 = expert_results_df['response'].apply(lambda x: lab_to_int[x])
    y2 = machine_results['pred_mean']

    kappas.append(cohen_kappa_score(y1, y2))

kappa = np.mean(kappas)

print(f"Cohen's Kappa: {kappa}")

p1 = df1['var_mean']
p2 = df2['variance']

spearman = spearmanr(p1, p2)

print(f"Spearman's Rho: {spearman.correlation}")



preds = np.empty((260, 4))

for i, expert in enumerate(experts[:4]):
    y_pred = expert_results[expert]['response'].iloc[:-1].apply(lambda x: lab_to_int[x])
    preds[:, i] = y_pred

from scipy import stats

mode = stats.mode(preds, axis=1).mode.reshape(260, 1)

perc_agree = np.mean(preds == mode, axis=1)

machine_results = "/Users/ima029/Desktop/IKT+ Uncertainty/Repository/stats/ensemble_stats/predictions.csv"
machine_results = pd.read_csv(machine_results)
machine_results['basename'] = machine_results['filename'].apply(lambda x: x.split('/')[-1])
machine_results = machine_results.set_index('basename').loc[expert_results[expert].iloc[:-1]['image']].reset_index()

idx = np.where(perc_agree < 1.0)[0]


(np.abs(machine_results['percentage_agree'].iloc[idx] - perc_agree[idx])).mean()

(np.abs(1.0 - perc_agree[idx])).mean()