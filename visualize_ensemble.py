import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

destination = 'stats/tta_stats'#'./ensemble_stats/'
path_to_df = os.path.join(destination, 'predictions.csv')

df = pd.read_csv(path_to_df)

x = df['predictive_variance']
y = df['total_variance']

fig, ax = plt.subplots(figsize=(10, 5))
plt.scatter(x, y, s=8)
plt.xlabel('Variance for the predicted class')
plt.ylabel('Total Variance for all classes')
plt.savefig(os.path.join(destination, 'predictive_variance_vs_total_variance.pdf'), bbox_inches='tight', dpi=300)

import scipy.stats as stats




x = df['percentage_agree']
#x, _ = stats.boxcox(x)
x = 1 - x
x += np.random.normal(0, 0.01, x.shape)
#x = np.sqrt(x)
#x = np.log(x + 1e-24)

y = df['epistemic_variance']
y -= np.min(y)
y /= np.max(y)
y /= (3/2)
#y = np.sqrt(y)
#y, _ = stats.boxcox(y)
#power = 0.5
#y = np.log(y + 1e-24)
fig, ax = plt.subplots(figsize=(10, 5))
plt.scatter(x, y, s=8)
plt.show()
