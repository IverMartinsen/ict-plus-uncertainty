import os
import glob
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from utils import lab_to_int, lab_to_long, make_dataset, load_data, compute_predictive_variance, plot_images

# hyperparameters
image_size = [224, 224]
batch_size = 32
destination = './ensemble_stats/'
path_to_val_data = './data/Training_Dataset_Cropped_Split/val/'

# load the models
path_to_models = './ensemble/'
models = glob.glob(path_to_models + '*.keras')
models.sort()
models = [tf.keras.models.load_model(m) for m in models]

# load the validation data
X_val, y_val = load_data(path_to_val_data)
ds_val = make_dataset(X_val, y_val, image_size, batch_size, shuffle=False)

# get the predictions
Y_pred = np.empty((len(y_val), len(models), len(lab_to_int)))
for i, model in enumerate(models):
    predictions = model.predict(ds_val)
    Y_pred[:, i, :] = predictions

# =============================================================================
# SUMMARY STATISTICS AND VISUALIZATIONS
# =============================================================================

# create a dataframe to store the predictions
columns = ['filename', 'label'] + [f'model_{i}' for i in range(len(models))]
df = pd.DataFrame(columns=columns)

df['filename'] = X_val
df['label'] = y_val

for i in range(len(models)):
    df[f'model_{i}'] = Y_pred[:, i, :].argmax(axis=1)
    
df['agree'] = np.prod(df.iloc[:, 2:] == df.iloc[:, 2].values[:, None], axis=1).astype(bool) # check if all models agree
df['pred_mode'] = df.iloc[:, 2:].mode(axis=1)[0] # majority vote
df['pred_mean'] = Y_pred.mean(axis=1).argmax(axis=1) # mean prediction
df['loss'] = -np.log(Y_pred.mean(axis=1)[np.arange(len(y_val)), y_val])
df.to_csv(os.path.join(destination, 'ensemble_predictions.csv'), index=False)

# confusion matrix
confusion = confusion_matrix(df['label'], df['pred_mean'])
disp = ConfusionMatrixDisplay(confusion, display_labels=list(lab_to_long.values()))
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax)
plt.savefig(os.path.join(destination, 'confusion_matrix.png'), bbox_inches='tight', dpi=300)

# total accuracy
summary= pd.DataFrame(columns=[f'model_{i}' for i in range(len(models))] + ['majority_vote', 'mean_vote'], index=['accuracy'])
for i in range(len(models)):
    summary[f'model_{i}'] = (df['label'] == df[f'model_{i}']).mean()
summary['majority_vote'] = (df['label'] == df['pred_mode']).mean()
summary['mean_vote'] = (df['label'] == df['pred_mean']).mean()
summary = summary.T
summary.to_csv(os.path.join(destination, 'ensemble_summary.csv'))

# class wise accuracy
class_wise_accuracy = classification_report(df['label'], df['pred_mean'], target_names=list(lab_to_long.values()), output_dict=True)
class_wise_df = pd.DataFrame(class_wise_accuracy).T
class_wise_df.to_csv(os.path.join(destination, 'class_wise_accuracy.csv')

# =============================================================================
# UNCERTAINTY ANALYSIS AND VISUALIZATIONS
# =============================================================================

tot_cov = compute_predictive_variance(Y_pred)
gen_var = np.linalg.det(tot_cov)
tot_var = np.trace(tot_cov, axis1=1, axis2=2)

# plot the uncertainty
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.scatter(tot_var, df['loss'], c=df['label'] == df['pred_mean'])
ax1.set_ylabel('Loss')
ax1.set_xlabel(r'tr$(\Sigma)$')
ax1.set_title('Total variance vs Loss', fontsize=10, fontweight='bold')
ax1.legend(['correct', 'wrong'])

ax2.scatter(np.log(gen_var), np.log(df['loss']), c=df['label'] == df['pred_mean'])
ax2.set_ylabel('log(Loss)')
ax2.set_xlabel(r'log|$\Sigma$|')
ax2.set_title('Generalized variance vs Loss', fontsize=10, fontweight='bold')
ax2.legend(['correct', 'wrong'])

plt.savefig('uncertainty.png', bbox_inches='tight', dpi=300)

# group the data based on the agreement
tricky = df[(df['agree'] == True) & (df['pred_mode'] != df['label'])]
hard = df[(df['agree'] == False) & (df['pred_mode'] != df['label'])]

plot_images(tricky, 5, 'tricky.png', image_size=image_size, destination=destination)
plot_images(hard, 5, 'hard.png', image_size=image_size, destination=destination)

# Group the data based on the uncertainty
threshold = np.median(gen_var)
tricky = df[(gen_var < threshold) & (df['pred_mean'] != df['label'])]
hard = df[(gen_var >= threshold) & (df['pred_mean'] != df['label'])]

plot_images(tricky, 2, 'tricky_gen_var.png', image_size=image_size, destination=destination)
plot_images(hard, 5, 'hard_gen_var.png', image_size=image_size, destination=destination)
