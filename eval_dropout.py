import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from utils import (
    lab_to_int, lab_to_long, make_dataset, load_data, compute_predictive_variance, plot_images
    )

# hyperparameters
path_to_model = './ensemble/20240312_155425.keras'
destination = './dropout_stats/'
image_size = [224, 224]
batch_size = 32
num_samples = 100

os.makedirs(destination, exist_ok=True)

# load the model
model = tf.keras.models.load_model(path_to_model)
base_model = model.layers[-2]
classification_head = model.layers[-1]

# Set the batch normalization layers.trainable to False.
# That way, BN statistics are constant even when calling the model with training=True.
for i, layer in enumerate(base_model.layers):
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        print(layer)
        layer.trainable = False

def dropout_call(x):
    x = base_model(x, training=True)
    x = model.layers[-1](x, training=True)
    return x


# load the validation data
path_to_val_data = './data/Training_Dataset_Cropped_Split/val/'
X_val, y_val = load_data(path_to_val_data)
ds_val = make_dataset(X_val, y_val, image_size, batch_size)

# get the predictions
Y_pred = np.empty((len(y_val), num_samples, len(lab_to_int)))

for i in range(num_samples):
    predictions = []
    for batch in ds_val:
        predictions.append(dropout_call(batch[0]))
    predictions = np.concatenate(predictions, axis=0)
    Y_pred[:, i, :] = predictions

# =============================================================================
# SUMMARY STATISTICS AND VISUALIZATIONS
# =============================================================================

# create a dataframe to store the predictions
columns = ['filename', 'label'] + [f'model_{i}' for i in range(num_samples)]
df = pd.DataFrame(columns=columns)

df['filename'] = X_val
df['label'] = y_val

for i in range(num_samples):
    df[f'model_{i}'] = Y_pred[:, i, :].argmax(axis=1)
    
df['agree'] = np.prod(df.iloc[:, 2:] == df.iloc[:, 2].values[:, None], axis=1).astype(bool) # check if all models agree
df['pred_mode'] = df.iloc[:, 2:].mode(axis=1)[0] # majority vote
df['pred_mean'] = Y_pred.mean(axis=1).argmax(axis=1) # mean prediction
df['loss'] = -np.log(Y_pred.mean(axis=1)[np.arange(len(y_val)), y_val])
df.to_csv(os.path.join(destination, 'dropout_predictions.csv'), index=False)

# total accuracy
summary= pd.DataFrame(columns=[f'model_{i}' for i in range(num_samples)] + ['majority_vote', 'mean_vote'], index=['accuracy'])
for i in range(num_samples):
    summary[f'model_{i}'] = (df['label'] == df[f'model_{i}']).mean()
summary['majority_vote'] = (df['label'] == df['pred_mode']).mean()
summary['mean_vote'] = (df['label'] == df['pred_mean']).mean()
summary = summary.T
summary.to_csv(os.path.join(destination, 'dropout_summary.csv'))

# confusion matrix
confusion = confusion_matrix(df['label'], df['pred_mean'])
disp = ConfusionMatrixDisplay(confusion, display_labels=list(lab_to_long.values()))
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax)
plt.savefig(os.path.join(destination, 'confusion_matrix.png'), bbox_inches='tight', dpi=300)

# class wise accuracy
class_wise_accuracy = classification_report(df['label'], df['pred_mean'], target_names=list(lab_to_long.values()), output_dict=True)
class_wise_df = pd.DataFrame(class_wise_accuracy).T
class_wise_df.to_csv(os.path.join(destination, 'class_wise_accuracy.csv'))

# =============================================================================
# UNCERTAINTY ANALYSIS AND VISUALIZATIONS
# =============================================================================

cov = compute_predictive_variance(Y_pred)

generalized_variance = np.linalg.det(cov)
total_variance = np.trace(cov, axis1=1, axis2=2)

# plot the uncertainty
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.scatter(total_variance, df['loss'], c=df['label'] == df['pred_mean'])
ax1.set_ylabel('Loss')
ax1.set_xlabel(r'tr$(\Sigma)$')
ax1.set_title('Total variance vs Loss', fontsize=10, fontweight='bold')
ax1.legend(['correct', 'wrong'])

ax2.scatter(np.log(generalized_variance), np.log(df['loss']), c=df['label'] == df['pred_mean'])
ax2.set_ylabel('log(Loss)')
ax2.set_xlabel(r'log|$\Sigma$|')
ax2.set_title('Generalized variance vs Loss', fontsize=10, fontweight='bold')
ax2.legend(['correct', 'wrong'])

plt.savefig('uncertainty.png', bbox_inches='tight', dpi=300)

# group the data based on the agreement
hard = df[(df['agree'] == False) & (df['pred_mode'] != df['label'])]
tricky = df[(df['agree'] == True) & (df['pred_mode'] != df['label'])]

plot_images(hard, 5, 'hard_agreement.png')
plot_images(tricky, 5, 'tricky_agreement.png')

# Group the data based on the uncertainty
threshold = np.median(generalized_variance)

hard = df[(generalized_variance >= threshold) & (df['pred_mean'] != df['label'])]
tricky = df[(generalized_variance < threshold) & (df['pred_mean'] != df['label'])]

plot_images(hard, 6, 'hard_gen_var.png')
plot_images(tricky, 2, 'tricky_gen_var.png')
