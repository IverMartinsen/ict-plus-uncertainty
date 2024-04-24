import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from utils import (
    lab_to_int, lab_to_long, make_dataset, load_data, compute_predictive_variance, plot_images
    )

# hyperparameters
path_to_model = './ensemble/20240312_155425.keras'
destination = './dropout_stats/'
image_size = [224, 224]
batch_size = 32
num_samples = 10

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

tf.random.set_seed(1)
np.random.seed(1)
random.seed(1)

for i in tqdm(range(num_samples)):
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
eigvals = np.sort(np.linalg.eigvals(cov), axis=1)[:, ::-1]

y = (df['label'] != df['pred_mean'])
log_model = LogisticRegression(class_weight='balanced')

for i in range(4):
    #logistic regression
    x = np.log(eigvals[:, i].reshape(-1, 1))
    try:
        log_model.fit(x, y)
        score = f1_score(y, log_model.predict(x))
    except ValueError:
        score = 0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, np.log(df['loss']), c=df['label'] == df['pred_mean'])
    ax.set_ylabel('Loss')
    ax.set_xlabel(r'$\lambda_{}$'.format(i+1))
    ax.set_title(f'Eigenvalue {i+1} vs Loss (F1-score: {score:.2f})', fontsize=10, fontweight='bold')
    ax.legend(['correct', 'wrong'])
    plt.savefig(os.path.join(destination, f'eigenvalue_{i+1}.png'), bbox_inches='tight', dpi=300)
    
    if i == 0:
        continue
    
    x = np.log(np.prod(eigvals[:, :i+1], axis=1).reshape(-1, 1))
    try:
        log_model.fit(x, y)
        score = f1_score(y, log_model.predict(x))
    except ValueError:
        score = 0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, np.log(df['loss']), c=df['label'] == df['pred_mean'])
    ax.set_ylabel('Loss')
    ax.set_xlabel(r'$\prod \lambda_j$'.format(i+1))
    ax.set_title(f'Product of first {i+1} eigenvalues vs Loss (F1-score: {score:.2f})', fontsize=10, fontweight='bold')
    ax.legend(['correct', 'wrong'])
    plt.savefig(os.path.join(destination, f'product_eigenvalue_{i+1}.png'), bbox_inches='tight', dpi=300)
    
    x = np.log(np.sum(eigvals[:, :i+1], axis=1).reshape(-1, 1))
    log_model = LogisticRegression(class_weight='balanced')
    log_model.fit(x, y)
    score = f1_score(y, log_model.predict(x))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, np.log(df['loss']), c=df['label'] == df['pred_mean'])
    ax.set_ylabel('Loss')
    ax.set_xlabel(r'$\sum \lambda_j$'.format(i+1))
    ax.set_title(f'Sum of first {i+1} eigenvalues vs Loss (F1-score: {score:.2f})', fontsize=10, fontweight='bold')
    ax.legend(['correct', 'wrong'])
    plt.savefig(os.path.join(destination, f'sum_eigenvalue_{i+1}.png'), bbox_inches='tight', dpi=300)    

# group the data based on the agreement
hard = df[(df['agree'] == False) & (df['pred_mode'] != df['label'])]
tricky = df[(df['agree'] == True) & (df['pred_mode'] != df['label'])]

plot_images(hard, 5, 'hard_agreement.png', destination)
plot_images(tricky, 5, 'tricky_agreement.png', destination)

# Group the data based on the uncertainty
eig_prod2 = np.prod(eigvals[:, :2], axis=1)
threshold = np.median(eig_prod2)

hard = df[(eig_prod2 >= threshold) & (df['pred_mean'] != df['label'])]
tricky = df[(eig_prod2 < threshold) & (df['pred_mean'] != df['label'])]

plot_images(hard, 6, 'hard_eig_prod_2.png', destination)
plot_images(tricky, 2, 'tricky_eig_prod_2.png', destination)
