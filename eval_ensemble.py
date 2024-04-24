import os
import glob
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
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
tricky = df[(df['agree'] == True) & (df['pred_mode'] != df['label'])]
hard = df[(df['agree'] == False) & (df['pred_mode'] != df['label'])]

plot_images(tricky, 3, 'tricky.png', image_size=image_size, destination=destination)
plot_images(hard, 5, 'hard.png', image_size=image_size, destination=destination)

# Group the data based on the uncertainty
eig1 = np.log(eigvals[:, 0])
threshold = np.median(eig1)
tricky = df[(eig1 < threshold) & (df['pred_mean'] != df['label'])]
hard = df[(eig1 >= threshold) & (df['pred_mean'] != df['label'])]

plot_images(tricky, 2, 'tricky_first_eig.png', image_size=image_size, destination=destination)
plot_images(hard, 5, 'hard_first_eig.png', image_size=image_size, destination=destination)
