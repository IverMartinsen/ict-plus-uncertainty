import os
import glob
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from utils import (
    lab_to_int, 
    lab_to_long, 
    make_dataset, 
    load_data, 
    compute_predictive_variance, 
    plot_images, 
    make_calibration_plots, 
    make_ordered_calibration_plot,
    store_predictions,
    store_confusion_matrix,
    store_summary_stats,
    plot_uncertainty
)
from optimizer import StochasticGradientLangevinDynamics

# hyperparameters
destination = 'stats/tta_stats_2'#'./ensemble_stats/'
image_size = [224, 224]

os.makedirs(destination, exist_ok=True)

Y_pred = np.load('tta/y_pred_2.npy')
y_val = np.load('tta/labs.npy')
X_val = np.load('tta/x_val_2.npy')

# =============================================================================
# STATISTICS
# =============================================================================

df = store_predictions(Y_pred, y_val, X_val, destination)

store_confusion_matrix(df['label'], df['pred_mean'], destination)

store_summary_stats(df, destination)

# class wise accuracy
class_wise_accuracy = classification_report(df['label'], df['pred_mean'], target_names=list(lab_to_long.values()), output_dict=True)
class_wise_df = pd.DataFrame(class_wise_accuracy).T
class_wise_df.to_csv(os.path.join(destination, 'class_wise_accuracy.csv'))

# =============================================================================
# UNCERTAINTY SCATTER PLOTS
# =============================================================================

percentage = 0.75

ep_cov, al_cov = compute_predictive_variance(Y_pred)
cov = ep_cov + al_cov
eigvals = np.sort(np.linalg.eigvals(cov), axis=1)[:, ::-1]

x = cov[:, np.arange(4), np.arange(4)] # diagonal of the covariance matrix
idx = np.argmax(Y_pred.mean(axis=1), axis=1) # get the index of the maximum prediction
x = x[np.arange(len(y_val)), idx] # get the diagonal of the covariance matrix at the maximum prediction

# compute number of wrong predictions below the 75th percentile
threshold = np.quantile(x, percentage)
score = np.array((df['label'] != df['pred_mean'])[x < threshold]).sum()
total_wrong = np.array(df['label'] != df['pred_mean']).sum()

plot_uncertainty(
    x=x,
    loss=np.log(df['loss']),
    c=df['label'] == df['pred_mean'],
    x_label='Predictive Variance',
    title=f'Predictive Variance vs Loss ({int(score)}/{int(total_wrong)} below {percentage} quantile)',
    destination=destination,
    quantile_percentage=percentage,
    filename='predictive_variance.png'
    )

x = df['percentage_agree'].values.reshape(-1, 1)
x = 1 - x
threshold = np.quantile(x, percentage)
score = np.array((df['label'] != df['pred_mean'])[x.flatten() < threshold]).sum()
x += np.random.normal(0, 0.01, x.shape)

plot_uncertainty(
    x=x,
    loss=np.log(df['loss']),
    c=df['label'] == df['pred_mean'],
    x_label='1 - Percentage Agreement',
    title=f'Percentage Agree vs Loss ({int(score)}/{int(total_wrong)} below {percentage} quantile)',
    destination=destination,
    quantile_percentage=percentage,
    filename='percentage_agree.png'
    )

for i in range(4):

    x = eigvals[:, i].reshape(-1, 1)
    threshold = np.quantile(x, percentage)
    score = np.array((df['label'] != df['pred_mean'])[x.flatten() < threshold]).sum()
    
    plot_uncertainty(
        x=x,
        loss=np.log(df['loss']),
        c=df['label'] == df['pred_mean'],
        x_label=r'$\lambda_{}$'.format(i+1),
        title=f'Eigenvalue {i+1} vs Loss ({int(score)}/{int(total_wrong)} below {percentage} quantile)',
        destination=destination,
        quantile_percentage=percentage,
        filename=f'eigenvalue_{i+1}.png'
        )
    
    if i == 0:
        continue
    
    x = np.prod(eigvals[:, :i+1], axis=1).reshape(-1, 1)
    x -= np.min(x)
    x += 1e-24
    x = np.log(x)
    threshold = np.quantile(x, percentage)
    score = np.array((df['label'] != df['pred_mean'])[x.flatten() < threshold]).sum()
        
    plot_uncertainty(
        x=x,
        loss=np.log(df['loss']),
        c=df['label'] == df['pred_mean'],
        x_label=r'$\prod \lambda_j$'.format(i+1),
        title=f'Product of first {i+1} eigenvalues vs Loss ({int(score)}/{int(total_wrong)} below {percentage} quantile)',
        destination=destination,
        quantile_percentage=percentage,
        filename=f'product_eigenvalue_{i+1}.png'
        )
    
    x = np.sum(eigvals[:, :i+1], axis=1).reshape(-1, 1)
    x -= np.min(x)
    x += 1e-24
    x = np.log(x)
    threshold = np.quantile(x, percentage)
    score = np.array((df['label'] != df['pred_mean'])[x.flatten() < threshold]).sum()
    
    plot_uncertainty(
        x=x, 
        loss=np.log(df['loss']), 
        c=df['label'] == df['pred_mean'], 
        x_label=r'$\sum \lambda_j$'.format(i+1), 
        title=f'Sum of first {i+1} eigenvalues vs Loss ({int(score)}/{int(total_wrong)} below {percentage} quantile)', 
        destination=destination,
        quantile_percentage=percentage,
        filename=f'sum_eigenvalue_{i+1}.png'
        )

make_calibration_plots(Y_pred.mean(axis=1), y_val, destination, num_bins=9)
make_ordered_calibration_plot(Y_pred.mean(axis=1), y_val, destination, num_bins=20)

# =============================================================================
# PLOT IMAGES
# =============================================================================

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

# Plot the 25 most uncertain images
idx = np.argsort(df['var_mean'])[-25:]
plot_images(df.iloc[idx], 5, 'most_uncertain.png', image_size=image_size, destination=destination)

#plot the 25 most certain images
idx = np.argsort(df['var_mean'])[:25]
plot_images(df.iloc[idx], 5, 'most_certain.png', image_size=image_size, destination=destination)
