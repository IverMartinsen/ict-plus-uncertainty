import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import classification_report
from utils import (
    lab_to_int, 
    lab_to_long, 
    make_dataset, 
    load_data, 
    compute_predictive_variance, 
    plot_images, 
    store_predictions, 
    store_confusion_matrix, 
    store_summary_stats,
    plot_uncertainty,
    make_calibration_plots,
    make_ordered_calibration_plot,
    )
from optimizer import StochasticGradientLangevinDynamics

# hyperparameters
path_to_model = './ensemble/ensemble/20240606_122403.keras'
destination = './stats/dropout_stats/'
image_size = [224, 224]
batch_size = 32
num_samples = 10

os.makedirs(destination, exist_ok=True)

# load the model
model = tf.keras.models.load_model(path_to_model, custom_objects={'pSGLangevinDynamics': StochasticGradientLangevinDynamics})
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
ds_val = make_dataset(X_val, y_val, image_size, batch_size, seed=1)

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
# STATISTICS
# =============================================================================

# create a dataframe to store the predictions
df = store_predictions(Y_pred, y_val, X_val, destination)
# confusion matrix
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

# Plot the 25 most uncertain images
idx = np.argsort(df['var_mean'])[-25:]
plot_images(df.iloc[idx], 5, 'most_uncertain.png', image_size=image_size, destination=destination)

#plot the 25 most certain images
idx = np.argsort(df['var_mean'])[:25]
plot_images(df.iloc[idx], 5, 'most_certain.png', image_size=image_size, destination=destination)
