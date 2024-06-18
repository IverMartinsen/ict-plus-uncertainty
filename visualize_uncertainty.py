import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import (
    compute_predictive_variance, 
    plot_images, 
    make_calibration_plots, 
    make_ordered_calibration_plot,
    plot_uncertainty
)

destination = 'stats/ensemble_stats'#'./ensemble_stats/'
path_to_df = os.path.join(destination, 'predictions.csv')
image_size = [224, 224]


if __name__ == '__main__':

    Y_pred = np.load(os.path.join(destination, 'predictions.npy'))
    df = pd.read_csv(path_to_df)

    # =============================================================================
    # UNCERTAINTY SCATTER PLOTS
    # =============================================================================


    x = df['predictive_variance']
    y = df['total_variance']

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.scatter(x, y, s=8)
    plt.xlabel('Variance for the predicted class')
    plt.ylabel('Total Variance for all classes')
    plt.savefig(os.path.join(destination, 'predictive_variance_vs_total_variance.pdf'), bbox_inches='tight', dpi=300)

    x = df['iqr'] + 1e-4
    x = np.log(x)
    y = df['loss']

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, np.log(y), c=df['label'] == df['pred_mean'])
    ax.set_ylabel('log (Loss)')
    ax.set_xlabel('log(IQR)')
    ax.axvline(x=np.quantile(x, 0.50), color='black', linestyle='--')
    ax.axvline(x=np.quantile(x, 0.75), color='black', linestyle='-.')
    ax.set_title('IQR vs Loss', fontsize=10, fontweight='bold')
    ax.legend(['correct', '0.50 quantile', '0.75 quantile'])
    plt.savefig(os.path.join(destination, 'iqr.png'), bbox_inches='tight', dpi=300)


    x = df['total_variance']
    y = df['loss']

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, y, c=df['label'] == df['pred_mean'])
    ax.set_ylabel('Loss')
    ax.set_xlabel('Total Variance')
    ax.axvline(x=np.quantile(x, 0.50), color='black', linestyle='--')
    ax.axvline(x=np.quantile(x, 0.75), color='black', linestyle='-.')
    ax.set_title('Total variance vs Loss', fontsize=10, fontweight='bold')
    ax.legend(['correct', '0.50 quantile', '0.75 quantile'])
    plt.savefig(os.path.join(destination, 'total_variance.png'), bbox_inches='tight', dpi=300)



    percentage = 0.75

    ep_cov, al_cov = compute_predictive_variance(Y_pred)
    cov = ep_cov + al_cov
    eigvals = np.sort(np.linalg.eigvals(cov), axis=1)[:, ::-1]

    x = cov[:, np.arange(4), np.arange(4)] # diagonal of the covariance matrix
    idx = np.argmax(Y_pred.mean(axis=1), axis=1) # get the index of the maximum prediction
    x = x[np.arange(len(df['label'])), idx] # get the diagonal of the covariance matrix at the maximum prediction

    # compute number of wrong predictions below the 75th percentile
    threshold = np.quantile(x, percentage)
    score = np.array((df['label'] != df['pred_mean'])[x < threshold]).sum()
    total_wrong = np.array(df['label'] != df['pred_mean']).sum()

    plot_uncertainty(
        x=x,
        loss=df['loss'],
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
        
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, np.log(df['loss']), c=df['label'] == df['pred_mean'])
    ax.set_ylabel('Loss')
    ax.set_xlabel('1 - Percentage Agreement')
    ax.axvline(x=threshold, color='black', linestyle='--')
    ax.set_title(f'Percentage Agree vs Loss ({int(score)}/{int(total_wrong)} below {percentage} quantile)', fontsize=10, fontweight='bold')
    ax.legend(['correct', f'{percentage} quantile'])
    plt.savefig(os.path.join(destination, 'percentage_agree.png'), bbox_inches='tight', dpi=300)

    x = df['epistemic_variance'].values.reshape(-1, 1)
    threshold = np.quantile(x, percentage)
    score = np.array((df['label'] != df['pred_mean'])[x.flatten() < threshold]).sum()

    plot_uncertainty(
        x=x,
        loss=df['loss'],
        c=df['label'] == df['pred_mean'],
        x_label='Epistemic Uncertainty',
        title=f'Epistemic Uncertainty vs Loss ({int(score)}/{int(total_wrong)} below {percentage} quantile)',
        destination=destination,
        quantile_percentage=percentage,
        filename='epistemic_uncertainty.png'
        )

    x = df['aleatoric_variance'].values.reshape(-1, 1)
    threshold = np.quantile(x, percentage)
    score = np.array((df['label'] != df['pred_mean'])[x.flatten() < threshold]).sum()

    plot_uncertainty(
        x=x,
        loss=df['loss'],
        c=df['label'] == df['pred_mean'],
        x_label='Aleatoric Uncertainty',
        title=f'Aleatoric Uncertainty vs Loss ({int(score)}/{int(total_wrong)} below {percentage} quantile)',
        destination=destination,
        quantile_percentage=percentage,
        filename='aleatoric_uncertainty.png'
        )    
    
    

    # =============================================================================
    # PLOT IMAGES
    # =============================================================================

    # group the data based on the agreement
    tricky = df[(df['agree'] == True) & (df['pred_mode'] != df['label'])]
    hard = df[(df['agree'] == False) & (df['pred_mode'] != df['label'])]

    plot_images(tricky, 3, 'tricky.png', image_size=image_size, destination=destination)
    plot_images(hard, 5, 'hard.png', image_size=image_size, destination=destination)

    # Plot the 25 most uncertain images
    idx = np.argsort(df['var_mean'])[-25:]
    plot_images(df.iloc[idx], 5, 'most_uncertain.png', image_size=image_size, destination=destination)

    #plot the 25 most certain images
    idx = np.argsort(df['var_mean'])[:25]
    plot_images(df.iloc[idx], 5, 'most_certain.png', image_size=image_size, destination=destination)
