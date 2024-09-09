# Lots of boilerplate code to visualize the uncertainty of the model
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import QuantileTransformer
from utils import (
    plot_images, 
    plot_uncertainty,
    int_to_lab,
    lab_to_long,
)

destination = 'stats/ensemble_stats/'
path_to_df = os.path.join(destination, 'predictions.csv')
image_size = [224, 224]

path_to_expert = 'results/expert_results/predictions.csv'
df_expert = pd.read_csv(path_to_expert)

q_val = 0.85

if __name__ == '__main__':

    Y_pred = np.load(os.path.join(destination, 'predictions.npy'))
    df = pd.read_csv(path_to_df)

    # Plot the 32 most uncertain images
    _, axs = plt.subplots(6, 6, figsize=(14, 14))
    
    idx = np.argsort(df['conf_mean'])[:32]
    group = df.iloc[idx]
    
    eu = 0
    
    for i, ax in enumerate(axs.flatten()):
        try:
            filename = group.iloc[i, 0]
            label = lab_to_long[int_to_lab[group['label'].iloc[i]]]
            pred = lab_to_long[int_to_lab[group['pred_mode'].iloc[i]]]
            expert_uncertainty = df_expert[df_expert['filename'] == os.path.basename(filename)]['mode_weights'].values[0]
            eu += expert_uncertainty
            ax.imshow(Image.open(filename).resize(image_size))
            ax.set_title(f'Label: {label}\nPred: {pred}\nFile: {os.path.basename(filename)}\nExpert Uncertainty: {expert_uncertainty:.2f}', fontsize=6, fontweight='bold')
        except IndexError:
            pass
        ax.axis('off')
    plt.suptitle(f'Average expert uncertainty: {eu / 25:.2f}', fontsize=10, fontweight='bold')
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(os.path.join(destination, 'most_uncertain.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # plot confidence vs variance
    x = df['conf_mean']
    x = np.array(x)
    #x = x.reshape(-1, 1)
    #qt = QuantileTransformer()
    #x = qt.fit_transform(x)
    x = x.flatten()

    y = df['var_mean']
    y = np.array(y)
    #y = y.reshape(-1, 1)
    #qt = QuantileTransformer()
    #y = qt.fit_transform(y)
    y = y.flatten()
    
    z = df['label'] == df['pred_mean']

    # fit y as a function of x
    #model = LinearRegression()
    #model.fit(x.reshape(-1, 1), y)
    #x_ = model.predict(x.reshape(-1, 1))

    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, c=z, s=8)
    #plt.plot(x, x_, color='black')
    plt.xlabel('Confidence')
    plt.ylabel('Variance')
    plt.savefig(os.path.join(destination, f'confidence_vs_variance_mean.pdf'), bbox_inches='tight', dpi=300)

    x = df['conf_mean']
    y = df['var_mean']
    # standardize the data by label
    for i in range(4):
        idx = df['label'] == i
        x.loc[idx] = (x.loc[idx] - x.loc[idx].mean()) / x.loc[idx].std()
        y.loc[idx] = (y.loc[idx] - y.loc[idx].mean()) / y.loc[idx].std()
    x = x.values.reshape(-1, 1)
    #qt = QuantileTransformer()
    #x = qt.fit_transform(x)
    x = x.flatten()

    y = y.values.reshape(-1, 1)
    #qt = QuantileTransformer()
    #y = qt.fit_transform(y)
    y = y.flatten()
    
    z = df['label'] == df['pred_mean']

    # fit y as a function of x
    #model = LinearRegression()
    #model.fit(x.reshape(-1, 1), y)
    #x_ = model.predict(x.reshape(-1, 1))

    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, c=z, s=8)
    #plt.plot(x, x_, color='black')
    plt.xlabel('Confidence')
    plt.ylabel('Variance')
    plt.savefig(os.path.join(destination, f'confidence_vs_variance_standardized.pdf'), bbox_inches='tight', dpi=300)

    x = df['conf_mean']
    y = df['var_mean']
    z = df['label'] == df['pred_mean']
    for i in range(4):
        x_ = x[df['label'] == i]
        y_ = y[df['label'] == i]
        z_ = z[df['label'] == i]
        x_ = x_.values.reshape(-1, 1)
        #qt = QuantileTransformer()
        #x_ = qt.fit_transform(x_)
        x_ = x_.flatten()
        y_ = y_.values.reshape(-1, 1)
        #qt = QuantileTransformer()
        #y_ = qt.fit_transform(y_)
        y_ = y_.flatten()
        #model = LinearRegression()
        #model.fit(x_.reshape(-1, 1), y_)
        #x__ = model.predict(x_.reshape(-1, 1))
        plt.figure(figsize=(10, 5))
        plt.scatter(x_, y_, c=z_, s=8)
        #plt.plot(x_, x__, color='black')
        plt.xlabel('Confidence')
        plt.ylabel('Variance')
        plt.savefig(os.path.join(destination, f'confidence_vs_variance_standardized_{int_to_lab[i]}.pdf'), bbox_inches='tight', dpi=300)
        

    
    # =============================================================================
    # TOTAL VARIANCE AGAINTS PREDICTIVE VARIANCE
    # =============================================================================
    #x = df['predictive_variance']
    #y = df['total_variance']

    #fig, ax = plt.subplots(figsize=(10, 5))
    #plt.scatter(x, y, s=8)
    #plt.xlabel('Variance for the predicted class')
    #plt.ylabel('Total Variance for all classes')
    #plt.savefig(os.path.join(destination, 'predictive_variance_vs_total_variance.pdf'), bbox_inches='tight', dpi=300)
    # =============================================================================
    # LOSS AGAINST IQR
    # =============================================================================
    #x = df['iqr'] #+ 1e-4
    #x = np.log(x)
    #y = df['loss']

    #fig, ax = plt.subplots(figsize=(10, 5))
    #ax.scatter(x, y, c=df['label'] == df['pred_mean'])
    #ax.set_ylabel('Loss')
    #ax.set_xlabel('IQR')
    #ax.axvline(x=np.quantile(x, 0.50), color='black', linestyle='--')
    #ax.axvline(x=np.quantile(x, q_val), color='black', linestyle='-.')
    #ax.set_title('IQR vs Loss', fontsize=10, fontweight='bold')
    #ax.legend(['correct', '0.50 quantile', f'{q_val} quantile'])
    #plt.savefig(os.path.join(destination, 'iqr.png'), bbox_inches='tight', dpi=300)
    # =============================================================================
    # LOSS AGAINST TOTAL VARIANCE
    # =============================================================================
    #x = df['total_variance']
    #y = df['loss']

    #fig, ax = plt.subplots(figsize=(10, 5))
    #ax.scatter(x, y, c=df['label'] == df['pred_mean'])
    #ax.set_ylabel('Loss')
    #ax.set_xlabel('Total Variance')
    #ax.axvline(x=np.quantile(x, 0.50), color='black', linestyle='--')
    #ax.axvline(x=np.quantile(x, q_val), color='black', linestyle='-.')
    #ax.set_title('Total variance vs Loss', fontsize=10, fontweight='bold')
    #ax.legend(['correct', '0.50 quantile', f'{q_val} quantile'])
    #plt.savefig(os.path.join(destination, 'total_variance.png'), bbox_inches='tight', dpi=300)
    # =============================================================================
    # LOSS AGAINST PREDICTIVE VARIANCE
    # =============================================================================
    #x = df['predictive_variance']
    #y = df['loss']

    #plot_uncertainty(
    #    x=x,
    #    loss=df['loss'],
    #    c=df['label'] == df['pred_mean'],
    #    x_label='Predictive Variance',
    #    title=f'Predictive Variance vs Loss',
    #    destination=destination,
    #    filename='predictive_variance.png',
    #    q_val=q_val
    #    )
    # =============================================================================
    # LOSS AGAINST PERCENTAGE AGREE
    # =============================================================================
    #x = df['percentage_agree'].values.reshape(-1, 1)
    #x = 1 - x
    #x += np.random.normal(0, 0.01, x.shape)
        
    #fig, ax = plt.subplots(figsize=(10, 5))
    #ax.scatter(x, np.log(df['loss']), c=df['label'] == df['pred_mean'])
    #ax.set_ylabel('Loss')
    #ax.set_xlabel('1 - Percentage Agreement')
    #ax.axvline(x=np.quantile(x, 0.50), color='black', linestyle='--')
    #ax.axvline(x=np.quantile(x, q_val), color='black', linestyle='-.')
    #ax.set_title(f'Percentage Agree vs Loss', fontsize=10, fontweight='bold')
    #ax.legend(['correct', '0.50 quantile', f'{q_val} quantile'])
    #plt.savefig(os.path.join(destination, 'percentage_agree.png'), bbox_inches='tight', dpi=300)
    # =============================================================================
    # LOSS AGAINST EPISTEMIC UNCERTAINTY
    # =============================================================================
    #x = df['epistemic_variance'].values.reshape(-1, 1)

    #plot_uncertainty(
    #    x=x,
    #    loss=df['loss'],
    #    c=df['label'] == df['pred_mean'],
    #    x_label='Epistemic Uncertainty',
    #    title=f'Epistemic Uncertainty vs Loss',
    #    destination=destination,
    #    filename='epistemic_uncertainty.png',
    #    q_val=q_val
    #    )
    # =============================================================================
    # LOSS AGAINST ALEATORIC UNCERTAINTY
    # =============================================================================
    #x = df['aleatoric_variance'].values.reshape(-1, 1)

    #plot_uncertainty(
    #    x=x,
    #    loss=df['loss'],
    #    c=df['label'] == df['pred_mean'],
    #    x_label='Aleatoric Uncertainty',
    #    title=f'Aleatoric Uncertainty vs Loss',
    #    destination=destination,
    #    filename='aleatoric_uncertainty.png',
    #    q_val=q_val
    #    )    

    # =============================================================================
    # BOX PLOTS
    # =============================================================================
    df = pd.read_csv(path_to_df)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    df['Group'] = df['label'].apply(lambda x: int_to_lab[x])
    df.boxplot(column='conf_mean', by='Group', ax=ax)
    ax.set_title('')
    plt.suptitle('')
    plt.savefig(os.path.join(destination, 'conf_mean_boxplot.png'), bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(figsize=(10, 5))
    df['Group'] = df['label'].apply(lambda x: int_to_lab[x])
    df.boxplot(column='var_mean', by='Group', ax=ax)
    ax.set_title('')
    plt.suptitle('')
    plt.savefig(os.path.join(destination, 'var_mean_boxplot.png'), bbox_inches='tight', dpi=300)

    # =============================================================================
    # UNCERTAINTY DISTRIBUTION
    # =============================================================================
    plt.figure(figsize=(10, 5))
    x = df[df['label'] == df['pred_mean']]['conf_mean']
    y = df[df['label'] != df['pred_mean']]['conf_mean']
    bins = np.linspace(np.min(df['conf_mean']), np.max(df['conf_mean']), 20)
    plt.hist(x, bins=bins, alpha=0.85, label='Correct', density=True)
    plt.hist(y, bins=bins, alpha=0.85, label='Incorrect', density=True)
    plt.legend()
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.title('Confidence distribution')
    plt.savefig(os.path.join(destination, 'confidence_distribution.pdf'), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    x = df[df['label'] == df['pred_mean']]['var_mean']
    y = df[df['label'] != df['pred_mean']]['var_mean']
    bins = np.linspace(np.min(df['var_mean']), np.max(df['var_mean']), 20)
    plt.hist(x, bins=bins, alpha=0.85, label='Correct', density=True)
    plt.hist(y, bins=bins, alpha=0.85, label='Incorrect', density=True)
    plt.legend()
    plt.xlabel('Variance')
    plt.ylabel('Density')
    plt.title('Variance distribution')
    plt.savefig(os.path.join(destination, 'variance_distribution.pdf'), dpi=300)
