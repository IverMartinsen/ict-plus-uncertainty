# Makes plot of the following:
#     - 32 most uncertain images
#     - Confidence vs Variance scatter plot
#     - Uncertainty distribution for correct and incorrect predictions
#     - Boxplot of uncertainty for each class
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import QuantileTransformer
from utils.utils import int_to_lab, lab_to_long

parser = argparse.ArgumentParser()
parser.add_argument("--destination", type=str, default='./results/ensemble_results/')
args = parser.parse_args()

df = pd.read_csv(os.path.join(args.destination, 'predictions.csv'))

uncertainty_metrics = [
    'entropy_median', 
    'entropy_mean', 
    'var_mean', 
    'weighted_confidence',
    'conf_mean',
    'conf_median',
    ]

pred_metrics = [
    'pred_median', 
    'pred_mean',
    ]

x_labels = {
    'entropy_median': 'Entropy', 
    'entropy_mean': 'Entropy', 
    'var_mean': 'Variance', 
    'weighted_confidence': 'Weighted Confidence',
    'conf_mean': 'Confidence',
    'conf_median': 'Confidence',
    }

if __name__ == '__main__':

    for pred_metric in pred_metrics:

        conf_metric = {'pred_mean': 'conf_mean', 'pred_median': 'conf_median'}[pred_metric]
        var_metric = {'pred_mean': 'var_mean', 'pred_median': 'var_median'}[pred_metric]
        
        lab_to_long = {'A': 'B. Agglutinated', 'B': 'B. Calcareous', 'S': 'Sediment', 'P': 'Planktic'}
        
        # Plot the 32 most uncertain images
        _, axs = plt.subplots(4, 8, figsize=(20, 10))
        
        idx = np.argsort(df[conf_metric])[:32]
        idx = np.sort(idx)
        group = df.iloc[idx]
        
        for i, ax in enumerate(axs.flatten()):
            try:
                filename = group.iloc[i, 0]
                label = lab_to_long[int_to_lab[group['label'].iloc[i]]]
                pred = lab_to_long[int_to_lab[group[pred_metric].iloc[i]]]
                ax.imshow(Image.open(filename).resize([224, 224]))
                ax.set_title(f'Label: {label}\nPred: {pred}\nFile: {os.path.basename(filename)}', fontsize=10, fontweight='bold')
                # add confidence in image
                ax.text(5, 15, f'{group[conf_metric].iloc[i]:.2f}', fontsize=10, fontweight='bold', color='white')
            except IndexError:
                pass
            ax.axis('off')
        plt.subplots_adjust(hspace=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.destination, f'most_uncertain_{conf_metric}.pdf'), bbox_inches='tight', dpi=300)
        plt.close()

        # visualize mistakes
        _, axs = plt.subplots(5, 5, figsize=(10, 10))
        
        idx = np.where(df['label'] != df[pred_metric])[0]
        idx = np.sort(idx)
        group = df.iloc[idx]
        
        for i, ax in enumerate(axs.flatten()):
            try:
                filename = group.iloc[i, 0]
                label = lab_to_long[int_to_lab[group['label'].iloc[i]]]
                pred = lab_to_long[int_to_lab[group[pred_metric].iloc[i]]]
                ax.imshow(Image.open(filename).resize([224, 224]))
                ax.set_title(f'Label: {label}\nPred: {pred}\nFile: {os.path.basename(filename)}', fontsize=10, fontweight='bold')
                # add confidence in image
                ax.text(5, 18, f'{group[conf_metric].iloc[i]:.2f}', fontsize=10, fontweight='bold', color='white')
            except IndexError:
                pass
            ax.axis('off')
        plt.subplots_adjust(hspace=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.destination, f'mistakes_{conf_metric}.pdf'), bbox_inches='tight', dpi=300)
        plt.close()        
        
        # plot the confidence vs variance
        x = df[conf_metric]
        x = np.array(x)
        x = x.reshape(-1, 1)
        qt = QuantileTransformer()
        x = qt.fit_transform(x)
        x = x.flatten()

        y = df[var_metric]
        y = np.array(y)
        y = y.reshape(-1, 1)
        qt = QuantileTransformer()
        y = qt.fit_transform(y)
        y = y.flatten()
        
        z = df['label'] == df[pred_metric]

        # fit y as a function of x
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        x_ = model.predict(x.reshape(-1, 1))

        plt.figure(figsize=(10, 5))
        plt.scatter(x, y, c=z, s=30, label='Correct predictions')
        plt.plot(x, x_, color='black', linestyle='--', linewidth=1)
        plt.xlabel('Confidence', fontsize=20)
        plt.ylabel('Variance', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.destination, f'confidence_vs_variance_{conf_metric}.pdf'), dpi=300)
        plt.close()

    for uncertainty_metric in uncertainty_metrics:
        for pred_metric in pred_metrics:
            # plot uncertainty distribution
            #plt.figure(figsize=(10, 5))
            fig, axs = plt.subplots(2, 1, figsize=(14, 7), height_ratios=[2, 1], sharex=True)
            y = df[df['label'] != df[pred_metric]][uncertainty_metric]
            x = df[df['label'] == df[pred_metric]][uncertainty_metric]
            bins = np.linspace(np.min(df[uncertainty_metric]), np.max(df[uncertainty_metric]), 20)
            #plt.hist(x, bins=bins, alpha=0.85, label='Correct', density=True)
            axs[0].hist(x, bins=bins, alpha=1.00, label='Correct', density=True)
            #plt.hist(y, bins=bins, alpha=0.85, label='Incorrect', density=True)
            plt.xticks(fontsize=15)
            axs[1].hist(y, bins=bins, alpha=1.00, label='Incorrect', density=True, color='tab:orange')
            #plt.hist([x, y], bins=bins, label=['Correct', 'Incorrect'], density=True)
            plt.xticks(fontsize=15)
            #plt.legend()
            plt.xlabel(x_labels[uncertainty_metric], fontsize=15)
            axs[0].set_ylabel('Density', fontsize=15)
            axs[1].set_ylabel('Density', fontsize=15)
            #axs[0].set_xticks(fontsize=15)
            #axs[1].set_xticks(fontsize=15)
            axs[0].set_yticks([], fontsize=15)
            axs[1].set_yticks([], fontsize=15)
            axs[0].legend(fontsize=15)
            axs[1].legend(fontsize=15)
            plt.tight_layout()
            plt.savefig(os.path.join(args.destination, f'{uncertainty_metric}_{pred_metric}_distribution.pdf'), dpi=300)
            plt.close()

        # plot the boxplot
        fig, ax = plt.subplots(figsize=(10, 5))
        df['Group'] = df['label'].apply(lambda x: int_to_lab[x])
        df.boxplot(column=uncertainty_metric, by='Group', ax=ax)
        ax.set_title('')
        plt.suptitle('')
        plt.savefig(os.path.join(args.destination, f'{uncertainty_metric}_boxplot.png'), bbox_inches='tight', dpi=300)
        plt.close()
        
        # plot num mistakes vs uncertainty
        num_mistakes = np.zeros(len(df))
        for i in range(10):
            num_mistakes += (df[f'model_{i}'].values != df['label']).astype(int)
        
        markers = ['o', 'x', 's', '>']
        
        x = num_mistakes + np.random.normal(0, 0.1, len(df))
        y = df[uncertainty_metric] + np.random.normal(0, 0.01, len(df))
        
        plt.figure(figsize=(10, 5))
        for i, label in enumerate(df['label'].unique()):
            idx = df['label'] == label
            plt.scatter(x[idx], y[idx], s=15, label=lab_to_long[int_to_lab[label]], marker=markers[i])
        plt.xlabel('Number of mistakes')
        plt.ylabel(x_labels[uncertainty_metric])
        plt.legend()
        plt.savefig(os.path.join(args.destination, f'num_mistakes_vs_{uncertainty_metric}.pdf'), dpi=300)
        plt.close()