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
        
        # Plot the 32 most uncertain images
        _, axs = plt.subplots(6, 6, figsize=(14, 14))
        
        idx = np.argsort(df[conf_metric])[:32]
        group = df.iloc[idx]
        
        for i, ax in enumerate(axs.flatten()):
            try:
                filename = group.iloc[i, 0]
                label = lab_to_long[int_to_lab[group['label'].iloc[i]]]
                pred = lab_to_long[int_to_lab[group[pred_metric].iloc[i]]]
                ax.imshow(Image.open(filename).resize([224, 224]))
                ax.set_title(f'Label: {label}\nPred: {pred}\nFile: {os.path.basename(filename)}', fontsize=6, fontweight='bold')
            except IndexError:
                pass
            ax.axis('off')
        plt.subplots_adjust(hspace=0.3)
        plt.savefig(os.path.join(args.destination, f'most_uncertain_{conf_metric}.pdf'), bbox_inches='tight', dpi=300)
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
        plt.scatter(x, y, c=z, s=8)
        plt.plot(x, x_, color='black')
        plt.xlabel('Confidence')
        plt.ylabel('Variance')
        plt.savefig(os.path.join(args.destination, f'confidence_vs_variance_{conf_metric}.pdf'), dpi=300)
        plt.close()

    for uncertainty_metric in uncertainty_metrics:
        for pred_metric in pred_metrics:
            # plot uncertainty distribution
            plt.figure(figsize=(10, 5))
            y = df[df['label'] != df[pred_metric]][uncertainty_metric]
            x = df[df['label'] == df[pred_metric]][uncertainty_metric]
            bins = np.linspace(np.min(df[uncertainty_metric]), np.max(df[uncertainty_metric]), 20)
            plt.hist(x, bins=bins, alpha=0.85, label='Correct', density=True)
            plt.hist(y, bins=bins, alpha=0.85, label='Incorrect', density=True)
            plt.legend()
            plt.xlabel(x_labels[uncertainty_metric])
            plt.ylabel('Density')
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
        