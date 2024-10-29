import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("--destination", type=str, default='./results/ensemble_results/')
parser.add_argument("--key", type=str, default='mean')
args = parser.parse_args()


key_pred, key_uncertainty = {'mean': ('pred_mean', 'conf_mean'), 'median': ('pred_median', 'conf_median')}[args.key]


if __name__ == '__main__':

    df = pd.read_csv(os.path.join(args.destination, 'predictions.csv'))

    x = 1 - df[key_uncertainty]
    y = df['label'] != df[key_pred]    
    
    precision, recall, _ = precision_recall_curve(y, x)
    
    pd.DataFrame({'precision': precision, 'recall': recall}).to_csv(os.path.join(args.destination, 'precision_recall.csv'))
    
    pr_auc = auc(recall, precision)
    
    roc_auc = roc_auc_score(y, x)
    
    fpr, tpr, _ = roc_curve(y, x)
    
    # plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})', linewidth=3)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=3)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Receiver Operating Characteristic', fontsize=20)
    plt.legend(loc='lower right', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(args.destination, 'roc_curve.png'))
    plt.close()
    
    # plot precision-recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})', linewidth=3)
    plt.xlabel('Recall (Detected mistakes)', fontsize=20)
    plt.ylabel('Precision (Correctly detected mistakes)', fontsize=20)
    plt.title('Precision-Recall curve', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='upper right', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(args.destination, 'pr_curve.png'))
    plt.close()
    
    auc_summary = {'AUPR (wrong)': [pr_auc], 'AUROC (wrong)': [roc_auc]}
    
    # turn it around
    x = 1 - x
    y = y == 0
    
    precision, recall, _ = precision_recall_curve(y, x)
    
    auc_summary['AUPR (correct)'] = [auc(recall, precision)]
    auc_summary['AUROC (correct)'] = [roc_auc_score(y, x)]
    
    pd.DataFrame(auc_summary).to_csv(os.path.join(args.destination, 'aupr_auroc.csv'))
    