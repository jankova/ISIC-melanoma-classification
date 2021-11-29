import numpy as np
import pandas as pd
import math
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from dataset import get_data_gen
import matplotlib.pyplot as plt


def plot_roc(test_labels, pred_probs):
    
    AUC = roc_auc_score(test_labels, pred_probs)
    
    fp, tp, _ = roc_curve(test_labels, pred_probs)
    plt.plot(fp*100, tp*100, label = f"AUC score: {AUC}")
    plt.legend(loc = 'lower right')
    plt.xlabel("FP rate")
    plt.ylabel("TP rate")
    plt.savefig('results/plots/roc_curve.jpg')
    plt.show()


def eval_net():
    
    # read in validation data
    
    _, val_ds = get_data_gen()
    
    # read in test labels

    df_test_labels = pd.read_csv('data/processed/test_labels.csv')

    test_labels = df_test_labels['labels']

    # read in ensembled labels
    
    df_ens_probs = pd.read_csv('results/predictions/ensembled_probs.csv')
    
    ens_probs = df_ens_probs['probs']
    
    pred_labels = [1 if prob > 0.5 else 0 for prob in ens_probs]
    
    # compute accuracy

    acc = accuracy_score(test_labels, pred_labels)
    AUC = roc_auc_score(test_labels, ens_probs)

    print("\nConfusion matrix:")
    print(confusion_matrix(test_labels, pred_labels))
    print(f'\nAUC score: {AUC}')
    print(f'\nAccuracy: {acc}')
    print('\nClassification report:')
    print(classification_report(test_labels, pred_labels))
  
    # plot ROC curve, AUC score

    plot_roc(test_labels, ens_probs)



if __name__ == '__main__':
    
    eval_net()
    