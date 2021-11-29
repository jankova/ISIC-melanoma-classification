#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:36:45 2021

@author: janajankova
"""

import numpy as np
import pandas as pd
import math
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from dataset import get_data_gen
import matplotlib.pyplot as plt


def plot_roc(test_labels, pred_probs):
    
    #pred_classes = [1 if val > 0.5 else 0 for val in predictions ]
    fp, tp, _ = roc_curve(test_labels, pred_probs)
    plt.plot(fp*100, tp*100)
    plt.xlabel("FP rate")
    plt.ylabel("TP rate")
    plt.savefig('roc_curve.jpg')
    plt.show()


def eval_net():
    
    # read in validation data
    
    _, val_ds = get_data_gen()
    
    # read in test labels

    df_test_labels = pd.read_csv('evaluate/test_labels.csv')

    test_labels = df_test_labels['labels']

    # read in ensembled labels
    
    df_ens_probs = pd.read_csv('results/predictions/ensembled_probs.csv')
    
    ens_probs = df_ens_probs['probs']
    
    pred_labels = [1 if prob > 0.5 else 0 for prob in ens_probs]
    
    # compute accuracy

    acc = accuracy_score(test_labels, pred_labels)
    AUC = roc_auc_score(test_labels, ens_probs)

    print(confusion_matrix(test_labels, pred_labels))
    print(f'AUC score: {AUC}')
    print(f'Accuracy: {acc}')
  
    # Comparing ROC curves, AUC scores

    plot_roc(test_labels, ens_probs)



if __name__ == '__main__':
    
    eval_net()
    