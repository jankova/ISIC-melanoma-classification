# basic imports
import numpy as np
import pandas as pd
import math
import numpy as np

# sklearn
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# tensorflow
import tensorflow as tf

# plotting
import matplotlib.pyplot as plt

# custom imports
from dataset import get_dataset
import config




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
    
    # load validation data
    train_files_nums, valid_files_nums = train_test_split(np.arange(0,15), test_size = 0.2, random_state = 0)
    
    #files_train = [config.TRAIN_PATH + 'train'+ str(x).zfill(2) + '*.tfrec' for x in train_files_nums]
    files_valid = [config.TRAIN_PATH + 'train' + str(x).zfill(2) + '*.tfrec' for x in valid_files_nums]
    
    #TRAINING_FILENAMES = tf.io.gfile.glob(files_train)  
    VALID_FILENAMES = tf.io.gfile.glob(files_valid)
    
    val_ds_tf = get_dataset(VALID_FILENAMES,
                            augment=None, 
                            shuffle=False,
                            repeat=False, 
                            dim=config.IMG_HEIGHT)
    
    
    test_labels = []
    
    for example, label in val_ds_tf:
        test_labels.extend(np.array(label))
             
    print(np.sum(test_labels))
        
    # read in test labels

    #df_test_labels = pd.read_csv('data/processed/test_labels.csv')

    #test_labels = df_test_labels['labels']

    # read in ensembled labels
    
    df_ens_probs = pd.read_csv('results/predictions/predictions-EfficientNet.csv')
    
    ens_probs = df_ens_probs['preds']
    
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
    