# basic imports
import numpy as np
import pandas as pd
import math
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

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

def parse_args():
    
    parser = ArgumentParser()
    parser.add_argument('--enet-type', type = str, required = True)
    
    args, _ = parser.parse_known_args()
    
    return args


def plot_roc(test_labels, pred_probs):
    
    AUC = roc_auc_score(test_labels, pred_probs)
    
    fpr, tpr, thresholds = roc_curve(test_labels, pred_probs)
    plt.plot(fpr*100, tpr*100, label = f"AUC score: {AUC}")
    plt.legend(loc = 'lower right')
    plt.xlabel("FP rate")
    plt.ylabel("TP rate")
    plt.savefig('results/plots/roc_curve.jpg')
    plt.show()
    
    opt_threshold = thresholds[np.argmax(tpr - fpr)]
    
    return opt_threshold


def eval_net():
    
    args = parse_args()
    
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
    
    
    
    df_ens_probs = pd.read_csv('results/predictions/predictions-valid-' + args.enet_type + '.csv') #valid
    
    ens_probs = df_ens_probs['target']#target
    true_labels = df_ens_probs['true_labels']
    
    
    # compute auc
    
    AUC = roc_auc_score(true_labels, ens_probs)

  
    # plot ROC curve, AUC score

    opt_threshold = plot_roc(true_labels, ens_probs)
    
    print(f"Optimal threshold: {opt_threshold}")
    
    pred_labels = [1 if prob > 0.7 else 0 for prob in ens_probs]
    acc = accuracy_score(true_labels, pred_labels)
    
    print("sum_labels", np.mean(pred_labels))
    
    print("\nConfusion matrix:")
    print(confusion_matrix(true_labels, pred_labels))
    print(f'\nAUC score: {AUC}')
    print(f'\nAccuracy: {acc}')
    print('\nClassification report:')
    print(classification_report(true_labels, pred_labels))
    
    pd.DataFrame({"opt_threshold": [opt_threshold]}).to_csv('results/predictions/optimal_threshold.csv', index = False)


if __name__ == '__main__':
    
    eval_net()
    