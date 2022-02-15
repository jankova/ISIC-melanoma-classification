# basic imports
import pandas as pd
import numpy as np

# tensorflow
import tensorflow as tf

# custom imports
import config

# for plotting
import matplotlib.pyplot as plt

def get_class_weights(add_malig):
    '''
    Compute class weights to feed into the model.
    '''
    train = pd.read_csv(config.TRAIN_PATH + 'train.csv')
    
    # adding malignant data
    if add_malig:
        train_malig = pd.read_csv(config.TRAIN_PATH_MALIG + 'train_malig_all.csv')
        train = pd.concat([train, train_malig], axis = 0)
  
    total = len(train)
    pos = len(train[train['target'] == 1]) #* 2 + 1627 + 580
    neg = len(train[train['target'] == 0]) #+ 8388
    
    print(f'Train examples: {total}')
    print(f'Number of malignant: {pos}')
    print(f'Number of benign: {neg}')
    
    # compute class weights
    
    weight_for_0 = (1 / neg) * (total) / 2.0 
    weight_for_1 = (1 / pos) * (total) / 2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f'Class weights: {class_weight}\n')
    
    return class_weight

def get_lr_callback(batch_size=8):
    '''
    Set learning rate across epochs.
    '''
    lr_start   = 0.000005
    lr_max     = 0.00000125 * config.BATCH_SIZE
    lr_min     = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep  = 0
    lr_decay   = 0.8
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback

def plot_history(history, enet_type):
    '''
    Plot loss and AUC across epochs.
    '''
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(enet_type + ' loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower left')
    plt.show()
    
    plt.plot(history.history['AUC'])
    plt.plot(history.history['val_AUC'])
    plt.title(enet_type + ' AUC score')
    plt.ylabel('AUC')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    

if __name__ == '__main__':  
    '''
    Unit testing of utils.py
    '''
    
    # print class weights
    get_class_weights(False)
    
    

    
    