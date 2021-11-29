#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:36:45 2021

@author: janajankova
"""

import tensorflow as tf
from config import IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE



def get_data_gen():
    
    print(BATCH_SIZE)
    
    data_dir_train = 'data/processed/train/'
    data_dir_test = 'data/processed/test/'

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir_train,
        labels = "inferred",
        validation_split = 0.0,
        #shuffle = True,
        seed = 42,
        #subset = "training",
        batch_size = BATCH_SIZE,
        image_size = (IMG_WIDTH, IMG_HEIGHT)
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir_test,
        labels = "inferred",
        validation_split = 0.0,
        shuffle = False,
        seed = 42, 
        batch_size = BATCH_SIZE,
        image_size = (IMG_WIDTH, IMG_HEIGHT)
    )
    
    return train_ds, val_ds
    
if __name__ == '__main__':
    
    get_df()
    