#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:36:45 2021

@author: janajankova
"""

import os
import argparse
import pandas as pd
import tensorflow as tf
from dataset import get_data_gen
from models.models_resnet import ResNet50
from models.models_efficient_net import EfficientNet
from models.models_vgg16 import VGG16
from config import IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--enet-type', type = str, required = True)
    parser.add_argument('--n-epochs', type = int, required = True)
    
    args, _ = parser.parse_known_args()
    return args

def run(train_ds, val_ds, args, ModelClass):
    print(args.enet_type)
    
    METRICS = [tf.keras.metrics.AUC(name = "AUC"), 
           tf.keras.metrics.BinaryAccuracy(name = "accuracy"), 
           tf.keras.metrics.Precision(name = "precision"),
           tf.keras.metrics.Recall(name = "recall")]
    
    #checkpoint_path = './training/cp-{epoch:04d}.ckpt'
    #checkpoint_dir = os.path.dirname(checkpoint_path)
    
    #cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #    filepath = checkpoint_path,
    #    save_weights_only = True,
    #    verbose = 1
    #)
        
    model = ModelClass(IMG_WIDTH, IMG_HEIGHT)

    lr = 3e-4

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = lr),
        loss = tf.keras.losses.BinaryCrossentropy(from_logits = False), 
        metrics = METRICS
    )
        
    history = model.fit(train_ds, 
                        epochs = args.n_epochs, 
                        validation_data = val_ds,
                        #callbacks = [cp_callback]
                       )
    
    # save model
    model.save('results/saved_models/'+args.enet_type)
    
    # save predictions
    
    preds = model.predict(val_ds)
    preds = [prob[0] for prob in preds]
    #print(preds)
    df_preds = pd.DataFrame({'preds': preds})
    df_preds.to_csv(f'results/predictions/predictions-{args.enet_type}.csv', index = False)

    
    
def main():
    
    args = parse_args()
    
    #IMG_WIDTH = 150
    #IMG_HEIGHT = 150
    #BATCH_SIZE = 32
    
    # load data
    train_ds, val_ds = get_data_gen()
    
    if args.enet_type == 'res_net':
        ModelClass = ResNet50
    elif args.enet_type == 'vgg16':
        ModelClass = VGG16
    elif args.enet_type == 'eff_net':
        ModelClass = EfficientNet
    else:
        raise NotImplementedError()
           
    # train models
    run(train_ds, val_ds, args, ModelClass)
    
    # ensemble
    
    # save submission predictions
    print(args.enet_type)
    print("hello world")
    

if __name__ == '__main__':
    
    main()
    
    