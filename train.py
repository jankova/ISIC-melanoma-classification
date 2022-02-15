# basic imports
import os
import argparse
import pandas as pd
import numpy as np
import random

# sklearn
from sklearn.model_selection import train_test_split

# tensorflow
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

# custom imports
from dataset import get_dataset, count_data_items
from models.models_resnet import ResNet50
from models.models_efficient_net import EfficientNet
from models.models_vgg16 import VGG16
import config
from utils import get_class_weights, plot_history, get_lr_callback


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--enet-type', type = str, required = True)
    parser.add_argument('--n-epochs', type = int, default = config.N_EPOCHS)
    parser.add_argument('--img-width', type = int, default = config.IMG_WIDTH)
    parser.add_argument('--img-height', type = int, default = config.IMG_HEIGHT)
    parser.add_argument('--add-malig', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--hair-augment', action='store_true')
    
    args, _ = parser.parse_known_args()
    return args



def run(train_ds_tf, val_ds_tf, args, ModelClass, steps_per_epoch):    
    
    # model setup
    METRICS = [tf.keras.metrics.AUC(name = "AUC"), 
               tf.keras.metrics.BinaryAccuracy(name = "accuracy"), 
               tf.keras.metrics.Precision(name = "precision"),
               tf.keras.metrics.Recall(name = "recall"),
               tf.keras.metrics.Recall(name = "F1")]
        
    model = ModelClass(args.img_width, args.img_height, n_meta_features = 3)
    
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = config.LEARNING_RATE),
        loss = tf.keras.losses.BinaryCrossentropy(from_logits = False), 
        metrics = METRICS,  
        #run_eagerly=True                  # SLOWS COMPUTATION???????????????????????????? but lets convert tensors to numpy
    )
    
    # early stopping
    es = EarlyStopping(
        monitor='val_AUC', min_delta=0, patience=3, verbose=0,
        mode='max', baseline=None, restore_best_weights=False
    )
    
    # checkpoints
    MODEL_PATH_BEST_AUC = config.MODEL_BASE_PATH + args.enet_type + '_model_best_auc.h5'
    MODEL_PATH_BEST_LOSS = config.MODEL_BASE_PATH + args.enet_type + '_model_best_loss.h5'
   
    checkpoint_auc = ModelCheckpoint(MODEL_PATH_BEST_AUC, monitor='val_AUC', mode='max', save_best_only=True,
                                     #save_weights_only=True, 
                                     verbose=0)
    checkpoint_loss = ModelCheckpoint(MODEL_PATH_BEST_LOSS, monitor='val_loss', mode='min', save_best_only=True,
                                      #save_weights_only=True, 
                                      verbose=0)
    
    # class weights
    class_weight = get_class_weights(args.add_malig)

    history = model.fit(train_ds_tf, 
                        validation_data=val_ds_tf,
                        steps_per_epoch=steps_per_epoch,
                        callbacks = [checkpoint_auc, 
                                     checkpoint_loss, 
                                     #get_lr_callback(config.BATCH_SIZE),  # learning rate callback
                                     es],
                        epochs = args.n_epochs, 
                        class_weight = class_weight,
                        )    

    
    model.load_weights(MODEL_PATH_BEST_AUC)
    
    txt_augment, txt_hair_augment, txt_malig = '', '', ''
    
    if args.augment:
        txt_augment = '_aug'
    if args.hair_augment:
        txt_hair_augment = '_hair_aug'
    if args.add_malig:
        txt_malig = '_add_malig'
        
    model.save('results/saved_models/' + args.enet_type + txt_augment + txt_hair_augment + txt_malig )
    
    return history
    
    

    
def main():
    
    args = parse_args()
    
    # logging info:
    #print(args)
    print(f"Architecture: {args.enet_type}")
    print(f"Using augmentation: {args.augment}")
    print(f"Using hair augmentation: {args.hair_augment}")
    print(f"Add malignant-only images: {args.add_malig}\n")

    # load train and valid data files
    
    train_files_nums, valid_files_nums = train_test_split(np.arange(0,15), test_size = 0.2, random_state = 1)
    
    files_train = [config.TRAIN_PATH + 'train'+ str(x).zfill(2) + '*.tfrec' for x in train_files_nums]
    files_valid = [config.TRAIN_PATH + 'train' + str(x).zfill(2) + '*.tfrec' for x in valid_files_nums]
    
    TRAINING_FILENAMES = tf.io.gfile.glob(files_train)  
    VALID_FILENAMES = tf.io.gfile.glob(files_valid)
      
    # add extra malignant images to train files only (not to valid files!)
    if args.add_malig:
        files_train_malig = [config.TRAIN_PATH_MALIG + 'train*.tfrec' ]
        TRAINING_FILENAMES += tf.io.gfile.glob(files_train_malig)
    
    np.random.shuffle(TRAINING_FILENAMES)
    np.random.shuffle(VALID_FILENAMES)
        
    # create data loaders

    
    train_ds_tf = get_dataset(TRAINING_FILENAMES, 
                              augment=args.augment,  ### CAREFUL WITH THIS
                              hair_augment = args.hair_augment,
                              shuffle=True, 
                              repeat=True,
                              dim=args.img_height, 
                              batch_size=config.BATCH_SIZE)    
    
    print(train_ds_tf.take(1))
    
    val_ds_tf = get_dataset(VALID_FILENAMES,
                            augment=None,  
                            shuffle=False,
                            repeat=False, 
                            dim=args.img_height)

    print("Datasets loaded...")
    
    steps_per_epoch = count_data_items(TRAINING_FILENAMES)/config.BATCH_SIZE

    # identify model
    if args.enet_type == 'ResNet':
        ModelClass = ResNet50
    elif args.enet_type == 'VGG16':
        ModelClass = VGG16
    elif args.enet_type == 'EfficientNet':
        ModelClass = EfficientNet
    elif args.enet_type == 'EfficientNetAug':
        ModelClass = EfficientNetAug    
    else:
        raise NotImplementedError()
           
    
    # train and save model
    history = run(train_ds_tf, val_ds_tf, args, ModelClass, steps_per_epoch)
    
    # plot history
    plot_history(history, args.enet_type)
    

if __name__ == '__main__':
    
    main()
    
    