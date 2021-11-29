#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:36:45 2021

@author: janajankova
"""

import tensorflow as tf


# plain AlexNet

def AlexNetCE_plain(input_shape = (150, 150, 3), classes = 6):
    X_input = tf.keras.Input(input_shape)
    X = X_input
    X = tf.keras.layers.Conv2D(96, (11, 11), strides = (4, 4), activation = "relu", name = 'conv1')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = tf.keras.layers.Conv2D(256, (5, 5), padding = "same",activation = "relu", name = 'conv2')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = 'bn_conv2')(X)
    X = tf.keras.layers.Conv2D(256, (3, 3), padding = "same",activation = "relu", name = 'conv5')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(4096, activation = "relu", name='fc' + str(1))(X)
    X = tf.keras.layers.Dense(4096, activation = "relu", name='fc' + str(2))(X)
    X = tf.keras.layers.Dense(classes, activation='softmax', name='fc' + str(classes))(X)
       
    # Create model
    model = tf.keras.Model(inputs = X_input, outputs = X, name='ALEXNETCE')

    return model


# AlexNet with dropout

def AlexNetCE(input_shape = (150, 150, 3), classes = 6):
    X_input = tf.keras.Input(input_shape)
    X = X_input
    X = tf.keras.layers.Conv2D(96, (11, 11), strides = (4, 4), activation = "relu", name = 'conv1')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = tf.keras.layers.Conv2D(256, (5, 5), padding = "same",activation = "relu", name = 'conv2')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = 'bn_conv2')(X)
    X = tf.keras.layers.Conv2D(256, (3, 3), padding = "same",activation = "relu", name = 'conv5')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dropout(0.5)(X)
    X = tf.keras.layers.Dense(4096, activation = "relu", name='fc' + str(1))(X)
    X = tf.keras.layers.Dropout(0.5)(X)
    X = tf.keras.layers.Dense(4096, activation = "relu", name='fc' + str(2))(X)
    X = tf.keras.layers.Dense(classes, activation='softmax', name='fc' + str(classes))(X)
       
    # Create model
    model = tf.keras.Model(inputs = X_input, outputs = X, name='ALEXNETCE')

    return model




class VGG16(tf.keras.Model):
    
    def __init__(self, IMG_WIDTH, IMG_HEIGHT):
        super(VGG16, self).__init__()

        # from pretrained network
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
        # Create the base model from the pre-trained model vgg16
        self.IMG_SHAPE = self.IMG_SIZE + (3,)
        self.base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet',
                                                       input_shape=(IMG_WIDTH, IMG_HEIGHT,3), classes=2, 
                                                       classifier_activation=None)
        
        self.base_model.trainable = False

        #self.input_layer = tf.keras.Input(shape=(150, 150, 3))

        self.flat = tf.keras.layers.Flatten()
        self.dense_layer1 = tf.keras.layers.Dense(1000, activation = "relu", kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))
        self.dropout_layer1 = tf.keras.layers.Dropout(0.5)
        self.dense_layer2 = tf.keras.layers.Dense(300, activation = "relu")
        self.dropout_layer2 = tf.keras.layers.Dropout(0.5)
        self.prediction_layer = tf.keras.layers.Dense(1, activation = "sigmoid")
        

    def call(self, inputs):
        #inputs = self.input_layer(inputs) #tf.keras.Input(shape=(150, 150, 3))
        #x = self.input_layer(inputs) #tf.keras.Input(shape=(150, 150, 3))
        x = self.base_model(inputs, training=False)
        x = self.flat(x)
        x = self.dense_layer1(x)
        x = self.dropout_layer1(x)
        x = self.dense_layer2(x)
        x = self.dropout_layer2(x)
        outputs = self.prediction_layer(x)
        #model = tf.keras.Model(inputs, outputs) 
        return outputs
    
class VGG16aug(tf.keras.Model):
    
    def __init__(self, IMG_WIDTH, IMG_HEIGHT):
        super(VGG16aug, self).__init__()

        # from pretrained network
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
        # Create the base model from the pre-trained model vgg16
        self.IMG_SHAPE = self.IMG_SIZE + (3,)
        self.base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet',
                                                       input_shape=(IMG_WIDTH, IMG_HEIGHT,3), classes=2, 
                                                       classifier_activation=None)
        
        self.base_model.trainable = False

        #self.input_layer = tf.keras.Input(shape=(150, 150, 3))

        self.flat = tf.keras.layers.Flatten()
        self.dense_layer1 = tf.keras.layers.Dense(2000, activation = "relu", kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))
        self.dropout_layer1 = tf.keras.layers.Dropout(0.5)
        self.dense_layer2 = tf.keras.layers.Dense(800, activation = "relu")
        self.dropout_layer2 = tf.keras.layers.Dropout(0.5)
        self.dense_layer3 = tf.keras.layers.Dense(100, activation = "relu")
        self.dropout_layer3 = tf.keras.layers.Dropout(0.5)
        
        self.prediction_layer = tf.keras.layers.Dense(1, activation = "sigmoid")
        

    def call(self, inputs):
        #inputs = self.input_layer(inputs) #tf.keras.Input(shape=(150, 150, 3))
        #x = self.input_layer(inputs) #tf.keras.Input(shape=(150, 150, 3))
        x = self.base_model(inputs, training=False)
        x = self.flat(x)
        x = self.dense_layer1(x)
        x = self.dropout_layer1(x)
        x = self.dense_layer2(x)
        x = self.dropout_layer2(x)
        x = self.dense_layer3(x)
        x = self.dropout_layer3(x)
        outputs = self.prediction_layer(x)
        #model = tf.keras.Model(inputs, outputs) 
        return outputs

    