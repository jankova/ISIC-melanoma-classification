import tensorflow as tf


class VGG16(tf.keras.Model):
    
    def __init__(self, IMG_WIDTH, IMG_HEIGHT, n_meta_features):
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
        self.dense_layer1 = tf.keras.layers.Dense(256, activation = "relu", kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))
        self.dropout_layer1 = tf.keras.layers.Dropout(0.5)
        #self.dense_layer2 = tf.keras.layers.Dense(300, activation = "relu")
        #self.dropout_layer2 = tf.keras.layers.Dropout(0.5)
        self.prediction_layer = tf.keras.layers.Dense(1, activation = "sigmoid")
        

    def call(self, inputs):
        
        x = self.base_model(inputs, training=False)
        x = self.flat(x)
        x = self.dense_layer1(x)
        x = self.dropout_layer1(x)
        #x = self.dense_layer2(x)
        #x = self.dropout_layer2(x)
        outputs = self.prediction_layer(x)
        
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

        self.flat = tf.keras.layers.Flatten()
        self.dense_layer1 = tf.keras.layers.Dense(2000, activation = "relu", kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))
        self.dropout_layer1 = tf.keras.layers.Dropout(0.5)
        self.dense_layer2 = tf.keras.layers.Dense(800, activation = "relu")
        self.dropout_layer2 = tf.keras.layers.Dropout(0.5)
        self.dense_layer3 = tf.keras.layers.Dense(100, activation = "relu")
        self.dropout_layer3 = tf.keras.layers.Dropout(0.5)
        
        self.prediction_layer = tf.keras.layers.Dense(1, activation = "sigmoid")
        

    def call(self, inputs):
        x = inputs[0]
        x = self.base_model(inputs, training=False)
        x = self.flat(x)
        x = self.dense_layer1(x)
        x = self.dropout_layer1(x)
        x = self.dense_layer2(x)
        x = self.dropout_layer2(x)
        x = self.dense_layer3(x)
        x = self.dropout_layer3(x)
        outputs = self.prediction_layer(x)
        
        return outputs

    