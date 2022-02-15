import tensorflow as tf

class ResNet50(tf.keras.Model):
    
    def __init__(self, IMG_WIDTH, IMG_HEIGHT, n_meta_features):
        super(ResNet50, self).__init__()

        # from pretrained network
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
        
        # Create the base model from the pre-trained model 
        self.IMG_SHAPE = self.IMG_SIZE + (3,)
        
        self.base_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet',
                                                       input_shape=(IMG_WIDTH, IMG_HEIGHT,3), classes=2, 
                                                       classifier_activation=None)
        
        self.base_model.trainable = False

        self.flat = tf.keras.layers.Flatten()
        self.dense_layer1 = tf.keras.layers.Dense(256, activation = "relu", 
                                                  kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))
        self.dropout_layer1 = tf.keras.layers.Dropout(0.5)
        #self.dense_layer2 = tf.keras.layers.Dense(300, activation = "relu")
        #self.dropout_layer2 = tf.keras.layers.Dropout(0.5)
        self.prediction_layer = tf.keras.layers.Dense(1, activation = "sigmoid")
        

    def call(self, inputs):
        x = inputs[0]
        x = self.base_model(inputs, training=False)
        x = self.flat(x)
        x = self.dense_layer1(x)
        x = self.dropout_layer1(x)
        #x = self.dense_layer2(x)
        #x = self.dropout_layer2(x)
        outputs = self.prediction_layer(x)
        
        return outputs