import tensorflow as tf

class EfficientNet(tf.keras.Model):
    
    def __init__(self, IMG_WIDTH, IMG_HEIGHT, n_meta_features):
        super(EfficientNet, self).__init__()

        # from pretrained network
        self.n_meta_features = n_meta_features
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
        # Create the base model from the pre-trained model vgg16
        self.IMG_SHAPE = self.IMG_SIZE + (3,)
        
        
        self.base_model = tf.keras.applications.efficientnet.EfficientNetB4(include_top=False, weights='imagenet',
                                                       input_shape=(IMG_WIDTH, IMG_HEIGHT,3), classes=2, 
                                                       classifier_activation=None)
        
        
        self.base_model.trainable = False
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        
        # model for meta data
        
        self.meta_model = tf.keras.Sequential()
        self.meta_model.add(tf.keras.layers.Dense(500, 
                                                  input_shape=(self.n_meta_features,), 
                                                  activation = "relu"))
        self.meta_model.add(tf.keras.layers.BatchNormalization())
        self.meta_model.add(tf.keras.layers.Dropout(0.2))
        self.meta_model.add(tf.keras.layers.Dense(250, activation = "relu"))
        self.meta_model.add(tf.keras.layers.BatchNormalization())
        self.meta_model.add(tf.keras.layers.Dropout(0.2))
        
        
        # combining CNN and meta data model 
        self.dense1 = tf.keras.layers.Dense(128, input_shape=(1796,), activation = "relu")
        self.drop1 = tf.keras.layers.Dropout(0.4)
    
        self.prediction_layer = tf.keras.layers.Dense(1, activation = "sigmoid")
        


    def call(self, inputs):
        # INPUTS:
            # inputs[0] = image
            # inputs[1] = meta features (3 features)
            
        # CNN model for image features
        x_image = inputs[0]
        x = self.base_model(x_image, training=False)
        x = self.pool(x)

        # FC network for meta features
        x_meta = inputs[1:]
        out = self.meta_model(x_meta)
        
        # concatenate CNN features and meta data features
        x_concat = tf.concat([x, out], axis = 1)
        
        # final FC network
        outputs = self.dense1(x_concat)
        outputs = self.drop1(outputs)
        outputs = self.prediction_layer(outputs)
         
        return outputs
    
    
