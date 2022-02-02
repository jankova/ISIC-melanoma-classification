# basic imports
import numpy as np
import pandas as pd
from argparse import ArgumentParser

# sklearn
from sklearn.model_selection import train_test_split

# tensorflow
import tensorflow as tf

# custom imports
import config 
from dataset import get_dataset
from models.models_efficient_net import EfficientNet, EfficientNetAug



def parse_args():
    
    parser = ArgumentParser()
    parser.add_argument('--enet-type', type = str, required = True)
    
    args, _ = parser.parse_known_args()
    
    return args
    

def save_preds():
    
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

    # load model
    #model = ModelClass(args.img_width, args.img_height)
    
    #model.compile(
    #    optimizer = tf.keras.optimizers.Adam(learning_rate = config.LEARNING_RATE),
    #    loss = tf.keras.losses.BinaryCrossentropy(from_logits = False), 
    #    metrics = METRICS
    #)
    model = tf.keras.models.load_model('results/saved_models/' + args.enet_type)
    
    # save predictions
    
    preds = model.predict(val_ds_tf)
    print(preds)
    preds = [prob[1] for prob in preds]
    
    df_preds = pd.DataFrame({'preds': preds})
    df_preds.to_csv(f'results/predictions/predictions-{args.enet_type}.csv', index = False)
    print(f"Predictions saved to 'results/predictions/predictions-{args.enet_type}.csv'.")
    
    
if __name__ == '__main__':
    
    save_preds()
    