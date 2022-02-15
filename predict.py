# basic imports
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm

# sklearn
from sklearn.model_selection import train_test_split

# tensorflow
import tensorflow as tf
from tensorflow.data import TFRecordDataset

# custom imports
import config 
from dataset import get_dataset
from models.models_efficient_net import EfficientNet



def parse_args():
    
    parser = ArgumentParser()
    parser.add_argument('--enet-type', type = str, required = True)
    parser.add_argument('--data', type = str, default = 'valid')
    
    args, _ = parser.parse_known_args()
    
    return args
    

def save_preds():
    
    args = parse_args()

    # load validation data
    valid_files_nums = pd.read_csv('results/train_valid_split/valid_files_nums.csv')
    valid_files_nums = valid_files_nums.values.flatten()
    files_valid = [config.TRAIN_PATH + 'train' + str(x).zfill(2) + '*.tfrec' for x in valid_files_nums]
    VALID_FILENAMES = tf.io.gfile.glob(files_valid)
    
    val_ds_tf = get_dataset(VALID_FILENAMES,
                            augment=None, 
                            shuffle=False,
                            repeat=False, 
                            dim=config.IMG_HEIGHT)

    # load test data
    files_test = [config.TRAIN_PATH + 'test' + str(x).zfill(2) + '*.tfrec' for x in range(16)] 
    TEST_FILENAMES = tf.io.gfile.glob(files_test)

    test_ds_tf = get_dataset(TEST_FILENAMES,
                             labeled=False,
                             augment=None, 
                             shuffle=False,
                             repeat=False, 
                             dim=config.IMG_HEIGHT)

    
    if args.data == "test":
        ds = test_ds_tf
    else:
        ds = val_ds_tf
    
    # load optimal threshold
    opt_threshold = pd.read_csv('results/predictions/optimal_threshold.csv')
    opt_threshold = opt_threshold.values[0]
    print(opt_threshold)

    # load best AUC model
    model = tf.keras.models.load_model('results/saved_models/' + args.enet_type)
    
    predictions = []
    image_names = []
    labels = []
    
    # create predictions
    for (x_batch, y_batch) in tqdm(ds):

        preds = model.predict(x_batch)
        preds = preds.flatten()

        if args.data == "test":
            
            img_names = [str(item.numpy())[2:-1] for item in y_batch ] 
            image_names.extend(img_names)
        else: 
            labels.extend(np.array(y_batch))
        predictions.extend(preds)
        
        
    if args.data == "test":
        fname = 'test'
        df_preds = pd.DataFrame({'image_name': image_names, 'target': predictions})
    else:
        fname = 'valid'
        df_preds = pd.DataFrame({'target': predictions, 'true_labels': labels})
        
    # save predictions
    
    df_preds.to_csv(f'results/predictions/predictions-{fname}-{args.enet_type}.csv', index = False)
    print(f"Predictions saved to 'results/predictions/predictions-{fname}-{args.enet_type}.csv'.")
    
    return df_preds
    
    
if __name__ == '__main__':
    
    save_preds()
    