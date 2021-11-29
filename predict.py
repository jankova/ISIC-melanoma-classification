import pandas as pd
import tensorflow as tf
from argparse import ArgumentParser
from config import IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE
from dataset import get_data_gen


def parse_args():
    
    parser = ArgumentParser()
    parser.add_argument('--enet-type', type = str, required = True)
    
    args, _ = parser.parse_known_args()
    
    return args
    

def save_preds():
    
    args = parse_args()
    
    # load validation data
    _, val_ds = get_data_gen()
    
    model = tf.keras.models.load_model('results/saved_models/' + args.enet_type)
    
    # save predictions
    
    preds = model.predict(val_ds)
    preds = [prob[0] for prob in preds]
    
    df_preds = pd.DataFrame({'preds': preds})
    df_preds.to_csv(f'results/predictions/predictions-{args.enet_type}.csv', index = False)
    print(f"Predictions saved to 'results/predictions/predictions-{args.enet_type}.csv'.")
    
    
if __name__ == '__main__':
    
    save_preds()
    