import tensorflow as tf
from config import IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE

def get_data_gen():
    
    print(f"Batch size: {BATCH_SIZE}\n")
    
    data_dir_train = 'data/processed/train/'
    data_dir_test = 'data/processed/test/'

    print("Train data:")
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir_train,
        labels = "inferred",
        validation_split = 0.0,
        # shuffle = True,
        seed = 42,
        # subset = "training",
        batch_size = BATCH_SIZE,
        image_size = (IMG_WIDTH, IMG_HEIGHT)
    )
    
    print("\nTest data:")
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
    
    get_data_gen()
    