# basic imports
import numpy as np
import re
import config

# tensorflow
import tensorflow as tf
from tensorflow.data import TFRecordDataset
AUTO = tf.data.experimental.AUTOTUNE

# custom imports
import config

# imports for unit testing
from sklearn.model_selection import train_test_split


def count_data_items(filenames):
    '''
    Count the total number of images corresponding to filenames.
    '''
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 
         for filename in filenames]
    return np.sum(n)

def read_labeled_tfrecord(example):
    '''
    Read a tfrecord which has a label (training and validation data), return image and label.
    '''
    tfrec_format = {
        'image'                        : tf.io.FixedLenFeature([], tf.string),
        'image_name'                   : tf.io.FixedLenFeature([], tf.string),
        'patient_id'                   : tf.io.FixedLenFeature([], tf.int64),
        'sex'                          : tf.io.FixedLenFeature([], tf.int64),
        'age_approx'                   : tf.io.FixedLenFeature([], tf.int64),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
        'diagnosis'                    : tf.io.FixedLenFeature([], tf.int64),
        'target'                       : tf.io.FixedLenFeature([], tf.int64)
    }           
    example = tf.io.parse_single_example(example, tfrec_format)
    return example['image'], example['target']

def read_unlabeled_tfrecord(example, return_image_name):
    '''
    Read a tfrecord which does not have a label (test data), return image and image name.
    '''
    tfrec_format = {
        'image'                        : tf.io.FixedLenFeature([], tf.string),
        'image_name'                   : tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return example['image'], example['image_name'] if return_image_name else 0
 
def prepare_image(img, augment=None, dim=256):  
    '''
    Do basic image preprocessing; decode, augment, resize.
    '''
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) #/ 255.0       ## REMOVING NORMALIZATION!!!!!
    
    if augment:
        raise("Augmentation not implemented yet!")
        img = augment(img)    # AUGMENTATION NOT IMPLEMENTED YET
           
    img = tf.reshape(img, [dim, dim, 3])
            
    return img

def get_dataset(files, augment=None, shuffle=False, repeat=False, 
                labeled=True, return_image_names=True, batch_size=16, dim=256):
    '''
    Load tfrecords into a TFRecordDataset instance.
    '''

    ds = TFRecordDataset(files, num_parallel_reads=None)
    ds = ds.cache()
    
    if repeat:
        ds = ds.repeat()
    
    if shuffle: 
        ds = ds.shuffle(1024*8)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
        
    # 
    if labeled: 
        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=None)
    else:
        ds = ds.map(lambda example: read_unlabeled_tfrecord(example, return_image_names), 
                    num_parallel_calls=None)      
    
    # preprocessing the image, augmentation
    ds = ds.map(lambda img, imgname_or_label: (prepare_image(img, 
                                                             augment=augment, 
                                                             dim=dim), 
                                               imgname_or_label), 
                num_parallel_calls=None)
    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTO)
    return ds


    
if __name__ == '__main__':
    '''
    Unit testing of dataset.py
    '''
   
    # Load data
    train_files_nums, valid_files_nums = train_test_split(np.arange(0,15), test_size = 0.2, random_state = 0)

    files_train = [config.TRAIN_PATH + 'train'+ str(x).zfill(2) + '*.tfrec' for x in train_files_nums]
    
    TRAINING_FILENAMES = tf.io.gfile.glob(files_train)

    train_ds_tf = get_dataset(TRAINING_FILENAMES, 
                              augment=None, 
                              shuffle=False, 
                              repeat=True,
                              dim = config.IMG_HEIGHT, 
                              batch_size=config.BATCH_SIZE)    
    # Print dataset info
    print(train_ds_tf)
    
    # Plot one batch of examples with labels
    
    import matplotlib.pyplot as plt
    
    if True:
        for example in train_ds_tf:
            fig, ax = plt.subplots(4,4, figsize = (10,10))
            for j in range(len(example[1])):
                
                img = example[0][j]
                ax.flatten()[j].imshow(img/255.0)
                ax.flatten()[j].set_title(f"Class: {int(example[1][j])}")
            
            fig.show()
            import sys
            sys.exit()
    
    