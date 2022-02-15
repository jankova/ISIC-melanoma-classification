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
from hair_augmentation import AdvancedHairAugmentation

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
    
    site_dct = ['site_head/neck', 'site_lower extremity', 
                'site_oral/genital', 'site_palms/soles', 
                'site_torso', 'site_upper extremity']
    
    #example_obj = tf.train.Example()
    #example_obj.ParseFromString(example.numpy())
    
    return example['image'], example['target'], tf.stack([example['age_approx'], 
                                                          example['sex'], 
                                                          example['anatom_site_general_challenge'] ])
            

def read_unlabeled_tfrecord(example, return_image_name):
    '''
    Read a tfrecord which does not have a label (test data), return image and image name.
    '''
    tfrec_format = {
        'image'                        : tf.io.FixedLenFeature([], tf.string),
        'image_name'                   : tf.io.FixedLenFeature([], tf.string),
        'sex'                          : tf.io.FixedLenFeature([], tf.int64),
        'age_approx'                   : tf.io.FixedLenFeature([], tf.int64),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return example['image'], example['image_name'] if return_image_name else 0, tf.stack([example['age_approx'], example['sex'], example['anatom_site_general_challenge'] ])
 
    
def augment_image(img):
    '''
    Augment data.
    '''
    hair_aug = AdvancedHairAugmentation(hairs_folder='data/augmentation/melanoma-hairs')

    aug_image = hair_aug.augment(img)
    
    return aug_image
    
def prepare_image(img, augment=None, hair_augment = None, dim=256):  
    '''
    Do basic image preprocessing; decode, augment, resize.
    '''
    
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) #/ 255.0      
    
    # Hair augmentation
    if hair_augment:
        img = augment_image(img)    # AUGMENTATION NOT IMPLEMENTED YET
      
    # Classical augmentations
    if augment:

        #img = tf.image.random_crop(img, [config.CROP_SIZE, config.CROP_SIZE, 3]) 
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_hue(img, 0.01)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_brightness(img, 0.1)
        
        # Coarse dropout
        #img = dropout(img, DIM=config.CROP_SIZE, PROBABILITY=config.DROP_FREQ, CT=config.DROP_CT, SZ=config.DROP_SIZE])
        
    img = tf.reshape(img, [dim, dim, 3])
            
    return img

def get_dataset(files, augment=None, hair_augment=None, shuffle=False, repeat=False, 
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
        print("UNLABELLED")
        ds = ds.map(lambda example: read_unlabeled_tfrecord(example, return_image_names), 
                    num_parallel_calls=None)      
    
    # preprocessing the image, augmentation
    ds = ds.map(lambda img, imgname_or_label, meta: ((prepare_image(img, 
                                                             augment=augment, 
                                                             dim=dim), 
                                                    meta),
                                                    imgname_or_label, 
                                                    ), 
                num_parallel_calls=None)
    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTO)
    
    
    return ds


def prep_img_display(img):
    img = tf.cast(img, tf.float32) 
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.reshape(img, [dim, dim, 3])
    return img

    
if __name__ == '__main__':
    '''
    Unit testing of dataset.py
    '''
   
    # Load data
    train_files_nums, valid_files_nums = train_test_split(np.arange(0,15), test_size = 0.2, random_state = 0)

    files_train = [config.TRAIN_PATH + 'train'+ str(x).zfill(2) + '*.tfrec' for x in train_files_nums]
    
    TRAINING_FILENAMES = tf.io.gfile.glob(files_train)

    train_ds_tf = get_dataset(TRAINING_FILENAMES, 
                              augment=False, 
                              shuffle=False, 
                              repeat=True,
                              dim = config.IMG_HEIGHT, 
                              batch_size=config.BATCH_SIZE)    
    # Print dataset info
    print(train_ds_tf)
    
    # Plot one batch of examples with labels
    
    import matplotlib.pyplot as plt
    
    tf.compat.v1.enable_eager_execution()
    
    # !!! examples are in the form of a nested tuple: ((image, meta_information), label)
          ## hence to access the image of an example, use example[0][0]
    
    if True:
        for example in train_ds_tf:
            
            plt.close()
            fig, ax = plt.subplots(4,4, figsize = (10,10))
           
            num_records_in_batch = len(example[1])  # gets the number of labels in a batch
               
            for j in range(num_records_in_batch):
                
                img = example[0][0][j]
                
                ax.flatten()[j].imshow(img/255.0)
                ax.flatten()[j].set_title(f"Class: {int(example[1][j])}")
            
            fig.show()
            
            import sys
            sys.exit()
            
        
    
