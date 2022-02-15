
# Default values for image resolution (use --img-width and --img-height in a call to override them)
IMG_WIDTH = 256
IMG_HEIGHT = 256

CROP_SIZE = 230

# Model training
BATCH_SIZE = 16
LEARNING_RATE = 0.001
N_EPOCHS = 10 # use --n-epochs in a call to override


# Paths   

# --Model
MODEL_BASE_PATH = "results/saved_models/"

# --Data
TRAIN_PATH = 'data/preprocessed_tfr/tfr_records_256/'
TRAIN_PATH_MALIG = 'data/preprocessed_tfr/tfr_records_256_malig/'



