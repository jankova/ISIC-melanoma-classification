# SIIM-ISIC melanoma classification
https://www.kaggle.com/c/siim-isic-melanoma-classification

## Goals and dataset
The goal is to predict whether a skin lesion is malignant or benign, based on its photograph.

For training we used the 2020 ISIC dataset, resized to 256x256 and triple stratified:

https://www.kaggle.com/cdeotte/melanoma-256x256 

The dataset contains 33,126 images for training of which only 584 (1.8%) are malignant, resulting in high imbalance. Therefore we additionally consider a dataset of 4000 malignant-only images https://www.kaggle.com/cdeotte/malignant-v2-256x256.

Training was run using Tensorflow 2.3.0 on an AWS G4 instance. The model is a combination of a CNN architecture for training on images and a DNN model for the meta data, the resulting features are concatenated and passed through a dense layer.
The CNN architectures experiment with re-training the last layers of pre-trained models (ResNet50, EfficientNetB1,B3,B4 and VGG16). 

The output from the training epochs can be found in [03-main-training.ipynb](03-main-training.ipynb). 

Python packages: [requirements.txt](requirements.txt)

## Training summary
EfficientNets turned out to perform best in this task.

We experimented with
1. adding additional 4000 examples of malignant-only images to the training data
2. adding class weights
3. adjusting learning rates, architecture, image resolution
4. adding classical augmentations (rotations, flips, etc.)
5. adding hair augmentation

## Training: example use
```
python train.py --enet-type ResNet --n-epochs 10
python train.py --enet-type VGG16 --n-epochs 10
python train.py --enet-type EfficientNet --n-epochs 10
python train.py --enet-type EfficientNet --n-epochs 20 --augment --hair-augment 
python train.py --enet-type EfficientNet --n-epochs 20 --augment --hair-augment --add-malig
```

## Predictions: example use
Compute predictions.

```
python predict.py --enet-type EfficientNet
```

## Evaluate: example use

```
python evaluate.py 
```

<!-- 
## Ensembling
The predictions from the three models were ensembled into final predictions.

```
python ensemble.py
```

## Evaluation

```
python evaluate.py
```
-->

## ROC curve of EfficientNetB4

![ROC curve of the final model](roc_curve_effnet.jpg)

### Further improvements:
To obtain a high performing solution in this competetion, it would help to train several different EfficientNet architectures (B0-B7) with different datasets (ISIC 2020, 2019, 2018), with different image resolutions (256 to cca 800), with and without data/hair augmentation.

