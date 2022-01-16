# SIIM-ISIC melanoma classification

The goal is to predict whether a lesion is malignant or benign, based on its image (photograph).

We used 12119 images for training and 1537 images for testing, with resolution 150x150 pixels. This is only a subset of the full [SIIM-ISIC-dataset](https://www.kaggle.com/c/siim-isic-melanoma-classification/data), which has around 40000 images.
With the reduced dataset, we obtain a final AUC score of 0.8175.

Training was run on an AWS instance using Tensorflow by re-training the last layers of ResNet, EfficientNet and VGG16 and ensembling the final predictions.

The output from the training epochs can be found in [01-run-training-notebook.ipynb](01-run-training-notebook.ipynb). 

Python packages requirements: see [requirements.txt](requirements.txt)

## Training

```
python train.py --enet-type res_net --n-epochs 10
python train.py --enet-type eff_net --n-epochs 10
python train.py --enet-type vgg16 --n-epochs 10
```

## Predictions
Compute predictions for each model. 

```
python predict.py --enet-type res_net 
python predict.py --enet-type eff_net
python predict.py --enet-type vgg16
```

## Ensembling
The predictions from the three models were ensembled into final predictions.

```
python ensemble.py
```

## Evaluation

```
python evaluate.py
```

## ROC curve
Plot ROC curve, AUC score.

![ROC curve of the final model](results/plots/roc_curve.jpg)

### Further improvements:
- train on the full dataset of 40000 images with a higher resolution
- implement data augmentation
- encorporate meta data
