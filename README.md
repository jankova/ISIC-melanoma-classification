# ISIC-melanoma-classification

The goal is to predict whether a mole is a malignant or benign, based on its image (photograph).

We used approximately 8000 images for training and 3000 images for testing (the full SIIM-ISIC dataset has around 30000 images),
with resolution 150x150 pixels.
With this reduced dataset, we obtain a final AUC score of 0.8079.

Training was run on an AWS instance in tensorflow using re-training the last layers of ResNet, EfficientNet and VGG16 and ensembling the final predictions.

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
