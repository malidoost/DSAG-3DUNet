# DSAG-3DUNet
Deeply Supervised Attention-Gated 3D U-Net for Brain Vasculature

## Overview

Cerebrovascular disease is a leading cause of death worldwide. Accurate blood vessel segmentation is a crucial step in diagnosing and treating these conditions. **DSAG-3DUNet** contains the code for our research presented in the paper titled "**Model utility of a deep learning-based segmentation is not Dice coefficient dependent: A case study in volumetric brain blood vessel segmentation**". We employed a deeply supervised attention-gated 3D U-Net, trained with the Focal Tversky loss function, to extract brain vasculature from volumetric magnetic resonance angiography (MRA) images. We developed our code based on the code that was shared in the publication by [*Isensee et al.*]([https://github.com/jenwilson521/PathFX](https://link.springer.com/chapter/10.1007/978-3-319-75238-9_25)) in 2018.

This repository contains code for the following aspects of our research:

- [Segmentation Code](#segmentation-code): Code for training the deep learning model for brain blood vessel segmentation.
- [Prediction Code](#prediction-code): Code for making predictions using the trained model.
- [Evaluation Code](#evaluation-code): Code for evaluating the model's performance.

## Segmentation Code

The segmentation code, **c_segment.py**, in this repository, is used to train a deeply supervised attention-gated 3D U-Net with the Focal Tversky loss function for brain blood vessel segmentation.
- If you are running out of memory, try training using "(64, 64, 64)" shaped patches.
- Reducing the "batch_size" and "validation_batch_size" parameters will also reduce the amount of memory required for training as smaller batch sizes feed smaller chunks of data to the CNN.
- If the batch size is reduced to 1 and you are still running out of memory, you can also try changing the patch size to "(32, 32, 32)".
- Keep in mind that smaller patch sizes may not perform as well as larger patch sizes.

### Inputs & Outputs:
- Place all the data in the ./data directory. Each case needs to be in a folder including the image 'mra' and the label "truth" both in NIFTI (.nii) format.
- You need to change the hyperparameters only in **c_segment.py**.
- To run different network structures, comment and uncomment the lines in the "instantiate new model" section in **c_segment.py**.
- "loss_graph.png": Figure of Loss (training and validation).
- "un_model.h5": Model is saved. 

## Prediction Code

The prediction code, **c_predict**, allows you to use the trained model to make predictions on new volumetric MRA images.
- The predictions will be written in the "prediction" folder along with the input data and ground truth labels for comparison.

## Evaluation Code

Evaluating the model's performance is crucial in medical image analysis. The evaluation code, **c_evaluate**, helps you assess the segmentation results.
- "metrics_boxplot.png": Box plots of the evaluation metrics including Dice, Precision, Recall, and F2. 
- "precision_recall_curve.png": Figure of precision and recall.

## Prerequisites

Requirements to install the code:

- Keras>=2
- numpy>=1.12.1
- nilearn>=0.3.0
- tables>=3.4.2
- nibabel>=2.1.0

## Citation

If you use this code or the associated research in your work, please consider citing our manuscript:

Alidoost, M., Ghodrati, V., Ahmadian, A., Shafiee, A., Hassani, C. H., Bedayat, A., & Wilson, J. L. (2023). Model utility of a deep learning-based segmentation is not Dice coefficient dependent: A case study in volumetric brain blood vessel segmentation. *Intelligence-Based Medicine*, 7, 100092.
