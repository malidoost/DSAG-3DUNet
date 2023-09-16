# DSAG-3DUNet
Deeply Supervised Attention-Gated 3D U-Net for Brain Vasculature

## Overview

Cerebrovascular disease is a leading cause of death worldwide. Accurate blood vessel segmentation is a crucial step in diagnosing and treating these conditions. **DSAG-3DUNet** contains the code for our research presented in the paper titled "**Model utility of a deep learning-based segmentation is not Dice coefficient dependent: A case study in volumetric brain blood vessel segmentation**". We employed a deeply supervised attention-gated 3D U-Net, trained with the Focal Tversky loss function, to extract brain vasculature from volumetric magnetic resonance angiography (MRA) images.

This repository contains code for the following aspects of our research:

- [Segmentation Code](#segmentation-code): Code for training the deep learning model for brain blood vessel segmentation.
- [Prediction Code](#prediction-code): Code for making predictions using the trained model.
- [Evaluation Code](#evaluation-code): Code for evaluating the model's performance.

## Segmentation Code

The segmentation code in this repository is used to train a deeply supervised attention-gated 3D U-Net with the Focal Tversky loss function for brain blood vessel segmentation. You can find the code in the [segmentation folder](segmentation/). To run the segmentation code, follow the instructions provided in the respective README within the folder.

## Prediction Code

The prediction code allows you to use the trained model to make predictions on new volumetric MRA images. You can find the code in the [prediction folder](prediction/). To use the prediction code, follow the instructions provided in the respective README within the folder.

## Evaluation Code

Evaluating the model's performance is crucial in medical image analysis. The evaluation code provided in the [evaluation folder](evaluation/) helps you assess the segmentation results. Refer to the README within the folder for usage instructions.

### Usage Instructions

1. Clone this repository to your local machine:



## Citation

If you use this code or the associated research in your work, please consider citing our manuscript:

Alidoost, M., Ghodrati, V., Ahmadian, A., Shafiee, A., Hassani, C. H., Bedayat, A., & Wilson, J. L. (2023). Model utility of a deep learning-based segmentation is not Dice coefficient dependent: A case study in volumetric brain blood vessel segmentation. *Intelligence-Based Medicine*, 7, 100092.





## Table of Contents

- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Citation](#Citation)

## Getting Started

These instructions will help you get a copy of the project up and running on your local machine for development and testing purposes. See [Usage](#usage) for notes on how to use the code in a research context.

## Prerequisites

Prerequisites that are necessary to run the code:

- **ToxPathEngine** was developed using the Python programming language.
- For setup guidance, please visit [**PathFX**](https://github.com/jenwilson521/PathFX) and follow the provided instructions.

## Usage

Detailed instructions on how to use the code:

- Clone this repository to your local machine/cluster.
- You can use the text file in the "data" folder. It contains the dataset we used for our analysis, the drug toxicity dataset, consisting of pairs of drugs and their associated side effects obtained from drug labels.
- To be able to run the last version of PathFX made in our analyses, first, you need to clone [**PathFX**](https://github.com/jenwilson521/PathFX). Afterward, you should add (copy/paste) the files available in the "pathfx" folder here in this GitHub repository to the same folder names (scripts/rscs/results) in your cloned PathFX folder on your local drive. Subsequently, you can use the "runpathfx_scr.py" script in our "scripts" folder to run the last version of PathFX on your operating system and re-generate the results.
-  The "scripts" folder includes all the scripts needed to re-generate our analyses:
   - "map_scr.py": Map drugs and map/match side effects.
   - "runpathfx_scr.py": Run the last version of PathFX to generate the results of the paper.
   - "sepred_scr.py": Evaluate per side effect in the baseline analysis (calculate the sensitivity and specificity values) and identify the key pathway genes.
   - "defpath_scr.py": Define novel pathways using the distinct identified genes and omics data in addition to the old associated pathways.
   - "evalnewpath_scr.py": Evaluate (the novel defined) pathways per side effect and their corresponding phenotypes and produce the evaluation plots.
- Important Note: Make sure to update the directory paths in all scripts to match your local environment before running them.
- You can find the PathFX outcome for the drug Alteplase, as an example, in the "pathfx/results/" directory, which represents our final analysis results.

