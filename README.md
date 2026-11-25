# Butterfly and Moth Classification
 
## Course Information
 
Course: Data Problems (DATA 4380) at UTA
 
Group Members:
 
- Isabela Alaniz
- Clarissa Terrones
- Annie Tran
- Manya Vikal
## Overview
 
This project trains a deep learning model to identify the species of a butterfly or moth from a single photo. The dataset has 100 species.
 
The goal is to answer two questions:
 
1. How accurately can a model classify these 100 species?
2. Which approach works best when there are only a small number of training photos?
## Problem Statement
 
This is a supervised image classification problem. Each photo has one label, which is its species. The model predicts one of 100 labels.
 
## Dataset
 
Source: Kaggle, "Butterfly & Moths Image Classification 100 Species" by gpiosenka
 
Link: https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species
 
| Split | Photos | Photos per species |
|---|---|---|
| Train (full folder) | 12,594 | Varies |
| Validation | 500 | 5 |
| Test | 500 | 5 |
 
Every image is 224 by 224 pixels, in color, saved as JPG.
 
We did not train on the full training folder. We used a training subset of 20 random photos per species, which is 2,000 photos in total. A fixed random seed was used so the subset is the same each time.
 
## Data Preparation
 
- Built a training subset of 20 photos per species, using a fixed random seed.
- Scaled pixel values to the 0 to 1 range for the CNN (divided by 255).
- Used the ImageNet mean and standard deviation for the pretrained models.
- Applied random flips, rotations, and zooms to the training photos only.
- Did not remove any photos. Some photos have watermarks and transparent backgrounds, which we list as a limitation.
## Models
 
| Model | Notebook | What it is |
|---|---|---|
| CNN v1 to v4 | Part 1 | A model built from scratch. We added one change at a time. |
| CNN tuning | Part 1 | Tests 3 learning rates and 3 batch sizes (9 runs). |
| ResNet50 | Part 2 | Pretrained on ImageNet. Trained with feature extraction, then fine-tuning. |
| EfficientNet-B0 | Part 2 | Pretrained on ImageNet. Trained with feature extraction, then fine-tuning. |
 
Feature extraction means the pretrained layers are frozen and only the new output layer is trained. Fine-tuning means the last part of the pretrained model is also trained, with a lower learning rate.
 
## Key Results
 
Test set results (500 photos):
 
| Model | Test accuracy | Test loss |
|---|---|---|
| CNN v1 (raw pixels) | 24.8% | 3.347 |
| CNN v2 (normalized, baseline) | 28.6% | 3.302 |
| CNN v3 (add augmentation) | 42.4% | 2.574 |
| CNN v4 (add batch norm and dropout) | 60.2% | 1.553 |
| ResNet50, feature extraction | 86.2% | 0.541 |
| ResNet50, fine-tuned | 90.2% | 0.452 |
| EfficientNet-B0, feature extraction | 90.8% | 0.392 |
| EfficientNet-B0, fine-tuned | 92.8% | 0.254 |
 
CNN tuning results:
 
| Batch size | Best validation accuracy |
|---|---|
| 16 | 70.2% (learning rate 0.0001) |
| 32 | 69.6% |
| 64 | 67.6% |
 
The tuning chart labels show "LR=0.0" for some settings because of a rounding issue in the plotting code. The learning rates were 0.0003 and 0.0001.
 
## Key Findings
 
- The pretrained models scored far higher than the CNN built from scratch.
- Fine-tuning improved both pretrained models.
- EfficientNet-B0 fine-tuned had the best test accuracy (92.8%) and the lowest test loss (0.254).
- Learning rate and batch size changed the score only a little, about 3 points. Design changes such as augmentation and dropout made a much bigger difference.
- The most common mistakes are between look-alike species, such as common wood-nymph and wood satyr.
## Repository Structure
 
```
notebooks/
    butterfly_part1_cnn_mac.ipynb         Part 1: data, preprocessing, CNN from scratch, tuning
    butterfly_part2_transfer_mac.ipynb    Part 2: ResNet50 and EfficientNet-B0
.gitignore
README.md
```
 
## How to Run
 
0. System requirement: this project was run locally on Mac’s MPS GPU so notebooks are compatible with Mac system.
1. Install the libraries:
```
pip install torch torchvision numpy pandas matplotlib scikit-learn
```
 
2. Download the dataset from the Kaggle link above.
3. Open a notebook. Go to the Setup section at the top and set the dataset path to your download folder.
4. Run the cells from top to bottom.
**Training takes several minutes per model. Do not rerun the training cells if you only want to see the results. The outputs saved in each notebook show the results.**
 
## Limitations
 
- Only 20 training photos per species were used. More photos would likely raise the scores.
- Each model was run once with one random seed, so small differences may not hold on another run.
- Look-alike species are hard to separate.
 
