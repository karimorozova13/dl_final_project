# Pet Adoption Speed Prediction

This project aims to predict the adoption speed of pets using both image and textual data. It leverages deep learning with **ResNet18** for image classification and Natural Language Processing (NLP) techniques for text feature extraction to classify pets into five adoption speed categories.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Training the Model](#2-training-the-model)
  - [3. Evaluation](#3-evaluation)
  - [4. Prediction on Test Data](#4-prediction-on-test-data)
- [Model Architecture](#model-architecture)
- [Hyperparameter Optimization](#hyperparameter-optimization)

## Project Overview

In this project, we predict the speed at which a pet is adopted using:
- **Image Data**: Pet images processed with a ResNet18 architecture pre-trained on ImageNet.
- **Text Data**: Descriptions of the pets processed with various NLP techniques including lemmatization, stopword removal, and contraction handling.

The model outputs a class representing the adoption speed:
- `0`: Same day adoption
- `1`: 1 to 7 days
- `2`: 8 to 30 days
- `3`: 31 to 90 days
- `4`: Over 100 days

The project is evaluated using the **Quadratic Weighted Kappa** metric, which measures the agreement between predicted and actual adoption speeds.

## Features

- Deep learning with **ResNet18** for image classification.
- NLP preprocessing for text data including **contraction handling**, **stopword removal**, and **lemmatization** using **spaCy**.
- Hyperparameter tuning using **Optuna**.
- Model evaluation using **Cohen Kappa Score**.

## Installation

### 1. Clone the repository
First, clone this repository to your local machine:
```bash
git clone https://github.com/karimorozova13/pet-adoption-speed-prediction.git
cd pet-adoption-speed-prediction
```
### 2. Install dependencies
You can install all required dependencies using pip:

```bash
pip install -r requirements.txt
```
Make sure you have the necessary resources for NLP and stopwords:

``` bash
# Download spaCy English model
python -m spacy download en_core_web_sm

# Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"
```

### 3. Set up GPU (Optional)
If you have a GPU available, you can configure PyTorch to use it. CUDA is automatically used if available. No additional setup is needed.

## Dataset
### 1. Dataset Structure
The project expects the dataset to be organized as follows:

* Train images: ./datasets/images/images/train/
* Test images: ./datasets/images/images/test/
* CSV files:
  - train.csv: Contains the training data with columns PetID, Description, and AdoptionSpeed.
  - test.csv: Contains the test data with columns PetID and Description.
### 2. Dataset Download
Make sure the dataset is available in the specified directory structure. If you're using your own dataset, ensure that it's in a similar format.

## Usage
### 1. Data Preprocessing
First, preprocess the textual data by normalizing the pet descriptions:

``` bash
# Load the data
import pandas as pd
df = pd.read_csv('./datasets/train.csv')

# Apply the normalization function to the description column
df['des_normalized'] = df['Description'].apply(normalize_text)
```

The normalize_text() function removes unwanted characters, applies lemmatization, and handles contractions and stopwords.

### 2. Training the Model
Once the data is preprocessed, you can train the model using the following steps:

1. Prepare DataLoader for training and validation data:

``` bash
train_dataset = PetAdoptionDataset(train_df, image_dir=image_dir_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
```

2. Train the model by running the train.py script:

The training loop will display the loss at each epoch, and after each epoch, the model is evaluated on the validation set.

### 3. Evaluation
The model's performance is evaluated using the Quadratic Weighted Kappa Score, which is printed after each epoch. To manually evaluate the trained model on the validation set, you can add the following evaluation loop:

``` bash
model.eval()
val_preds = []
val_labels = []

with torch.no_grad():
    for images, descriptions, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        val_preds.extend(preds.cpu().numpy())
        val_labels.extend(labels.numpy())

kappa_score = cohen_kappa_score(val_labels, val_preds, weights='quadratic')
print(f'Validation Quadratic Weighted Kappa: {kappa_score:.4f}')
```

### 4. Prediction on Test Data
* Load the test data.
* Use the trained model to make predictions.
* Create a CSV file in the format required for submission.

## Model Architecture
The model architecture is based on ResNet18, a popular convolutional neural network for image classification, modified as follows:

* The final fully connected layer of ResNet is replaced with:
  * A fully connected layer to reduce the dimension to 128.
  * Another layer reducing it to 64 dimensions.
  * The output layer for the 5 classes corresponding to adoption speeds.

### Dropout & Batch Normalization:
The model includes:

* A Dropout layer to avoid overfitting (dropout rate is a tunable hyperparameter).
* Batch Normalization (optional) to stabilize training.
  
### Optimizer & Loss:
* Optimizer: Adam
* Loss Function: CrossEntropyLoss for multi-class classification.

## Hyperparameter Optimization
Hyperparameter tuning is performed using Optuna, a framework for automated hyperparameter optimization.

In train.py, you can define the study for tuning parameters such as:

* Learning rate
* Dropout rate
* Whether to use batch normalization
Example of an Optuna study:

``` bash
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```
