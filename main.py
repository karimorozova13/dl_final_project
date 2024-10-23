import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import cohen_kappa_score

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from PIL import Image

import os
import glob
import re

import spacy
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from tqdm.auto import tqdm
tqdm.pandas()

contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}

stop_words = set(stopwords.words('english')).union({'also', 'would', 'much', 'many'})
negations = {
    'aren',
    "aren't",
    'couldn',
    "couldn't",
    'didn',
    "didn't",
    'doesn',
    "doesn't",
    'don',
    "don't",
    'hadn',
    "hadn't",
    'hasn',
    "hasn't",
    'haven',
    "haven't",
    'isn',
    "isn't",
    'mightn',
    "mightn't",
    'mustn',
    "mustn't",
    'needn',
    "needn't",
    'no',
    'nor',
    'not',
    'shan',
    "shan't",
    'shouldn',
    "shouldn't",
    'wasn',
    "wasn't",
    'weren',
    "weren't",
    'won',
    "won't",
    'wouldn',
    "wouldn't"
}
stop_words = stop_words.difference(negations)
stemmer = PorterStemmer()
nlp = spacy.load("en_core_web_sm", disable = ['parser','ner'])

# Data normalization function
def normalize_text(raw_review):
    if not isinstance(raw_review, str):
        return ""
    
    text = re.sub("<[^>]*>", " ", raw_review) 
    
    text = re.sub("\S*@\S*[\s]+", " ", text) 
    
    text = re.sub("https?:\/\/.*?[\s]+", " ", text) 
        
    text = text.lower().split()
    
    text = [contractions.get(word) if word in contractions else word 
            for word in text]
   
    text = " ".join(text).split()    
    
    text = [word for word in text if not word in stop_words]

    text = " ".join(text)
          
    text = re.sub("[^a-zA-Z' ]", "", text) 

    doc = nlp(text)
    text = " ".join([token.lemma_ for token in doc if len(token.lemma_) > 1 ])
    
    text = re.sub("[\s]+", " ", text)    
    
    return(text)

# Load data
df = pd.read_csv('./datasets/train.csv')
df['des_normalized'] = df['Description'].apply(normalize_text)

# Encoding labels
label_encoder = LabelEncoder()
df['AdoptionSpeed'] = label_encoder.fit_transform(df['AdoptionSpeed'])

# Train-test split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
image_dir_train = './datasets/images/images/train'
image_dir_test = './datasets/images/images/test'
class PetAdoptionDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, is_test=False):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx] 
        pet_id = row['PetID'] 
 
        # Search for any image file that starts with the PetID 
        image_pattern = os.path.join(self.image_dir, f"{pet_id}-*.jpg") 
        matching_images = glob.glob(image_pattern) 
 
       
        if not matching_images: 
            matching_images = glob.glob(os.path.join(self.image_dir, f"0{pet_id}-*.jpg")) 
 
        image_path = matching_images[0] 
        image = Image.open(image_path).convert('RGB') 

        if self.transform: 
            image = self.transform(image) 

        if not self.is_test:  # If it's not a test set, return label
            description = row['des_normalized']
            label = row['AdoptionSpeed']
            return image, description, label
        else:
            description = row['des_normalized']
            return image, description  

# Transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = PetAdoptionDataset(train_df, image_dir=image_dir_train, transform=transform)
val_dataset  = PetAdoptionDataset(val_df, image_dir=image_dir_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Model Definition
class PetAdoptionModel(nn.Module):
    def __init__(self, dropout_rate, use_batchnorm):
        super(PetAdoptionModel, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128)  # Modify the final layer
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 5)  # Assuming AdoptionSpeed has 5 classes
        self.dropout = nn.Dropout(dropout_rate)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn1 = nn.BatchNorm1d(64)

    def forward(self, images):
        x = self.resnet(images)
        x = torch.relu(self.fc1(x))
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.dropout(x)
        return self.fc2(x)

lr = 1.1221703972084995e-05
dropout_rate = 0.11
use_batchnorm = False
batch_size = 256
num_epochs = 10

# Model training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PetAdoptionModel(dropout_rate, use_batchnorm).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, descriptions, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

# Validation loop
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

test_df = pd.read_csv('./datasets/test.csv')
test_df['des_normalized'] = test_df['Description'].apply(normalize_text)

test_dataset = PetAdoptionDataset(test_df, image_dir=image_dir_test, transform=transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
predictions = []

with torch.no_grad():
    model.eval()
    for images, descriptions in test_loader:  
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        
submission = pd.DataFrame({
    'PetID': test_df['PetID'].astype(str),
    'AdoptionSpeed': predictions
})

submission.to_csv(os.path.join('./datasets', 'submission.csv'), index=False)