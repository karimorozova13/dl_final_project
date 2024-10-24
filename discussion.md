This solution combines image data of pets with textual descriptions to predict their adoption speed. Below is a detailed breakdown of the approach.

1. Importing Libraries and Packages
   First, necessary libraries are imported. These include pandas and numpy for data manipulation, scikit-learn for splitting the dataset and evaluation, and PyTorch for building and training the neural network. Libraries for handling images (PIL and torchvision) and text processing (spacy and nltk) are also used to handle different data modalities.

2. Text Preprocessing
   For the text preprocessing pipeline:

HTML tags, email addresses, and URLs are removed from the descriptions.
Contractions (e.g., "can't") are expanded into their full forms.
Stopwords, except for negations like "not", are filtered out to avoid losing important meaning.
Tokenization is applied to split the text into words, followed by lemmatization to convert words into their base forms (e.g., "running" becomes "run"). This results in normalized text descriptions that are cleaner and more suitable for use in the model. 3. Loading and Preparing the Data
The CSV data is loaded, and the text preprocessing is applied to the description column. The target variable (AdoptionSpeed) is then encoded into numerical categories representing different adoption speeds. The dataset is split into training and validation sets using an 80/20 ratio to ensure that the model is properly validated.

4. Creating a Custom Dataset
   A custom PyTorch Dataset is created to handle the input data. This dataset reads both image files (based on PetID) and the preprocessed textual descriptions. It applies the necessary transformations to the images, such as resizing, normalizing, and converting them into tensors. For the training set, both the image, description, and the corresponding label (AdoptionSpeed) are returned, while for the test set, only the image and description are returned.

5. Image Transformations
   To prepare the images for the model, they are resized to 224x224 pixels to match the input size of the ResNet-18 architecture. The pixel values are normalized using the mean and standard deviation of ImageNet, as the ResNet-18 model is pretrained on this dataset.

6. Model Definition
   The core of the model is based on ResNet-18, a convolutional neural network pretrained on ImageNet. The final layer of ResNet is modified to output a 128-dimensional feature vector, which is then passed through two fully connected layers. These layers reduce the dimensions and produce the final output of 5 classes, representing the different adoption speed categories. The model also includes options for dropout (to reduce overfitting) and batch normalization (to stabilize and improve training).

7. Model Training
   The Adam optimizer and CrossEntropy loss function are used to train the model. During each epoch, the training loop processes batches of images and descriptions, performs forward passes through the network, computes the loss, and backpropagates to update the model’s weights. The loss is accumulated and tracked to monitor training progress.

8. Validation
   After training, the model is evaluated on the validation set. Predictions are generated for each validation batch, and these are compared with the true labels. The evaluation metric used is the Quadratic Weighted Kappa Score, which measures the agreement between predicted and actual adoption speeds, giving more weight to predictions that are closer to the correct label.

9. Test Prediction and Submission
   Once the model is trained, it is applied to the test set to generate predictions. These predictions, along with the corresponding PetID, are saved to a CSV file for submission. The final submission file includes each PetID and its predicted adoption speed.

Summary
This approach leverages both visual and textual data to predict pet adoption speed. The visual features are extracted using a pretrained ResNet-18 model, while the textual descriptions are processed through a series of natural language processing techniques. By combining these two data modalities, the model is trained to predict the adoption speed of pets and achieves competitive performance.
