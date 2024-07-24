# UTKFace Age and Gender Prediction
This project uses the UTKFace dataset to build a Convolutional Neural Network (CNN) model that predicts the age and gender of individuals from grayscale images

### Dataset
The UTKFace dataset consists of over 20,000 face images labeled by age, gender, and ethnicity. For this project, we focus on the age and gender labels.

### Setup and Dependencies
The following Python libraries are required to run the code:

pandas
numpy
matplotlib
seaborn
PIL
tensorflow
keras
scikit-learn
tqdm

# Code Explanation
### Import Libraries
The required libraries are imported, and some basic configurations are set up.

### Load and Process Data
Define Base Directory: The base directory for the UTKFace dataset is defined.
List Filenames: All filenames in the directory are listed.
Extract Labels: Using list comprehensions, image paths, age labels, and gender labels are extracted.
Create DataFrame: A pandas DataFrame is created to store the image paths, age labels, and gender labels.
Visualize Age Distribution: A boxplot of the age distribution is generated using seaborn.
### Feature Extraction
A function extract_features is defined to:

Load each image in grayscale mode.
Resize the image to 128x128 pixels.
Convert the image to a NumPy array and store it in a list.
Reshape the array to fit the input shape required by the CNN model.
### Prepare Data for Training
Normalize Data: The pixel values are normalized by dividing by 255.0.
Convert Labels to NumPy Arrays: The age and gender labels are converted to NumPy arrays.
Split Data: The data is split into training and testing sets using an 80/20 ratio.
### Build and Train CNN Model
Define Model Architecture: The model is built using Keras with multiple convolutional layers, max-pooling layers, a flatten layer, and fully connected layers.
Compile Model: The model is compiled with binary cross-entropy loss for gender prediction and mean absolute error (MAE) loss for age prediction.
Train Model: The model is trained using the training data with a batch size of 32 and 25 epochs. Validation split of 20% is used.
### Make Predictions and Evaluate
Select an Image: An image index is chosen to display and make predictions.
Print Original Labels: The original gender and age labels of the selected image are printed.
Make Predictions: The model makes predictions on the selected image.
Print Predicted Labels: The predicted gender and age labels are printed.
Display Image: The selected image is displayed using matplotlib.
