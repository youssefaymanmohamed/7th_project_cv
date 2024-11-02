import os
import numpy as np
import cv2
from PIL import Image
from skimage.feature import local_binary_pattern, hog
from tensorflow.keras.applications.vgg16 import preprocess_input
import requests
import zipfile
from io import BytesIO



def load_images_from_folder(folder, target_size=(224, 224)):
    """
    Load images from a directory and its subdirectories.

    Parameters:
    folder (str): Path to the directory containing images.
    target_size (tuple): Target size to resize images.

    Returns:
    np.array: Array of loaded images.
    """
    images = []
    for root, _, files in os.walk(folder): # os.walk() generates the file names in a directory tree by walking either top-down or bottom-up
        for file in files: 
            img_path = os.path.join(root, file) # join the root directory and the file name to get the full path of the image
            if os.path.isfile(img_path) and img_path.endswith(('png', 'jpg', 'jpeg')):
                img = Image.open(img_path).resize(target_size) # open the image and resize it to the target size
                img = np.array(img)
                images.append(img)
    return np.array(images)

def extract_color_histogram(images):
    """
    Extracts color histograms from a list of images.

    This function converts each image from RGB to HSV color space, calculates the histogram
    for each image, normalizes the histogram, and then flattens it. The resulting histograms
    are returned as a list.

    Parameters:
    images (list of numpy.ndarray): A list of images in RGB color space.

    Returns:
    list of numpy.ndarray: A list of flattened and normalized histograms for each image.
    """
    histograms = []
    for img in images:
        # Convert the image from RGB to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # Calculate the histogram of the image
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])  # hsv channels, 0 ,1, 2 are the channels of hsv | none is the mask |  8, 8, 8 are the number of bins for each channel | 0, 256, 0, 256, 0, 256 are the ranges for each channel
        # Normalize the histogram
        hist = cv2.normalize(hist, hist).flatten()  # normalize the histogram and flatten it
        histograms.append(hist) # append the histogram to the list of histograms for all images  
    return histograms

# Calculate LBP features for the images
def extract_lbp(images):
    """
    Extracts Local Binary Pattern (LBP) features from a list of images.

    Parameters:
    images (list of numpy.ndarray): List of images in RGB format.

    Returns:
    list of numpy.ndarray: List of normalized histograms of LBP features for each image.
    """
    lbp_features = []
    for img in images:
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Calculate the LBP features
        lbp = local_binary_pattern(gray, 8, 1, method='uniform') # 8 neighbors, radius 1, uniform method is used to get a uniform pattern
        # Calculate the histogram of the LBP image
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 60), range=(0, 59)) # 59 bins for the histogram | range is 0 to 59  | ravel() is used to flatten the array 
        # Normalize the histogram
        hist = hist.astype('float') / hist.sum()
        lbp_features.append(hist)
    return lbp_features

# Calculate HOG features for the images
def extract_hog(images):
    """
    Extract Histogram of Oriented Gradients (HOG) features from a list of images.

    Parameters:
    images (list of numpy.ndarray): List of images in RGB format.

    Returns:
    list of numpy.ndarray: List of HOG feature arrays for each image.
    """
    hog_features = []
    for img in images:
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Calculate the HOG features
        hog_feat = hog(gray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2-Hys') # 8 orientations, 16x16 pixels per cell, 1x1 cells per block, L2-Hys block normalization
        hog_features.append(hog_feat)
    return hog_features

def extract_SIFT_features(images):
    """
    Extracts SIFT (Scale-Invariant Feature Transform) features from a list of images.

    Parameters:
    images (list of numpy.ndarray): A list of images in BGR format.

    Returns:
    list of numpy.ndarray: A list where each element is an array of SIFT descriptors for the corresponding image.
    """
    sift = cv2.SIFT_create() # create a SIFT object
    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None) # detect keypoints and compute SIFT descriptors for the image
        features.append(des)
    return features

def extract_deep_features_vgg16(images, model):
    """
    Extract deep features using VGG16.

    Parameters:
    images (np.array): Array of images.
    model (tf.keras.Model): Pre-trained VGG16 model.

    Returns:
    np.array: Flattened deep features.
    """
    images_preprocessed = preprocess_input(images)
    features = model.predict(images_preprocessed)
    features_flattened = features.reshape((features.shape[0], -1))
    return features_flattened


# Function to extract deep features using ResNet50
def extract_deep_features_resnet(images):
    """
    Extract deep features from images using a pre-trained ResNet model.

    Args:
        images (numpy.ndarray): A batch of images to process. The shape should be 
                                (batch_size, height, width, channels).

    Returns:
        numpy.ndarray: Flattened deep features extracted from the images. The shape 
                       will be (batch_size, features_dim), where features_dim is the 
                       total number of features extracted from each image.
    """
    # Preprocess the images
    images_preprocessed = preprocess_input(images)
    # Extract features
    features = resnet_model.predict(images_preprocessed)
    # Flatten the features
    features_flattened = features.reshape((features.shape[0], -1))
    return features_flattened



# Function to extract deep features using InceptionV3
def extract_deep_features_inception(images):
    """
    Extract deep features from images using the Inception model.

    Args:
        images (numpy.ndarray): A batch of images to extract features from. 
                                The shape should be (batch_size, height, width, channels).

    Returns:
        numpy.ndarray: Flattened deep features extracted from the images. 
                       The shape will be (batch_size, features).
    """
    # Preprocess the images
    images_preprocessed = preprocess_input(images)
    # Extract features
    features = inception_model.predict(images_preprocessed)
    # Flatten the features
    features_flattened = features.reshape((features.shape[0], -1))
    return features_flattened

def evaluate_features(features, labels):
    """
    Evaluate the performance of k-Nearest Neighbors (k-NN) classifier on the given features and labels.

    This function splits the data into training and testing sets, trains a k-NN classifier on the training set,
    predicts the labels for the test set, and calculates the accuracy of the predictions.

    Parameters:
    features (array-like): The input features for the dataset.
    labels (array-like): The corresponding labels for the dataset.

    Returns:
    float: The accuracy of the k-NN classifier on the test set.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    # Initialize the k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    # Train the classifier
    knn.fit(X_train, y_train)
    # Predict the labels for the test set
    y_pred = knn.predict(X_test)
    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

