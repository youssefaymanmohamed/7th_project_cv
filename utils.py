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
    for root, _, files in os.walk(folder):
        for file in files:
            img_path = os.path.join(root, file)
            if os.path.isfile(img_path) and img_path.endswith(('png', 'jpg', 'jpeg')):
                img = Image.open(img_path).resize(target_size)
                img = np.array(img)
                images.append(img)
    return np.array(images)

def extract_color_histogram(image, bins=(8, 8, 8)):
    """
    Extract color histogram from an image.

    Parameters:
    image (np.array): Input image.
    bins (tuple): Number of bins for each color channel.

    Returns:
    np.array: Flattened color histogram.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_lbp(image, num_points=24, radius=8):
    """
    Extract Local Binary Patterns (LBP) from an image.

    Parameters:
    image (np.array): Input image.
    num_points (int): Number of points for LBP.
    radius (int): Radius for LBP.

    Returns:
    np.array: LBP histogram.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, num_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False):
    """
    Extract Histogram of Oriented Gradients (HOG) from an image.

    Parameters:
    image (np.array): Input image.
    pixels_per_cell (tuple): Size of the cell.
    cells_per_block (tuple): Number of cells per block.
    visualize (bool): Whether to visualize the HOG.

    Returns:
    np.array: HOG features.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hog_features = hog(gray, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=visualize)
    return hog_features

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

