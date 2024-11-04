import cv2 
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image as tf_image
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import local_binary_pattern
import time
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
import time

# Color Histogram
def extract_color_histogram(image, bins=(8, 8, 8)):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Compute the color histogram
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    # Normalize the histogram
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Extract texture features 
def extract_texture_features(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Extract LBP features
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    # Compute the histogram of LBP features
    hist, _ = np.histogram(lbp, bins=np.arange(0, 27), range=(0, 26))
    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_hog (img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Compute HOG features
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(gray)
    return hog_features



def extract_vgg16_features(img):
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Resize the image to 224x224
    img = cv2.resize(img, (224, 224))
    # Convert the image to an array
    img_array = tf_image.img_to_array(img)
    # Expand the dimensions of the image
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess the image
    img_array = preprocess_input(img_array)
    # Extract features using the VGG16 model
    features = vgg16_model.predict(img_array)
    # Flatten the features
    features = features.flatten()
    return features

def evaluate_retrieval(query_features, dataset_features, dataset_labels, top_k=10):
    start_time = time.time()
    
    # Calculate cosine similarities between the query and dataset features
    similarities = cosine_similarity(query_features.reshape(1, -1), dataset_features).flatten()
    
    # Get the indices of the top-k most similar images
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    
    # Calculate precision at 1
    precision_at_1 = precision_score([dataset_labels[top_k_indices[0]]], [1])
    
    # Calculate precision at 10
    precision_at_10 = precision_score([dataset_labels[i] for i in top_k_indices], [1] * top_k)
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve([dataset_labels[i] for i in top_k_indices], similarities[top_k_indices])
    
    # Calculate retrieval time
    retrieval_time = time.time() - start_time
    
    return precision_at_1, precision_at_10, precision, recall, retrieval_time