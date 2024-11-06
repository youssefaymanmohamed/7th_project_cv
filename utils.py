import cv2 
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image as tf_image
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import local_binary_pattern
import time
from skimage.feature import hog
from skimage import exposure
from sklearn.metrics import precision_score, recall_score, precision_recall_curve

def Calculate_color_histograms(image, bins=(64, 64, 64)):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Extract texture features of locat binary pattern
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

def compute_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features, hog_image = hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        transform_sqrt=True,
        visualize=True
    )
    return hog_features, hog_image

def sift_describe_and_normalize(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sift = cv2.SIFT_create()
    keypoints, features = sift.detectAndCompute(image, None)
    
    if features is None:
        return keypoints, None

    # Normalize features
    features /= features.sum(axis=1, keepdims=True) + 1e-7
    features = np.sqrt(features)
    features /= np.linalg.norm(features)
    
    return keypoints, features

def flann_matcher(
    query_descriptors: np.ndarray,
    train_descriptors: np.ndarray,
    k: int = 2,
    ratio: float = 0.9,
) -> np.ndarray:


    # Create the FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Convert the descriptors to numpy arrays
    query_descriptors = np.array(query_descriptors, np.float32)
    train_descriptors = np.array(train_descriptors, np.float32)

    # Compute the matches
    try:
        matches = flann.knnMatch(query_descriptors, train_descriptors, k=k)
    except Exception as e:
        return []

    # Apply the ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return good_matches