import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from PIL import TiffImagePlugin
TiffImagePlugin.USE_LIBTIFF = False

# Constants
POLLEN_DIR = "../Dataset"
LBP_POINTS = 8
LBP_RADIUS = 1
HSV_BINS = 16

# Feature extraction
def extract_hsv_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [HSV_BINS], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [HSV_BINS], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [HSV_BINS], [0, 256]).flatten()
    return np.concatenate([hist_h, hist_s, hist_v])

def extract_lbp_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    return hist / (hist.sum() + 1e-6)

from PIL import Image

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

import sys
import os
import contextlib

@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as fnull:
        stderr = sys.stderr
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stderr = stderr

def read_image(image_path):
    try:
        with suppress_stderr():
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Failed to read {image_path}: {e}")
        return None

def extract_features(image_path):
    image = read_image(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (256, 256))
    hsv = extract_hsv_features(image)
    lbp = extract_lbp_features(image)
    return np.concatenate([hsv, lbp])

# Load data
features, labels = [], []
label_names = sorted([d for d in os.listdir(POLLEN_DIR) if os.path.isdir(os.path.join(POLLEN_DIR, d))])

for idx, label in enumerate(label_names):
    folder = os.path.join(POLLEN_DIR, label)
    for file in os.listdir(folder):
        if file.lower().endswith(VALID_EXTENSIONS):
            path = os.path.join(folder, file)
            feat = extract_features(path)
            if feat is not None:
                features.append(feat)
                labels.append(idx)

X = np.array(features)
y = np.array(labels)

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train SVM
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)

# Evaluation
from sklearn.utils.multiclass import unique_labels

y_pred = svm.predict(X_test)
used_labels = unique_labels(y_test, y_pred)
used_target_names = [label_names[i] for i in used_labels]
print(classification_report(y_test, y_pred, target_names=used_target_names, labels=used_labels))

# Save model
joblib.dump(svm, 'svm_model.joblib')
joblib.dump(scaler, 'scaler.joblib')