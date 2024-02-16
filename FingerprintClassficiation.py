import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os
import glob

data_directory = 'SOCOFing/Real' 

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (128, 128))
    return img.flatten()

def extract_features(image):
    return image

def load_dataset(data_directory):
    features = []
    labels = []
    for image_path in glob.glob(os.path.join(data_directory, '*.BMP')):
        preprocessed_image = preprocess_image(image_path)
        feature_vector = extract_features(preprocessed_image)
        features.append(feature_vector)
        gender_label = 'M' if '_M_' in image_path else 'F'
        labels.append(gender_label)
    return np.array(features), np.array(labels)

features, labels = load_dataset(data_directory)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print(classification_report(y_test, predictions))
