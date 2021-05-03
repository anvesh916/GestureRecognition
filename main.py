"""
@author: Voona Anvesh Kumar
"""
import cv2
import numpy as np
import os
from handshape_feature_extractor import HandShapeFeatureExtractor
from scipy import spatial
model = HandShapeFeatureExtractor.get_instance()


def generatePenultimateLayer(inputPathName):
    videos = []
    for fileName in os.listdir(inputPathName):
        if fileName.endswith(".mp4"):
            videos.append(os.path.join(inputPathName, fileName))
    featureVectors = []
    print("Extracting Frames of " + inputPathName)
    for video in videos:
        frame = frameExtractor(video)
        feature = model.extract_feature(frame)
        featureVectors.append(feature)
    return featureVectors


def frameExtractor(videopath):
    cap = cv2.VideoCapture(videopath)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    cap.set(1, int(video_length / 2.5))
    ret, frame = cap.read()
    return frame

# =============================================================================
# Get the penultimate layer for training data
# =============================================================================
training_layer = generatePenultimateLayer("traindata")

# =============================================================================
# Get the penultimate layer for test data (Our Data)
# =============================================================================
testing_layer = generatePenultimateLayer("test")

# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================
#
featureLabel = []
cosineSimilarity = []
for test in testing_layer:
    for train in training_layer:
        dist = spatial.distance.cosine(test, train)
        cosineSimilarity.append(dist)
    featureLabel.append(int(cosineSimilarity.index(min(cosineSimilarity))))
    cosineSimilarity = []

totalCorrect = 0
for i, label in enumerate(featureLabel):
    if label == (i % 17):
        totalCorrect += 1
print("Total Correct are : " + str(totalCorrect) + "/" + str(len(testing_layer)))
print("Accuracy is =" + str((100 * totalCorrect) / len(testing_layer)))

np.savetxt("Results.csv", featureLabel, fmt="%d")

