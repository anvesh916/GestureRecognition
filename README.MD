## Gesture Recognition - MC Final Exam

# Penultimate Layer Generation & Feature Extraction
- Read the video files in the directory passed ( test/traindata) and then use the frame extractor function  to extract the middle frame of the whole video 
- Store the frames in a new directory for further calculation
- For each of the frames, extract the hand feature using the pre existing CNN model provided in the source code using 
```python
	HandShapeFeatureExtractor.get_instance().extract_feature(<IMG>)
```
# Cosine Similarity
Once the above feature vectors are generated we can use the cosine similarity and return the index of the vector which is having the least distance.
	
```python
spatial.distance.cosine(<test>, <train>)
```
# Output
Print the output in a results.csv file

# Generate Training Data
To generate the training images in 200 by 200 format in a directory based on the labels format


# Training & Model Creation
Create CNN model with the training data. Then model fit with the test and train data. Save the model to a file.
