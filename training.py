from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np
import cv2
import glob
from sklearn.model_selection import train_test_split

model = keras.Sequential(
    [layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation='relu', input_shape=[200, 200, 1]),
     layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
     layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation='relu'),
     layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
     layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation='relu'),
     layers.Flatten(),
     layers.Dense(32, activation='relu'),
     layers.Dense(17, activation='softmax')
     ]
)
inputPath = "TrainingData_traindata"
listDirs = os.listdir(inputPath)
listImages = []
labels = []
for dir in listDirs:
    print(dir)
    videos = glob.glob(os.path.join(inputPath, dir, "*.png"))
    for im_path in videos:
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = im.reshape(200, 200, 1)
        listImages.append(im)
        lab = np.zeros((17))
        lab[int(dir)] = 1
        labels.append(lab)

listImages = np.array(listImages) / 255.0
labels = np.array(labels)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

data_train, data_test, labels_train, labels_test = train_test_split(listImages, labels, test_size=0.20, shuffle=True,
                                                                    random_state=42)
model.fit(x=data_train, y=labels_train, batch_size=32, epochs=10, shuffle=True, validation_data=(data_test, labels_test))
model.save("cnn_new.h5")
