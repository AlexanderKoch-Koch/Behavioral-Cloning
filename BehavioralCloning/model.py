import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Conv2D, Lambda, Dropout

epochs = 5
batch_size = 150
use_old_model = True
model_file_name = "model3.h5"

#read data
csvfile =  open("/home/alex/Documents/udacity-simulator/sim-data/lake-left/driving_log.csv")
reader = list(csv.reader(csvfile))
samples = []
n_samples = len(reader)
X_train = np.zeros(shape=[n_samples, 160, 320, 3])
y_train = np.zeros(shape=[n_samples])
for i in range(n_samples):
    X_train[i] = cv2.imread(reader[i][0])
    y_train[i] = reader[i][3]
print("number samples: " + str(n_samples))


if(use_old_model):
    model = load_model(model_file_name)
else:
    model = Sequential([
        Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)),
        Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation="relu"),
        Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation="relu"),
        Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation="relu"),
        Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation="relu"),
        Flatten(),
        Dense(300),
        Dropout(0.5),
        Dense(100),
        Dropout(0.5),
        Dense(40),
        Dropout(0.5),
        Dense(1)
    ])
    model.compile(loss="mse", optimizer="adam")

print(y_train.shape)

model.fit(X_train, y_train, epochs=epochs, validation_split=0.1, shuffle=True, batch_size=batch_size)

model.save(model_file_name)
