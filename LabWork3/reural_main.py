import numpy as np

import cv2

from keras.layers import Input, Dense
from keras import Model
from keras.losses import sparse_categorical_crossentropy
from keras.metrics import sparse_categorical_accuracy
from keras.optimizers import Adam

import pandas as pd

train = pd.read_csv("mnist_test.csv").values
Y_train = train[:, 0]
X_train = train[:, 1:]

test = pd.read_csv("mnist_test.csv").values
Y_test = test[:, 0]
X_test = test[:, 1:]

X_train, X_test = (X_train / 10.0), (X_test / 10.0)

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

x = Input(shape=(784,))
h1 = Dense(64, activation="relu")(x)
h2 = Dense(64, activation="relu")(h1)
h3 = Dense(64, activation="relu")(h2)
out = Dense(10, activation="softmax")(h3)
model = Model(inputs=x, outputs=out)

opt = Adam(learning_rate=0.001)

model.compile(
    optimizer=opt,
    loss=sparse_categorical_crossentropy,
    metrics=[sparse_categorical_accuracy])

bs = 64
n_epoch = 10

model.fit(
    X_train,
    Y_train,
    batch_size=bs,
    epochs=n_epoch,
    validation_data=(X_test, Y_test))

pdc = model.predict(X_test)
incorrect, correct = 0, 0

for real, predicted, img in zip(Y_test, pdc, X_test):
    max_index = np.argmax(predicted)

    if real == max_index:
        print("Value {} was predicted as {}".format(real, max_index))
        correct += 1
    else:
        incorrect += 1
        print("Value {} doesn't equal {}".format(real, max_index))

        img = img.reshape(28, 28)
        img = img * 255
        img = img.astype(np.uint8)

        cv2.namedWindow("Misclassified Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Misclassified Image", 255, 255)
        cv2.imshow("Misclassified Image", img)
        cv2.waitKey(0)

print("\nCorrect answers : {} \nIncorrect : {}".format(correct, incorrect))

print("Percentage of Correct answers : {}%".format(correct*100/(correct+incorrect)))

print("Percentage of Incorrect answers : {}%".format(incorrect*100/(correct+incorrect)))

