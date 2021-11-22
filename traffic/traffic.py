import cv2
from cv2 import data
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images, labels = [],[]
    for n in range(0, NUM_CATEGORIES):
        folder = os.path.join(data_dir, str(n))
        for filename in os.listdir(folder):
            # append filename to get the file location
            file = os.path.join(folder, filename)
            # reads a colour image to numpy n-dimensional array in BGR format
            img = cv2.imread(file)
            # converting BGR to RGB format
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # resizing the image
            img_final = cv2.resize(img, dsize = (IMG_WIDTH, IMG_HEIGHT))
            if img_final is not None:
                images.append(img_final)
                labels.append(n)
    return (images, labels)

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Sequential model
    model = tf.keras.Sequential()
    # Adding convolutional layer with 32 filters using 3*3 kernel
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    # Adding max-pooling layer with pool size 2*2
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Adding another convolutional layer (64 layers, 3*3 kernel) and max-pooling layer (pool size 3*3)
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))
    # Adding dropout to the pool layer
    model.add(tf.keras.layers.Dropout(0.25))

    # Flattening
    model.add(tf.keras.layers.Flatten())
    
    # Adding hidden layer with dropout
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.25))

    # Adding output layer
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))

    # Training the Neural Network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    main()
