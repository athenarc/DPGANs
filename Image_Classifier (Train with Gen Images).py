import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import numpy
from PIL import Image
import os
import pickle

image_dim = 28

def load_image_dataset(directory):

    image_dataset = []

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        img = Image.open(filepath, 'r')
        pix_val = list(img.getdata())

        pix_val_flat = [sets[0] for sets in pix_val]

        start = 0
        end = 28
        final_image = []

        while (end <= 784):
            final_image.append(pix_val_flat[start:end])
            start = start + image_dim
            end = end + image_dim

        final_image = numpy.asarray(final_image)
        image_dataset.append(final_image)

    image_dataset = numpy.asarray(image_dataset).astype('float32')
    image_dataset /= 255

    image_dataset = image_dataset.reshape(image_dataset.shape[0], image_dim, image_dim, 1)
    print (image_dataset.shape)
    return image_dataset

CATEGORIES = [0, 1, 2, 3, 4,5, 6,7,8,9]

(temp1, temp2), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.reshape(x_test.shape[0], image_dim, image_dim, 1)
input_shape = (image_dim, image_dim, 1)
x_test = x_test.astype('float32')
x_test /= 255

x_train = load_image_dataset("Samples/Now") # Use generated images for testing instead of MNIST ones
with open ('labels', 'rb') as fp:
    y_train = pickle.load(fp)
y_train = numpy.asarray(y_train)

# Importing the required Keras modules containing model and layers
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(image_dim, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=10)
results = model.evaluate(x_test, y_test)
print('test loss, test acc:', results)
