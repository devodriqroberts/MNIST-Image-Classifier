 # Import the libraries
import numpy as np
# Keras module layer
from keras.models import Sequential
# Keras core layers
from keras.layers import Dense, Dropout, Activation, Flatten
# Keras CNN layers
from keras.layers import Conv2D, MaxPooling2D
# Keras utilities
from keras.utils import np_utils

# Load the image data
from keras.datasets import mnist

# Load pre shuffled data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Plotting the first sample of the dataset
import matplotlib.pyplot as plt
plt.imshow(X_train[0])

# Reshaping the input data to declare depth of 1 (n, depth, width, height)
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

print(X_train.shape)

# Converting the d-types to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(y_train.shape)
print(y_train[:10])

# Convert the 1-dim array into 10 distinct class matrices 
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

print(y_train.shape)
print(y_train[:10])

# Defining Model Architecture
# Sequential Model
model = Sequential()

# Input Layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1,28,28), data_format='channels_first'))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam', 
#               metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=200, 
          epochs=2, 
          verbose=1)

score = model.evaluate(X_test, y_test, verbose=0)
print('Loss:',score[0], 'Accuracy:', score[1])