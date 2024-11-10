import tensorflow as tf
import numpy as np
from tensorflow import keras

EPOCHS = 200 
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10 # number of outputs = number of digits
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2 # how much TRAIN is reserved for VALIDATION
DROPOUT = 0.3 #randomized drop out to generalize the program

# Loading MNIST dataset
# verify
# You can verify that the split between train and test is 60,000 and 10,000 respectively
# Labels have one-hot representation.is automatically applied
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train is 60000 rows of 28x28 values; we  --> reshape it to 60000 x 784
RESHAPED = 784

x_train = x_train.reshape(60000, RESHAPED)
x_test = x_test.reshape(10000, RESHAPED)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize inputs to be within [0,1]
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# one-hot representation of the labels.
y_train = tf.keras.utils.to_categorical(y_train, NB_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NB_CLASSES)

# Build the model
model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(N_HIDDEN,
                             input_shape=(RESHAPED,),
                             name='dense_layer',
                             activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(N_HIDDEN,
                             input_shape=(RESHAPED,),
                             name='dense_layer_2',
                             activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(NB_CLASSES,
                             name='dense_layer_3', activation='softmax'))

# summary of the model
model.summary()

# compiling the model
model.compile(optimizer='RMSProp', #optimizer function used after each epoch(times exposed to training set) adjusts weights so that objective function is minimized
                                   #optimizers allow you to not get "stuck" in local minima in order to find glabal minima
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
model.fit(x_train, y_train,
          batch_size=BATCH_SIZE, epochs=EPOCHS,
          verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# evalute the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('test accuracty', test_acc)