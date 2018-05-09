# test basic multilayer perceptron
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
np.random.seed(7)

# network and training
# The number of times the model is exposed to the training set(the number of iterations).
NB_EPOCH = 50
# The number of intances observed before the optimizer performs a weight update.
BATCH_SIZE = 128
VERBOSE = 1
# The number of neuron in the hidden layer
NB_HIDDEN = 128
DROPOUT = 0.2
NB_CLASSES = 10  # number of outputs = number of digits
OPTIMIZER = Adam()  # stochastic gradient optimizer
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2  # how much training examples should be reserved for validation
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape 60000*28*28 to 60000*784
X_train.shape
RESHAPED = 784
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# definition of the model
model = Sequential()
model.add(Dense(NB_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation(('softmax')))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy']
              )

# training the neural network
history = model.fit(X_train, Y_train,
                    batch_size=BATCH_SIZE,
                    epochs=NB_EPOCH,
                    verbose=VERBOSE, validation_split=VALIDATION_SPLIT
                    )

# evaluation of the model
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("Test score: ", score[0])
print("Test accuracy: ", score[1])
