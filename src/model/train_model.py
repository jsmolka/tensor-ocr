import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.utils import to_categorical
from keras import backend as K

NIST_DIR = "D:/Users/Tom/Documents/NIST"
BATCH_SIZE = 128
NUM_CLASSES = 62
EPOCHS = 3
HEIGHT, WIDTH = 128, 128
CHANNELS = 3

h5_file = h5py.File(NIST_DIR+'/nist_C.h5', 'r')
nist_x = h5_file['nist_x'][:]
nist_y = h5_file['nist_y'][:]

# change when needed
np.random.shuffle(nist_x)

test_samples = int(nist_x.size * 0.1)

test_x = nist_x[:test_samples-1]
train_x = nist_x[test_samples:]

test_y = nist_y[:test_samples-1]
train_y = nist_y[test_samples:]

if K.image_data_format() == 'channels_first':
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[3], train_x[1], train_x[2])
    test_x = test_x.reshape(train_x.shape[0], train_x.shape[3], train_x[1], train_x[2])
    input_shape = (CHANNELS, HEIGHT, WIDTH)
else:
    input_shape = (HEIGHT, WIDTH, CHANNELS)

train_x = train_x.astype('float32')
test_x = test_x.astype('float32')

train_x /= 255
test_x /= 255

mean = np.std(train_x)
train_x -= mean
test_x -= mean

train_y = to_categorical(train_y, NUM_CLASSES, dtype=int)
test_y = to_categorical(test_y, NUM_CLASSES, dtype=int)

# experimental; untested!
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())  # flattens the rank 4 tensor to a rank 2
model.add(Dense(128, activation='relu'))  # Dense = fully connected Layer
model.add(Dropout(0.5))
model.add(Dense(units=NUM_CLASSES, activation='softmax'))  # output !classification! layer using softmax

model.compile(loss=categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])

# train CNN
history = model.fit(train_x,
                    train_y,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=(test_x, test_y))

# evaluate on test set
score = model.evaluate(test_x, test_y, verbose=0)
print(model.metrics_names)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# visualize the loss function in each epoch
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values)+1)
plt.plot(epochs, loss_values, 'bo')
plt.plot(epochs, val_loss_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# visualize accuracy in each epoch
plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'bo')
plt.plot(epochs, val_acc_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
