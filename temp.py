import os
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras.utils import to_categorical
import keras
from keras.layers import Concatenate
import tensorflow as tf
from icecream import ic

matplotlib.rcParams['interactive'] == True

def createData(yesDirectory, noDirectory, allDirectory):
    yesFileNames = os.listdir(yesDirectory)
    noFileNames = os.listdir(noDirectory)
    
    yesFileLabels = np.ones(len(yesFileNames))
    noFileLabels = np.zeros(len(noFileNames))

    allFileNames = yesFileNames + noFileNames
    allFileNames = np.asarray(allFileNames)
    allLabels = np.concatenate((yesFileLabels, noFileLabels), axis=0)
        
    sizes = [Image.open(allDirectory + "/" + f, 'r').size for f in allFileNames]
    maxSize = max(sizes)
    centerSize = tuple([int(x/2) for x in maxSize])
    fourthSize = tuple([int(x/4) for x in maxSize])
    eigthSize = tuple([int(x/8) for x in maxSize])
    allImages = []
    
    for imgName in allFileNames:
        bgImage = Image.new("L", centerSize, color ="black")
        currImage = Image.open(allDirectory + "/" + imgName)
        imgOffset = tuple([int(x/2) for x in currImage.size])
        imgCoor= np.subtract(fourthSize, imgOffset)
        bgImage.paste(currImage, tuple(imgCoor))
        bgImage = bgImage.resize(eigthSize)
        allImages.append(np.array(bgImage)/255)
    
    train_images, test_images, train_labels, test_labels = train_test_split(allImages, allLabels, test_size=0.33, random_state=42)
    train_images = np.asarray(train_images, dtype="float64")
    test_images = np.asarray(test_images, dtype="float64")

    return train_images, test_images, train_labels, test_labels

train_images, test_images, train_labels, test_labels = createData("brain_tumor_dataset_after_augmentation\\bt_yes","brain_tumor_dataset_after_augmentation\\bt_no", "brain_tumor_dataset_after_augmentation\\bt_all")

train_labels = to_categorical(train_labels.astype("float32"))
test_labels = to_categorical(test_labels.astype("float32"))
test_images = test_images.reshape((1538,135,240,1)).astype("float32")
train_images = train_images.reshape((3122,135,240,1)).astype("float32")

cnn_input = layers.Input(shape=(135, 240, 1))
rnn_input = layers.Input(shape=(135, 240))

cnn = models.Sequential()
cnn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(135, 240,1)))
cnn.add(layers.MaxPooling2D((2, 2)))
cnn.add(layers.Dense(8, activation='relu'))
cnn.add(layers.Dense(2, activation='sigmoid'))
cnn.add(layers.Flatten())

cnn.add(layers.Dense(8, activation='relu'))
cnn.add(layers.Dense(2, activation='sigmoid'))
cnn.add(layers.Flatten())

cnn.add(layers.Dense(8, activation='relu'))
cnn.add(layers.Dense(2, activation='sigmoid'))
cnn.add(layers.Flatten())

cnn_output = cnn(cnn_input)

cnn.summary()   

lstm_rnn = models.Sequential()
lstm_rnn.add(keras.Input(shape=(135, 240)))
lstm_rnn.add(layers.LSTM(32, activation='relu'))
lstm_rnn.add(layers.Dense(8, activation='relu'))
lstm_rnn.add(layers.Dense(2, activation='sigmoid'))
lstm_rnn.add(layers.Flatten())

lstm_rnn.add(layers.Dense(8, activation='relu'))
lstm_rnn.add(layers.Dense(2, activation='sigmoid'))
lstm_rnn.add(layers.Flatten())

lstm_rnn.add(layers.Dense(8, activation='relu'))
lstm_rnn.add(layers.Dense(2, activation='sigmoid'))
lstm_rnn.add(layers.Flatten())

rnn_output = lstm_rnn(rnn_input)

lstm_rnn.summary()

combined = Concatenate()([cnn_output, rnn_output])

hidden = layers.Dense(2, activation='sigmoid')(combined)

model = models.Model(inputs=[cnn_input, rnn_input], outputs=hidden)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit([train_images, train_images], train_labels, validation_split=0.1, epochs=10, batch_size=1)

test_loss, test_acc = model.evaluate([test_images, test_images], test_labels)
print()
print()
print()
print('test_acc:', test_acc)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Train/Validation Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation',], loc='upper right')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Train/Validation Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()