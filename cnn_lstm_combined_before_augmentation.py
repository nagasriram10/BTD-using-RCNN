# =============================================================================
# Prerequisites & Imports
# =============================================================================
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

#Enables image plotting
matplotlib.rcParams['interactive'] == True


# =============================================================================
# Create Data Method
# =============================================================================

def createData(yesDirectory, noDirectory, allDirectory):
    #Read in the filenames  of the images in the dataset
    yesFileNames = os.listdir(yesDirectory)
    noFileNames = os.listdir(noDirectory)
    
    #Create labels for each of these images (yes = 1, no = 0)
    yesFileLabels = np.ones(len(yesFileNames))
    noFileLabels = np.zeros(len(noFileNames))

    #Combine yes and no for filenames and labels convert to np arrays
    allFileNames = yesFileNames + noFileNames
    allFileNames = np.asarray(allFileNames)
    allLabels = np.concatenate((yesFileLabels, noFileLabels), axis=0)
        
    # =============================================================================
    # Get the images standardized
    # =============================================================================
    
    #Pasting images onto background image to standardize
    sizes = [Image.open(allDirectory + "/" + f, 'r').size for f in allFileNames]
    maxSize = max(sizes)
    centerSize = tuple([int(x/2) for x in maxSize])
    fourthSize = tuple([int(x/4) for x in maxSize])
    eigthSize = tuple([int(x/8) for x in maxSize])
    allImages = []
    
    for imgName in allFileNames:
        #Create a background that is 8-bit grayscale with pillow
        bgImage = Image.new("L", centerSize, color ="black")
        currImage = Image.open(allDirectory + "/" + imgName)
        imgOffset = tuple([int(x/2) for x in currImage.size])
        imgCoor= np.subtract(fourthSize, imgOffset)
        bgImage.paste(currImage, tuple(imgCoor))
        bgImage = bgImage.resize(eigthSize)
        #Image values standardized
        allImages.append(np.array(bgImage)/255)
    #plt.imshow(allImages[252])
    
    
    #Split images into test and train data with scikitlearn
    train_images, test_images, train_labels, test_labels = train_test_split(allImages, allLabels, test_size=0.33, random_state=42)
    train_images = np.asarray(train_images, dtype="float64")
    test_images = np.asarray(test_images, dtype="float64")

    #Return train and test images as well as the train and test labels
    return train_images, test_images, train_labels, test_labels


# =============================================================================
# Call createData method to create test and train images and labels.
# =============================================================================
train_images, test_images, train_labels, test_labels = createData("brain_tumor_dataset_before_augmentation\\bt_yes","brain_tumor_dataset_before_augmentation\\bt_no", "brain_tumor_dataset_before_augmentation\\bt_all")

# =============================================================================
# Make all labels and images float32 and OHE
# =============================================================================
train_labels = to_categorical(train_labels.astype("float32"))
test_labels = to_categorical(test_labels.astype("float32"))
test_images = test_images.reshape((84,135,240,1)).astype("float32")
train_images = train_images.reshape((169,135,240,1)).astype("float32")

# =============================================================================
# 
# train_labels = train_labels.astype("float32")
# test_labels = test_labels.astype("float32")
# test_images = test_images.reshape((84, 135, 240, 1)).astype("float32")
# train_images = train_images.reshape((169, 135, 240, 1)).astype("float32")
# =============================================================================

cnn_input = layers.Input(shape=(135, 240, 1))
rnn_input = layers.Input(shape=(135, 240))
# =============================================================================
# Create the Convolutional Neural Network - Having ResourceExhaust error and NAN for loss 
# =============================================================================
cnn = models.Sequential()
cnn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(135, 240,1)))
cnn.add(layers.MaxPooling2D((2, 2)))
#cnn.add(layers.Flatten()) #Forgetting this leads to a output dimension mismatch
cnn.add(layers.Dense(8, activation='relu'))
cnn.add(layers.Dense(2, activation='sigmoid')) #Must match the usage of OHE and the number of classes
cnn.add(layers.Flatten())
cnn_output = cnn(cnn_input)

cnn.summary()

lstm_rnn = models.Sequential()
lstm_rnn.add(keras.Input(shape=(135, 240)))
lstm_rnn.add(layers.LSTM(32, activation='relu'))
#lstm_rnn.add(layers.Flatten()) #Forgetting this leads to a output dimension mismatch
lstm_rnn.add(layers.Dense(8, activation='relu'))
lstm_rnn.add(layers.Dense(2, activation='sigmoid')) #Must match the usage of OHE and the number of classes
lstm_rnn.add(layers.Flatten())
rnn_output = lstm_rnn(rnn_input)

lstm_rnn.summary()

combined = Concatenate()([cnn_output, rnn_output])

# Add final dense layer for classification
hidden = layers.Dense(2, activation='sigmoid')(combined)

# Define the complete model with both inputs and output
model = models.Model(inputs=[cnn_input, rnn_input], outputs=hidden)

# Compile and train the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
ic(train_images.shape)
ic(train_labels.shape)
history = model.fit([train_images, train_images], train_labels, validation_split=0.1, epochs=1, batch_size=1)
ic(train_images.shape)
ic(train_labels.shape)

'''combined = Concatenate([cnn, lstm_rnn])
hidden = layers.Dense(1, activation='sigmoid')
# =============================================================================
# Compile and Train the ConvNet
# =============================================================================
model = models.Sequential()
model.add(combined)
model.add(hidden)
model.compile(optimizer = 'rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

#Working epoch/batch combos with acc are 5/20 (79.88% Train / 70.23% Test), 6/15(77.51% Train / 72.60% Test)
history = model.fit(train_images, train_labels, validation_split = 0.1, epochs=8, batch_size=1)''' #Wrong combination leads to OOM / ResourceExhaustedError

# =============================================================================
# Display accuracy results after running tests
# =============================================================================
#test_loss, test_acc = model.evaluate(test_images, test_labels)
#test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(1)
ic(test_images.shape)
ic(test_labels.shape)
test_loss, test_acc = model.evaluate([test_images, test_images], test_labels)
ic(test_images.shape)
ic(test_labels.shape)

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