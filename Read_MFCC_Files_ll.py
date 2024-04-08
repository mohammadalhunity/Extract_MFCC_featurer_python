import glob
from pathlib import Path
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.layers import InputLayer, Conv2D, Activation, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation,Flatten, Conv2D, InputLayer
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras import layers, models




data = []

labels = []

max_frames = 0




for mfcc_file in sorted(glob.glob('All_Datasets/*.npy')):
    mfcc_data = np.load(mfcc_file)
    data.append(mfcc_data)
    stemFilename = (Path(os.path.basename(mfcc_file)).stem)
    label = stemFilename.split('_')
    labels.append(label[0])
    max_frames = max(max_frames, mfcc_data.shape[0])



for i in range(len(data)):
    data[i] = np.pad(data[i], ((max_frames - data[i].shape[0], 0), (0, 0)))








labell = np.array(labels)
dataa = np.array(data)







label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(labels)
y_categorical = to_categorical(y_int)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(dataa, y_categorical, test_size=0.1, random_state=45)
# Reshape the data to fit the model input

X_train = np.array(X_train).reshape(-1, X_train[0].shape[0], X_train[0].shape[1], 1)
X_val = np.array(X_val).reshape(-1, X_val[0].shape[0], X_val[0].shape[1], 1)



# labell = np.array(labels)
# dataa = np.array(data)
# # 
# labels[0]

# LE=LabelEncoder()
# labelll=to_categorical(LE.fit_transform(labell))

# X_train, X_tmp, y_train, y_tmp = train_test_split(dataa, labelll, test_size=0.1, random_state=0)
# X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.6, random_state=0)

X_train.shape

def create_model():
    num_classes = 20
    # model = models.Sequential()
    # # model.add(InputLayer(input_shape = (917, 32, 1)))
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(917, 32, 1)))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(254, activation='relu'))
    # model.add(layers.Dense(num_classes, activation='softmax'))
    # return model

    model = Sequential()
    
    model.add(InputLayer((917, 16, 1)))

    # Convolutional layer with batch normalization and activation
    model.add(Conv2D(32, (3,3), padding='same'))  # Using smaller filters and 'same' padding
    model.add(BatchNormalization())  # Batch Normalization layer
    model.add(Activation('relu'))

    # Optional: Add more Convolutional layers here if needed

    # Flatten the convolution output to feed into the dense layer
    model.add(Flatten())

    # First dense layer with dropout
    model.add(Dense(254))  # Increased number of neurons
    model.add(BatchNormalization())
    
    
    # Batch Normalization layer
    model.add(Activation('relu'))
    model.add(Dropout(0.5))  # Dropout layer

    # Output layer with 'num_classes' units
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model
    
model = create_model()
model.compile(loss='categorical_crossentropy',    metrics=['accuracy'], optimizer=Adam(learning_rate=0.001))
model.summary()

# input_shape
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.2))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.2))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dropout(0.2))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(24, activation='softmax'))

# Creat The Packeg to Trining 
num_epochs = 100
num_batch_size = 32
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=num_batch_size, epochs=num_epochs, verbose=1)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

predicted_probs = model.predict(X_val, verbose=0)
predicted = np.argmax(predicted_probs, axis=1)
actual = np.argmax(y_val, axis=1)
accuracy = metrics.accuracy_score(actual, predicted)
print(f'Accuracy: {accuracy * 100}%')

model.save_weights('digit_classification_Finalllll.h5')

confusion_matrix = metrics.confusion_matrix(np.argmax(y_val,axis=1), predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix =
confusion_matrix)
cm_display.plot()

model.save_weights('digit_classification_Finallllll.h5')
 
# def Model():
# 
#     # Convolutional layers with Batch Normalization
#     model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(X_train[0].shape[0], X_train[0].shape[1], 1)))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
    
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
    
#     model.add(Conv2D(128, (3, 3), activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
    

# model.save(model_path)









