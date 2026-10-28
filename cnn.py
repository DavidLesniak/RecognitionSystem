from pprint import pprint
import os
from tqdm import tqdm
from PIL import Image
import numpy as np

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
import matplotlib.pyplot as plt

train_model_version = '2'

"""
Ustawienia parametrow
"""
image_size = (64, 64)
batch_size = 32 # Linux + GPU 128
seed = 13
epochs = 35

path_train = os.path.join('PTSD', 'train')
path_test = os.path.join('PTSD', 'test')



"""
Wykrywanie dostępnych kart graficznych w celu przyśpieszenia szkolenia modelu
"""
print("Wykryte karty graficzne:", tf.config.list_physical_devices('GPU'))



"""
Wczytanie danych treningowych, rozdzielenie na 3 ziobry typu tf.data.Dataset:
    * treningowy - do trenowania modelu
    * validacyjny - do sprawdzania modelu podczas trenowania
    * testowy - do sprawdzenia gotowego modelu
"""
train_dataset = tf.keras.utils.image_dataset_from_directory(
    path_train,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    path_train,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size
)



"""
Optymalizacja wczytywanych danych do pamięci
"""
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)



"""
Stworzenie modelu CNN
"""
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

model = Sequential()

model.add(Input(shape=(64, 64, 3)))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(92, activation='softmax'))

model.compile(
    optimizer='adam',                
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()



"""
Trenowanie modelu
"""
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs)



"""
Generownaie wykresu podsumowania
"""
plt.figure(figsize=(12,5))

# Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.figure(figsize=(12,5))
plt.savefig(f'model_accuracy_v{train_model_version}.png')

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig(f'model_loss_v{train_model_version}.png')



"""
Zapisanie modelu
"""
model.save(f'traffic_classifier_v{train_model_version}.keras')