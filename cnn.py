from pprint import pprint
import os
from tqdm import tqdm
from PIL import Image
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
import matplotlib.pyplot as plt



path = os.path.join('PTSD', 'train')

# Wczytanie zdjec
images = []
labels = []

category_dirs = os.listdir(path)
print(f'Klasy: {len(category_dirs)}')

for category in tqdm(category_dirs):
    files = os.listdir(os.path.join(path, category))

    for img in files:
        image = Image.open(os.path.join(path, category, img))
        image = image.resize((30, 30))
        image = np.array(image)
        images.append(image)
        labels.append(category)

images = np.array(images)
labels = np.array(labels)

print(len(images))
print(len(labels))

# Zamiana nazw etykiet na liczby
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# Przygotowanie podziału danych do szkolenia
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=13)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Zamiana liczb na wektory one-hot
num_classes = len(encoder.classes_)

# przygotowanie wyjscia
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


# Stworzenie modelu
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(30,30,3)))
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
    loss='categorical_crossentropy', 
    metrics=['accuracy']            
)

model.summary()


# Trenowanie
epochs = 35
history1 = model.fit(X_train, y_train, batch_size=128, epochs=epochs, validation_data=(X_test, y_test))


# Podsumowanie
plt.figure(figsize=(12,5))

# Accuracy
plt.plot(history1.history['accuracy'], label='Train Accuracy')
plt.plot(history1.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.figure(figsize=(12,5))

# Loss
plt.plot(history1.history['loss'], label='Train Loss')
plt.plot(history1.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save('traffic_classifier.keras')