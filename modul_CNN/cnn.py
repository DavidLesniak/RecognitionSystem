import os
import numpy as np
import json
import pandas as pd

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt


"""
Optymalizacja obliczeń GPU
    * Aktywacja rdzeni Tensor (FP16)
"""
mixed_precision.set_global_policy('mixed_float16')



"""
Parametry używane do nazw plików
"""
model_version = 'v3'
model_name = f'cnn_znaki_{model_version}'
model_suffix = '.keras'



"""
Ścieżki do folderów z danymi
"""
path_train = os.path.join(os.getcwd(), '..', 'PTSD_znaki_zabalansowane', 'train')
path_test  = os.path.join(os.getcwd(), '..', 'PTSD_znaki_zabalansowane', 'test' )



"""
Parametry wejściowe modelu
"""
image_size = (128, 128)
batch_size = 256
seed = 13
epochs = 100

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=12,
    min_delta=0.001,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    filepath=f'best_{model_name}{model_suffix}',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)



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

test_dataset = tf.keras.utils.image_dataset_from_directory(
    path_test,
    image_size=image_size,
    batch_size=batch_size
)

class_count_train = len(train_dataset.class_names)
class_count_test  = len(test_dataset.class_names)

if class_count_train != class_count_test:
    print('UWAGA niektórych zbior treningowy i testowy roznia sie iloscia klas')
    print('Trening:', class_count_train)
    print('Testowy:', class_count_test)

assert train_dataset.class_names == test_dataset.class_names



"""
Optymalizacja wczytywanych danych do pamięci
"""
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)



"""
Stworzenie modelu CNN
"""
from tensorflow.keras.layers import Input, Rescaling, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

model = Sequential()

model.add(Input(shape=(image_size[0], image_size[1], 3)))
model.add(Rescaling(1./255))

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

model.add(Dense(class_count_train, activation='softmax', dtype='float32')) # dtype='float32' jeśli ustawiono mixed_float16

model.compile(
    optimizer='adam',                
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    jit_compile=True
)

model.summary()



"""
Trenowanie modelu
"""
history = model.fit(
    train_dataset, 
    validation_data=validation_dataset, 
    epochs=epochs,
    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)



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
plt.savefig(f'model_accuracy_v{model_name}.png')
plt.show()
plt.figure(figsize=(12,5))

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'model_loss_v{model_name}.png')
plt.show()



"""
Testowanie modelu
"""
loss, accuracy = model.evaluate(test_dataset, verbose=1)
print(f'\nSkuteczność na zbiorze testowym: {accuracy*100:.2f}%')
print(f'Strata na zbiorze testowym: {loss*100:.2f}')



"""
Zapisanie modelu
"""
model.save(f'{model_name}{model_suffix}')



"""
Zapis do pliku
"""
summary_to_file = {
    'image_size': image_size,
    'batch_size': batch_size,
    'seed': seed,
    'epochs': epochs,
    'model': model_name,
    'test_accuracy': f'{accuracy*100:.2f}%',
    'test_loss': f'{loss:.4f}'
}

with open(f'summary_{model_name}{model_suffix}.json', 'w') as f:
    json.dump(summary_to_file, f, indent=4, ensure_ascii=False)

history_df = pd.DataFrame(history.history)
history_df.to_csv(f'train_history_{model_name}{model_suffix}.csv', index_label='epoch')