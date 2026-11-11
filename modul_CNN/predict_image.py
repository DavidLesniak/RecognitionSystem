### TESTOWA WERSJA !!!!!!!!!!!!!!!!!!!!


import numpy as np
import tensorflow as tf
import os

image_size = (64, 64)


model = tf.keras.models.load_model('/run/media/ufo/Windows/Users/ufo/Desktop/Projekt inżynierski/Repozytorium/RecognitionSystem/modul_CNN/best_traffic_classifier_v5_series.keras')

img_path = os.path.join('PTSD', 'new', 'new3.jpg')
img = tf.keras.utils.load_img(img_path, target_size=(64, 64))

# 2. Konwersja obrazu na macierz NumPy
img_array = tf.keras.utils.img_to_array(img)

# 3. Dodanie wymiaru paczki (Batch Dimension)
# Zamienia kształt z (128, 128, 3) na (1, 128, 128, 3)
img_array = tf.expand_dims(img_array, 0) 

# 4. GŁÓWNA FUNKCJA: Przewidywanie (funkcja zwraca prawdopodobieństwa dla wszystkich klas)
predictions = model.predict(img_array)

# 5. Wyciągnięcie klasy o najwyższym prawdopodobieństwie
print(predictions)
predicted_class_index = np.argmax(predictions[0])

# 6. Odczytanie tekstowej nazwy klasy
# Wykorzystujemy listę nazw klas z Twojego wcześniejszego zbioru train_ds
class_names = sorted(os.listdir(os.path.join('PTSD', 'train')))
predicted_class_name = class_names[predicted_class_index]
confidence = 100 * np.max(predictions[0])

print(f"Rozpoznany znak: {predicted_class_name} ({confidence:.2f}% pewności)")
