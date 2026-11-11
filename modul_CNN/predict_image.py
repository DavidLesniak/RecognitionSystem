import numpy as np
import tensorflow as tf
import os


image_size = (128, 128)
img_path = os.path.join('new.jpg')

model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'model_cnn_gotowy', 'v3', 'znaki', 'best_cnn_znaki_v3.keras'))


img = tf.keras.utils.load_img(img_path, target_size=image_size)

# 2. Konwersja obrazu na macierz NumPy
img_array = tf.keras.utils.img_to_array(img)

# 3. Dodanie wymiaru paczki (Batch Dimension) zamienia kształt z (128, 128, 3) na (1, 128, 128, 3)
img_array = tf.expand_dims(img_array, 0) 

# 4. GŁÓWNA FUNKCJA: Przewidywanie (funkcja zwraca prawdopodobieństwa dla wszystkich klas)
predictions = model.predict(img_array)

# 5. Wyciągnięcie klasy o najwyższym prawdopodobieństwie
print(predictions)
predicted_class_index = np.argmax(predictions[0])

# 6. Odczytanie tekstowej nazwy klasy
class_names = ['A-1', 'A-11', 'A-11a', 'A-12a', 'A-12b', 'A-12c', 'A-14', 'A-15', 'A-16', 'A-17', 'A-18b', 'A-2', 'A-20', 'A-21', 'A-24', 'A-29', 'A-3', 'A-30', 'A-32', 'A-4', 'A-6a', 'A-6b', 'A-6c', 'A-6d', 'A-6e', 'A-7', 'A-8', 'B-1', 'B-18', 'B-2', 'B-20', 'B-21', 'B-22', 'B-25', 'B-26', 'B-27', 'B-33', 'B-34', 'B-36', 'B-41', 'B-42', 'B-43', 'B-44', 'B-5', 'B-6-B-8-B-9', 'B-8', 'B-9', 'C-10', 'C-12', 'C-13', 'C-13-C-16', 'C-13a', 'C-13a-C-16a', 'C-16', 'C-2', 'C-4', 'C-5', 'C-6', 'C-7', 'C-9', 'D-1', 'D-14', 'D-15', 'D-18', 'D-18b', 'D-2', 'D-21', 'D-23', 'D-23a', 'D-24', 'D-26', 'D-26b', 'D-26c', 'D-27', 'D-28', 'D-29', 'D-3', 'D-34b', 'D-40', 'D-41', 'D-42', 'D-43', 'D-4a', 'D-4b', 'D-51', 'D-52', 'D-53', 'D-6', 'D-6b', 'D-7', 'D-8', 'D-9', 'G-1a', 'G-3']
predicted_class_name = class_names[predicted_class_index]
confidence = 100 * np.max(predictions[0])

print(f"Rozpoznany znak: {predicted_class_name} ({confidence:.2f}% pewności)")
