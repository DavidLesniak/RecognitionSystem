import keras
import tensorflow as tf
import os



"""
Parametry nazw plików
"""
model_version = 'v3'
model_name = f'cnn_znaki_{model_version}'
model_suffix = '.keras'



"""
Ścieżki do folderów z danymi
"""
path_test  = os.path.join(os.getcwd(), '..', 'PTSD_znaki_zbalansowane', 'test')
path_model = os.path.join(os.getcwd(), 'model_cnn_gotowy', model_version, f'{model_name}{model_suffix}')



"""
Parametry wejściowe modelu
"""
image_size = (128, 128)



"""
Ewaluacja modelu
"""
test_dataset = tf.keras.utils.image_dataset_from_directory(
    path_test,
    image_size=image_size,
)

model = keras.models.load_model(path_model)

loss, accuracy = model.evaluate(test_dataset, verbose=1)

print()
print(f'Skuteczność na zbiorze testowym : {accuracy*100:.2f}%')
print(f'Strata na zbiorze testowym      : {loss}')