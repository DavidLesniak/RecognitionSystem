import keras
import tensorflow as tf

import os

train_model_version = '2'

image_size = (64, 64)
batch_size = 32

path_test = os.path.join(os.getcwd(), '..', 'PTSD', 'test')

test_dataset = tf.keras.utils.image_dataset_from_directory(
    path_test,
    image_size=image_size,
    batch_size=batch_size
)

model = keras.models.load_model(os.path.join(os.getcwd(), f'traffic_classifier_v{train_model_version}.keras'))

loss, accuracy = model.evaluate(test_dataset, verbose=1)

print()
print(f'Skuteczność na zbiorze testowym : {accuracy*100:.2f}%')
print(f'Strata na zbiorze testowym      : {loss*100:.2f}%')