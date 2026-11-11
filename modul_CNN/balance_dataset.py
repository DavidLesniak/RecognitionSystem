import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array



path_dataset = os.path.join(os.getcwd(), '..', 'PTSD_znaki_zbalansowane_v2', 'train')
target_count = 500

datagen = ImageDataGenerator(
    rotation_range=12,
    width_shift_range=0.08,
    height_shift_range=0.08,
    shear_range=0.08,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

for class_name in os.listdir(path_dataset):
    class_path = os.path.join(path_dataset, class_name)

    existing_files = [os.path.join(class_path, f) for f in os.listdir(class_path)]
    current_count = len(existing_files)
    
    if current_count >= target_count or current_count == 0:
        print(f'Klasa: {class_name:5} | Zdjęć: {current_count:4} | Dodano: 0')
        continue
        
    needed = target_count - current_count
    generated = 0
    file_idx = 0
    
    while generated < needed:
        img_path = existing_files[file_idx % current_count]
        file_idx += 1
        
        img = load_img(img_path)
        x = img_to_array(img).reshape((1,) + img_to_array(img).shape)
        
        for _ in datagen.flow(x, batch_size=1, save_to_dir=class_path, save_prefix='aug', save_format='jpg'):
            generated += 1
            break

    print(f'Klasa: {class_name:5} | Zdjęć: {current_count:4} | Dodano: {needed}')

print('Zakończono balansowanie zbioru!')