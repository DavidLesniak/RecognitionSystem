import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import random

path_train = os.path.join('PTSD', 'train')

classes = os.listdir(path_train)
classes_count = []

print("Liczba klas znakow:", len(classes))

for c in classes:
	folder = os.listdir(os.path.join('PTSD', 'train', c))
	classes_count.append(len(folder))
	print(c, len(folder))


plt.figure(figsize=(12, 5))
plt.bar(classes, classes_count)
plt.title("Rozkład liczby obrazów na klasę")
plt.xlabel("Nazwa klasy")
plt.ylabel("Liczba obrazów")
plt.xticks(classes, rotation=90, ha='right')
plt.tight_layout()
plt.show()