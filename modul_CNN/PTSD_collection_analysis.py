import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import random
from tqdm import tqdm

path_train = os.path.join(os.getcwd(), '..', 'PTSD', 'train')

classes = os.listdir(path_train)
classes_count = []
classes_width = []
classes_height = []

print("Liczba klas znakow:", len(classes))

for c in classes:
	folder = os.listdir(os.path.join(path_train, c))

	w, h = 0, 0

	for f in folder:
		img = Image.open(os.path.join(path_train, c, f))
		w, h = img.size

		classes_width.append(w)
		classes_height.append(h)
	
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

plt.figure(figsize=(8, 6))

plt.scatter(classes_width, classes_height, alpha=0.4)

plt.title("Rozkład szerokości i wysokości obrazów GTSRB")
plt.xlabel("Width (piksele)")
plt.ylabel("Height (piksele)")

plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(classes_width, bins=20, edgecolor="black")
plt.title("Rozkład szerokości obrazów")
plt.xlabel("Szerokość (piksele)")
plt.ylabel("Liczba obrazów")

plt.subplot(1, 2, 2)
plt.hist(classes_height, bins=20, edgecolor="black")
plt.title("Rozkład wysokości obrazów")
plt.xlabel("Wysokość (piksele)")
plt.ylabel("Liczba obrazów")

plt.tight_layout()
plt.show()


imgRatio = np.array(classes_width) / np.array(classes_height)

plt.figure(figsize=(12, 5))

plt.hist(imgRatio, bins=20, edgecolor="black")

plt.title("Rozkład proporcji obrazów")
plt.xlabel("Proporcja szerokości do wysokości (Width / Height)")
plt.ylabel("Liczba obrazów")

plt.tight_layout()
plt.show()
