import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import random

dir = 'GTSRB'

df_Train= pd.read_csv(os.path.join(dir, 'Train.csv'))
df_Test = pd.read_csv(os.path.join(dir, 'Test.csv'))

# ---------------------------------------------------------------------------------------------------------------------------

print("Liczba obrazów treningowych:", len(df_Train))
print("Liczba obrazów testowych:", len(df_Test))
print("Liczba klas znaków:", df_Train["ClassId"].nunique())

# ---------------------------------------------------------------------------------------------------------------------------

classCount = df_Train["ClassId"].value_counts().sort_index()

plt.figure(figsize=(12, 5))
plt.bar(classCount.index, classCount.values)
plt.title("Rozkład liczby obrazów na klasę")
plt.xlabel("Numer klasy")
plt.ylabel("Liczba obrazów")
plt.xticks(range(df_Train['ClassId'].nunique()))
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------------------------------------------------------

plt.figure(figsize=(12, 5))

for i in range(15):
    plt.subplot(3, 5, i + 1)

    sample = random.randint(0, len(df_Train["Path"]))

    path = df_Train.iloc[sample]["Path"]

    img = Image.open(os.path.join(dir, path))
    plt.imshow(img)
    plt.title("Klasa: " + str(df_Train.iloc[sample]["ClassId"]))
    plt.axis("off")

plt.suptitle("Przykładowe obrazy zbioru GTSRB")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df_Train["Width"], bins=20, edgecolor="black")
plt.title("Rozkład szerokości obrazów")
plt.xlabel("Szerokość (piksele)")
plt.ylabel("Liczba obrazów")

plt.subplot(1, 2, 2)
plt.hist(df_Train["Height"], bins=20, edgecolor="black")
plt.title("Rozkład wysokości obrazów")
plt.xlabel("Wysokość (piksele)")
plt.ylabel("Liczba obrazów")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(df_Train["Width"], df_Train["Height"], alpha=0.4)

plt.title("Rozkład szerokości i wysokości obrazów GTSRB")
plt.xlabel("Width (piksele)")
plt.ylabel("Height (piksele)")

plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------------------------------------------------------

imgRatio = df_Train["Width"] / df_Train["Height"]

plt.figure(figsize=(12, 5))

plt.hist(imgRatio, bins=20, edgecolor="black")

plt.title("Rozkład proporcji obrazów")
plt.xlabel("Proporcja szerokości do wysokości (Width / Height)")
plt.ylabel("Liczba obrazów")

plt.tight_layout()
plt.show()


