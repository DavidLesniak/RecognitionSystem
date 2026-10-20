import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
	Zbiór treningowy
"""

# Wczytanie metadanych zbioru treningowego
df_Train = pd.read_csv(os.path.join('GTSRB', 'Train.csv'))
print(df_Train.shape)
print(df_Train)

# Obliczenie ilosci klas znaków w zbiorze
classCount = df_Train['ClassId'].value_counts().sort_index()
#print(classCount)

plt.subplot(2, 1, 1)
ax = classCount.plot(width=0.9, rot=0, kind='bar', figsize=(12, 8))
ax.bar_label(ax.containers[0], padding=3)
plt.title('Rozkład klas poszczegołnych klass znaków w zbiorze Treningowym', fontsize=14)
plt.xlabel('ID klasy', fontsize=12)
plt.ylabel('Liczba wystąpień', fontsize=12)


"""
	Zbiór testowy
"""
# Wczytanie metadanych zbioru treningowego
df_Test = pd.read_csv(os.path.join('GTSRB', 'Test.csv'))
print(df_Test.shape)
print(df_Test)

# Obliczenie ilosci klas znaków w zbiorze
classCount = df_Test['ClassId'].value_counts().sort_index()
#print(classCount)

plt.subplot(2, 1, 2)
ax = classCount.plot(width=0.9, rot=0, kind='bar', color='green', figsize=(12, 8))
ax.bar_label(ax.containers[0], padding=3)
plt.title('Rozkład klas poszczegołnych klass znaków w zbiorze Testowym', fontsize=14)
plt.xlabel('ID klasy', fontsize=12)
plt.ylabel('Liczba wystąpień', fontsize=12)

plt.tight_layout()
plt.show()

