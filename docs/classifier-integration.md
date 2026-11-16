# Przygotowanie integracji klasyfikatora

Interfejs edytora przechowuje obrazy jako tablice BGR uint8 w pełnej rozdzielczości:
`viewer.image` to oryginał, a `viewer.result` to wynik ostatniej zakończonej edycji.
Do analizy należy przekazywać te tablice, a nie pomniejszone podglądy Tkinter.
Wynik edycji jest aktualny, gdy nie trwa przetwarzanie, nie oczekuje zadanie
odświeżenia i `viewer.settings() == viewer.applied_settings`.

## Model dostępny w repozytorium

`traffic classifier archive/v2/traffic_classifier_v2.keras` ma wejście
`(None, 64, 64, 3)` i wyjście z 92 prawdopodobieństwami softmax.
Kod `modul_CNN/cnn.py` opisuje już kolejny wariant 128×128; nie należy
przenosić tego rozmiaru do obsługi zapisanego modelu v2.

Adapter konwertuje BGR na RGB, skaluje przez Pillow z interpolacją bilinear
i dodaje wymiar partii. Wynik ma typ float32 i wartości 0–255.
To przygotowanie odpowiada `load_img` i `img_to_array` użytym przez
`modul_weryfikacji/pierwsza_predykcja.py`. Nie dodajemy dzielenia przez 255:
w skrypcie predykcji go nie ma, a zapisany model v2 nie zawiera Rescaling.
Przed użyciem kolejnej wersji modelu należy potwierdzić jej sposób przygotowania
danych; sam kształt wejścia nie określa normalizacji.

## Wywołanie z przyszłej integracji

```python
from classifier_adapter import ClassifierAdapter, DEFAULT_MODEL

classifier = ClassifierAdapter.from_files(DEFAULT_MODEL, "class_names.json")
comparison = classifier.compare(original_bgr, edited_bgr)
result = comparison["original"]
print(result.class_name, result.confidence, result.probabilities)
```

Adapter ładuje model tylko raz. Oba obrazy przechodzą ten sam sposób przygotowania.
`Prediction` zawiera indeks klasy, nazwę, pewność i pełen wektor prawdopodobieństw.
Wektor może być później przekazany do modułu weryfikacji.
Wywołania ładowania i predykcji należy wykonywać poza głównym wątkiem Tkinter.

`class_names.json` musi być listą nazw w dokładnej kolejności użytej w treningu
konkretnego modelu. Pliku nie ma jeszcze w repozytorium. Powinien zostać
wyeksportowany przez autora modelu z `train_dataset.class_names` przed
operacjami `cache()` i `prefetch()`. Nie odtwarzamy etykiet z CSV ani z numerów
znaków. Adapter sprawdza zgodność liczby klas z wyjściem modelu.

## Sprawdzenie przygotowania bez TensorFlow

```powershell
.\.venv\Scripts\python.exe classifier_adapter.py "ścieżka\do\zdjęcia.jpg"
```

Polecenie sprawdza konfigurację wejścia archiwum modelu i przygotowuje obraz.
Nie ładuje wag i nie wykonuje predykcji. Właściwa klasyfikacja wymaga TensorFlow
z `requirements.txt` oraz potwierdzonego pliku klas.

Testy sprawdzają konfigurację rzeczywistego archiwum v2, kanały RGB, rozmiar,
typ i zakres wejścia, zachowanie oryginału, etykiety i porównanie dwóch obrazów.
Przepływ predykcji jest testowany modelem zastępczym; nie jest to dowód działania
wag modelu ani jego skuteczności. Integracja przycisków i wyników w GUI pozostaje
do wykonania po dostarczeniu listy klas i uruchomieniu rzeczywistej predykcji.
