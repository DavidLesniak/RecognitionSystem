# Integracja klasyfikatora

## Model używany w aplikacji

Aplikacja ładuje `modul_CNN/model_cnn_gotowy/v3/znaki/best_cnn_znaki_v3.keras`
i plik `class_names.json` z tego samego katalogu. Kolejność 94 klas pochodzi
ze skryptu `modul_CNN/predict_image.py`. Przy wczytaniu adapter sprawdza zgodność
liczby klas z liczbą wyjść zapisanego modelu.

Przygotowanie wejścia odpowiada demonstracji weryfikacji v3. Adapter konwertuje
BGR na RGB i skaluje obraz metodą bilinear w TensorFlow do 128 × 128.
Warstwa `Rescaling` w modelu przelicza wartości pikseli z zakresu 0–255.
Każda predykcja jest oceniana przez `evaluate_prediction()` z
`modul_weryfikacji/v3/ocena_predykcji_v3.py`. Ocena korzysta z najwyższego
prawdopodobieństwa, różnicy między dwoma najwyższymi wynikami i entropii.

## Sprawdzenie aplikacji na zapisanym modelu

1. Uruchom aplikację zgodnie z README i rozpakuj `PTSD_znaki.zip`.
2. Otwórz jeden z obrazów poniżej. Kliknij „Klasyfikuj zdjęcie” i odczytaj wynik
   w panelu „Decyzja modelu”.
3. Ustaw jasność na `+15`. Aplikacja odświeży podgląd i wyczyści poprzedni wynik.
4. Po odświeżeniu kliknij „Klasyfikuj zdjęcie” i odczytaj wynik dla aktualnego obrazu.

Przykładowe wyniki z aplikacji w Windows i TensorFlow 2.21.0:

| Ścieżka w rozpakowanym archiwum | Oryginał | Jasność +15 |
| --- | --- | --- |
| `PTSD_znaki/test/A-1/2019_0719_142601_003 0157_0.jpg` | A-1 · 100,0% · ACCEPT | A-1 · 100,0% · ACCEPT |
| `PTSD_znaki/test/C-9/2019_0809_155001_003 1218_0.jpg` | C-10 · 28,5% · REJECT | D-29 · 50,3% · REJECT |

Wyniki są zaokrąglone tak jak w aplikacji. Przykład C-9 pokazuje predykcję
niezgodną z etykietą zbioru, którą reguły oznaczyły jako mało wiarygodną.

## Przekazywanie obrazów z edytora

Interfejs edytora przechowuje obrazy jako tablice BGR uint8 w pełnej rozdzielczości:
`viewer.image` to oryginał, a `viewer.result` to wynik ostatniej zakończonej edycji.
Do analizy należy przekazywać te tablice, a nie pomniejszone podglądy Tkinter.
Wynik edycji jest aktualny, gdy nie trwa przetwarzanie, nie oczekuje zadanie
odświeżenia i `viewer.settings() == viewer.applied_settings`.

## Adapter dla modelu v2

`modul_CNN/model_cnn_gotowy/v2/traffic_classifier_v2.keras` ma wejście
`(None, 64, 64, 3)` i wyjście z 92 prawdopodobieństwami softmax.
Kod `modul_CNN/cnn.py` opisuje już kolejny wariant 128×128; nie należy
przenosić tego rozmiaru do obsługi zapisanego modelu v2.

Adapter konwertuje BGR na RGB, skaluje przez Pillow z interpolacją bilinear
i dodaje wymiar partii. Wynik ma typ float32 i wartości 0–255.
To przygotowanie odpowiada `load_img` i `img_to_array` użytym przez
`modul_weryfikacji/v2/pierwsza_predykcja.py`. Nie dodajemy dzielenia przez 255:
w skrypcie predykcji go nie ma, a zapisany model v2 nie zawiera Rescaling.
Przed użyciem kolejnej wersji modelu należy potwierdzić jej sposób przygotowania
danych; sam kształt wejścia nie określa normalizacji.

## Wywołanie adaptera v2 poza aplikacją

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

`class_names.json` dla v2 musi zawierać nazwy w kolejności użytej w jego treningu.
Plik dołączony do modelu v3 dotyczy v3 `znaki`. Plik dla v2 powinien zostać
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
wag v2 ani jego skuteczności. Sprawdzenie aplikacji z rzeczywistym modelem v3
opisano powyżej.
