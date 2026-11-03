# RecognitionSystem

Projekt inżynierski dotyczący weryfikacji decyzji modelu w klasyfikacji znaków drogowych.

## Uruchomienie środowiska

Wymagania: Python 3.12 i Git. Poniższe polecenia są przeznaczone dla PowerShell na Windows.

```powershell
git clone https://github.com/DavidLesniak/RecognitionSystem.git
cd RecognitionSystem
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_environment.py
```

## Edycja zdjęć

Uruchom aplikację w folderze repozytorium:

```powershell
.\.venv\Scripts\python.exe app.py
```

Przycisk „Wybierz zdjęcie” otwiera plik. Aplikacja pokazuje oryginał i wynik
obok siebie oraz nazwę i oryginalne wymiary obrazu.

Ustaw jasność, kontrast, nasycenie, gamma, wyostrzanie, odszumianie lub
lokalną poprawę kontrastu (CLAHE). Podgląd odświeża się automatycznie po krótkiej
przerwie w zmianie ustawień (180 ms). Suwaki pozostają dostępne podczas obliczeń;
przy wolniejszych filtrach aplikacja pomija nieaktualne wyniki.
Gamma poniżej 1 rozjaśnia obraz. „Przywróć oryginał” zeruje zmiany.
Każde przeliczenie zaczyna się od oryginału; tylko podglądy są pomniejszane.

„Zapisz wynik” zapisuje pełnowymiarowy obraz PNG i plik JSON o tej samej
nazwie z parametrami, identyfikatorem źródła i wersjami bibliotek.
Zapis jest dostępny po zakończeniu przeliczania aktualnych ustawień. Plik źródłowy jest chroniony
przed nadpisaniem. Błędny plik powoduje komunikat i pozostawia poprzedni obraz.

Kolejność: odszumianie → jasność i kontrast → gamma → CLAHE → nasycenie → wyostrzanie.
Edycja przygotowuje dane do późniejszych eksperymentów. Poprawa wyglądu zdjęcia
nie oznacza potwierdzonej poprawy klasyfikacji. Model i Grad-CAM nie są jeszcze
połączone z interfejsem.

## Pliki

- `app.py` — interfejs wyboru, porównania i edycji zdjęć.
- `image_processing.py` — przekształcenia i zapis obrazu z parametrami.
- `requirements.txt` — zależności środowiska.
- `check_environment.py` — sprawdzenie interpretera i importu OpenCV.

## Stan projektu

Przygotowano środowisko, skrypt sprawdzający import OpenCV i edytor zdjęć.
Odczyt wideo oraz integracja modeli klasyfikacji i weryfikacji pozostają do wykonania.

## Sprawdzenie edytora

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Sprawdzenia obejmują przekształcenia, odtwarzanie wyniku z parametrów,
zapis PNG/JSON, reset, obsługę błędów i przetwarzanie w tle.
