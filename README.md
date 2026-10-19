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

## Podgląd zdjęć

Uruchom aplikację w folderze repozytorium:

```powershell
.\.venv\Scripts\python.exe app.py
```

Przycisk „Wybierz zdjęcie” otwiera wybór pliku. Aplikacja odczytuje obraz
przez OpenCV, wyświetla jego nazwę, oryginalne wymiary i podgląd.
Duże obrazy są pomniejszane na potrzeby podglądu z zachowaniem proporcji.
Nieprawidłowy lub uszkodzony plik powoduje wyświetlenie komunikatu błędu.

## Pliki

- `app.py` — interfejs wyboru i podglądu zdjęć.
- `requirements.txt` — zależności środowiska.
- `check_environment.py` — sprawdzenie interpretera i importu OpenCV.

## Stan projektu

Przygotowano środowisko, skrypt sprawdzający import OpenCV i prototyp obsługi zdjęć.
Odczyt wideo oraz integracja modeli klasyfikacji i weryfikacji pozostają do wykonania.
