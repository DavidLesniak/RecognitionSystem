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

## Pliki

- `requirements.txt` — zależności środowiska.
- `check_environment.py` — sprawdzenie interpretera i importu OpenCV.

## Stan projektu

Przygotowano środowisko i skrypt sprawdzający import OpenCV.
Obsługa obrazów i wideo, interfejs oraz integracja modelu pozostają do wykonania.
