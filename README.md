# RecognitionSystem

RecognitionSystem to projekt inżynierski dotyczący klasyfikacji polskich znaków
drogowych i oceny wiarygodności predykcji. Aplikacja pokazuje, jak zmienia się
wynik klasyfikacji tego samego zdjęcia przed zastosowaniem filtrów i po nim.

## Uruchomienie na Windows

Wymagania: Python 3.12 i Git. Poniższe polecenia są przeznaczone dla PowerShell na Windows.

```powershell
git clone https://github.com/DavidLesniak/RecognitionSystem.git
cd RecognitionSystem
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Jeśli repozytorium i zależności są już na komputerze, przy kolejnym uruchomieniu
wystarczy wykonać ostatnie polecenie w folderze `RecognitionSystem`. Skrypt
`check_environment.py` pozwala sprawdzić używany interpreter i import OpenCV:

```powershell
.\.venv\Scripts\python.exe check_environment.py
```

## Edycja zdjęć

Przycisk „Wybierz zdjęcie” otwiera plik. Aplikacja pokazuje oryginał i wynik
obok siebie oraz nazwę i oryginalne wymiary obrazu.

Ustaw jasność, kontrast, nasycenie, gamma, wyostrzanie, odszumianie lub
lokalną poprawę kontrastu (CLAHE). Podgląd odświeża się automatycznie po krótkiej
przerwie w zmianie ustawień. Gamma poniżej 1 rozjaśnia zdjęcie.
„Przywróć oryginał” resetuje filtry. Każde przeliczenie zaczyna się od oryginału.

„Zapisz wynik” zapisuje obraz PNG w pełnej rozdzielczości i plik JSON o tej samej
nazwie. JSON zawiera ustawienia filtrów, identyfikator źródła i wersje bibliotek,
co ułatwia odtworzenie edycji. Zapis jest dostępny po przeliczeniu podglądu.

## Klasyfikacja i weryfikacja

Wczytaj zdjęcie znaku, ustaw filtry i kliknij „Porównaj klasyfikację”.
Pod każdym podglądem zobaczysz kod znaku, pewność modelu CNN v3 i ocenę
wiarygodności: `ACCEPT`, `WARN` albo `REJECT`. Po zmianie zdjęcia lub filtrów
kliknij przycisk ponownie, aby obliczyć nowe wyniki.

Aplikacja używa zapisanego modelu CNN v3 `znaki` z 94 klasami. Przed predykcją
adapter przygotowuje dla modelu obraz RGB o rozmiarze 128 × 128 pikseli. Oryginał
i zdjęcie po edycji przechodzą przez ten sam model i te same reguły weryfikacji.

Weryfikator modelu v3 `znaki` bierze pod uwagę pewność modelu, różnicę między
dwoma najwyższymi prawdopodobieństwami oraz entropię, która rośnie, gdy model
rozkłada prawdopodobieństwo na więcej klas.

| Ocena | Kiedy się pojawia |
| --- | --- |
| `ACCEPT` | Pewność ≥ 97%, różnica ≥ 95 punktów procentowych **i** entropia ≤ 0,12. Wszystkie trzy warunki muszą zostać spełnione. |
| `WARN` | Pozostałe wyniki, które nie spełniają warunków `ACCEPT` ani `REJECT`. |
| `REJECT` | Pewność < 90%, różnica < 70 punktów procentowych **lub** entropia > 0,50. Wystarczy jeden z tych warunków. |

Ocena opisuje wiarygodność predykcji według [reguł v3](modul_weryfikacji/v3/reguly_weryfikacji_v3.py).

## Przykład do sprawdzenia

1. Rozpakuj `PTSD_znaki.zip` i otwórz w aplikacji zdjęcie
   `PTSD_znaki/test/A-1/2019_0719_142601_003 0157_0.jpg`.
2. Ustaw jasność na `+15` i po odświeżeniu podglądu kliknij „Porównaj klasyfikację”.
3. W przeprowadzonej próbie obie wersje zdjęcia dały wynik `A-1` i ocenę `ACCEPT`.

Przykład z oceną `REJECT` i pełniejsze wyniki prób są w [opisie integracji](docs/classifier-integration.md#sprawdzenie-aplikacji-na-zapisanym-modelu).

## Pliki

- `app.py` — interfejs wyboru, porównania i edycji zdjęć.
- `image_processing.py` — przekształcenia i zapis obrazu z parametrami.
- `classifier_adapter.py` — przygotowanie obrazów i wspólny interfejs klasyfikatora.
- `modul_CNN/` — modele, trening i ocena klasyfikacji.
- `modul_weryfikacji/` — reguły i wyniki oceny wiarygodności predykcji.
- [Opis integracji i przykłady do sprawdzenia](docs/classifier-integration.md).

## Testy

Testy edytora i adaptera obejmują przekształcenia obrazu, zapis PNG i JSON,
odtwarzanie ustawień oraz przygotowanie danych wejściowych klasyfikatora.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Reguły weryfikacji CNN v3 można sprawdzić osobno:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s modul_weryfikacji/v3 -p test_reguly_weryfikacji_v3.py -v
```
