# Wpływ filtrów na klasyfikację CNN v3

Sprawdzono zapisany model CNN v3 i reguły weryfikacji używane przez aplikację. Każdy filtr stosowano osobno. Wynik porównano z predykcją dla tego samego kadru bez korekty. Konfiguracja filtrów, wersje bibliotek i suma SHA-256 modelu są zapisane w plikach `parametry.json`.

## Wyniki

- [Pilotaż](pilotaz/podsumowanie.csv) obejmował 376 zdjęć testowych PTSD (po cztery dla każdej z 94 klas). Na podzbiorze 188 zdjęć zasymulowano również przyciemnienie, spadek kontrastu i szum. Wyniki próby przedstawiają wykresy [zmian klasy i decyzji weryfikatora](pilotaz/ranking-wplywu-filtrow.png), [trafności według warunków](pilotaz/wplyw-filtrow-trafnosc.png) i [błędnych predykcji z decyzją ACCEPT](pilotaz/wplyw-filtrow-weryfikacja.png).
- [Pełny test](pelny-test/podsumowanie.csv) obejmował 4298 zdjęć ze zbioru testowego PTSD bez symulowanych zmian. Oryginał i siedem wariantów filtrów dały razem 34 384 predykcje. Przynajmniej jeden filtr skorygował klasę na 38 zdjęciach. Każde z badanych ustawień popsuło więcej poprawnych rozpoznań, niż skorygowało błędnych. [Wybrane przypadki](pelny-test/wybrane-przypadki.csv) pokazują wyniki przed korektą i po niej.
- Szczegóły są w plikach [predykcji pilotażu](pilotaz/predykcje.csv.gz) i [predykcji pełnego testu](pelny-test/predykcje.csv.gz). Każdy wiersz zawiera identyfikator kadru, prawdziwą i przewidzianą klasę, ustawienie filtra, pewność, margines, entropię i decyzję weryfikatora.

## Powtórzenie testu

Uruchom polecenia z katalogu głównego repozytorium, w środowisku z zależnościami z `requirements.txt`. Skrypt korzysta z archiwum `PTSD_znaki.zip` oraz zapisanego CNN v3.

```sh
python docs/filter-evaluation/benchmark_filters.py . wyniki-filtrow/pilotaz
python docs/filter-evaluation/plot_filter_benchmark.py wyniki-filtrow/pilotaz
python docs/filter-evaluation/plot_filter_ranking.py wyniki-filtrow/pilotaz
python docs/filter-evaluation/benchmark_filters.py . wyniki-filtrow/pelny-test --full-test
python docs/filter-evaluation/find_filter_repairs.py . wyniki-filtrow/pelny-test
```

Tabele i wykresy opublikowane tutaj powstały z użyciem kodu projektu z commita `46c0f36a`. Pełny test i pilotaż są osobnymi zestawieniami i mają różne rozkłady klas.
