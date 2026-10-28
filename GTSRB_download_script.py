import os
import kaggle

"""
Skrypt do automatycznego pobierania zbioru danych GTSRB - German Traffic Sign Recognition Benchmark z platformy kaggle.
Do prawidłowego działania należy wygenerować token API. 
W tym celu:
   1. Zaloguj się do serwisu kaggle
   2. Przejdź do zakładki setting twojego konta
   3. Przejdź do zakładki API Tokens
   4. Kliknij Generate New Token
   5. Wpisz nazwe tokena
   6. Kliknij Generate
   7. Skopiuj wartość API Token
   8. Przejdź do folderu użytkownika windows np. C: -> Users -> student
   9. Utwórz strukturę .kaggle\access_token
  10. Wklej skopiowany API Token do pliku access_token
"""


"""
   Pobranie zbioru GTSRB z platformy kaggle
"""

dir = 'PTSD'

if not os.path.exists(dir):
   print('Utworzenie folderu GTSRB')
   os.mkdir(dir)

print('Pobieranie zbioru GTSRB')

kaggle.api.authenticate()
kaggle.api.dataset_download_files('kasia12345/polish-traffic-signs-dataset', path=dir, unzip=True)
   
print('Pobieranie zakonczone')