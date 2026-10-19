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

dir = 'GTSRB'

os.mkdir(dir)

kaggle.api.authenticate()
kaggle.api.dataset_download_files('meowmeowmeowmeowmeow/gtsrb-german-traffic-sign', path=dir, unzip=True)

print('Pobieranie zakończone')