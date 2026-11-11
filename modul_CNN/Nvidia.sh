#!/bin/bash

if [ "$1" == "start" ]; then
    echo ">>> Przełączanie RTX 5060 w tryb TRENINGU AI (2400-2950 MHz, 145W)..."
    sudo nvidia-smi -pm 1
    sudo nvidia-smi -pl 145
    sudo nvidia-smi -lgc 2400,2950
    echo ">>> Gotowe! Karta skonfigurowana pod maksymalną wydajność TensorFlow."
    nvidia-smi --query-gpu=name,persistence_mode,power.limit,clocks.gr,temperature.gpu --format=csv

elif [ "$1" == "stop" ]; then
    echo ">>> Przywracanie domyślnych ustawień RTX 5060..."
    sudo nvidia-smi -rgc
    sudo nvidia-smi -pm 0
    echo ">>> Gotowe! Karta wróciła do cichego trybu pulpitu."
    nvidia-smi --query-gpu=name,persistence_mode,power.limit,clocks.gr,temperature.gpu --format=csv

else
    echo "Użycie: ~/gpu_tryb.sh [start|stop]"
    echo "  start - blokuje zegary 2400-2950 MHz i limit mocy 145 W pod trening"
    echo "  stop  - przywraca automatyczne zarządzanie energią"
fi
