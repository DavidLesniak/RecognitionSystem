"""Sprawdzenie środowiska projektu: python check_environment.py."""

import sys

import cv2


if __name__ == "__main__":
    print(f"Python: {sys.version.split()[0]}")
    print(f"Interpreter: {sys.executable}")
    print(f"Wirtualne środowisko: {'TAK' if sys.prefix != sys.base_prefix else 'NIE'}")
    print(f"OpenCV: {cv2.__version__}")
    print("Import OpenCV: OK")
