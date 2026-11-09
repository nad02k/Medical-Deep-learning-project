"""Point d'entrée pour lancer l'interface utilisateur DataOps.

Usage:
    python app.py
ou:
    make ui
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.app import main

if __name__ == "__main__":
    main()
