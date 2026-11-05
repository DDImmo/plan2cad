import os
import sys

if __package__ in (None, ""):
    # Lancé comme « python plan2cad ... » (dossier exécuté directement) :
    # on ajoute le dossier parent au chemin pour pouvoir importer le paquet.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from plan2cad.cli import main
else:
    from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
