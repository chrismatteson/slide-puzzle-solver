import sys
from pathlib import Path

# The scripts import each other as top-level modules, as `python scripts/<name>.py` runs them.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
