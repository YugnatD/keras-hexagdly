import sys
from pathlib import Path

# Allow running the test suite without an editable install (`pip install -e .`).
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "tests"))
