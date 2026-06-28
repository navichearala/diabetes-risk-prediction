"""Pytest bootstrap: put src/ on the path so tests import modules cleanly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
