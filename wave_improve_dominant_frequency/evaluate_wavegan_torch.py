"""Evaluate the dominant-frequency distribution-alignment experiment."""

from pathlib import Path
import sys


BASE_EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / 'wavegan_improve'
sys.path.insert(0, str(BASE_EXPERIMENT_DIR))

from evaluate_wavegan_torch import main


if __name__ == '__main__':
  main()
