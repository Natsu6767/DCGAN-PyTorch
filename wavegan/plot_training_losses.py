"""Plot discriminator (D) and generator (G) losses from WaveGAN/SpecGAN logs."""

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
LOG_PATTERN = re.compile(
    r'Epoch\s+(?P<epoch>\d+)/(?P<epochs>\d+)\s+\|\s+'
    r'batch\s+(?P<batch>\d+)/(?P<batches>\d+)\s+\|\s+'
    r'D=(?P<discriminator>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+\|\s+'
    r'G=(?P<generator>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')


def parse_loss_log(log_path):
  """Return training-progress coordinates and D/G losses from one log file."""
  records = []
  with log_path.open(encoding='utf-8', errors='replace') as log_file:
    for line in log_file:
      match = LOG_PATTERN.search(line)
      if match is None:
        continue
      values = match.groupdict()
      epoch = int(values['epoch'])
      progress = (epoch - 1) + int(values['batch']) / int(values['batches'])
      records.append((progress, float(values['discriminator']), float(values['generator'])))
  if not records:
    raise ValueError('No D/G records found in {}.'.format(log_path))
  return np.asarray(records, dtype=np.float64)


def draw_losses(axis, records, title):
  """Draw raw logged D/G values for one model on one axis."""
  axis.plot(records[:, 0], records[:, 1], color='#1769aa', linewidth=0.8, label='D loss')
  axis.plot(records[:, 0], records[:, 2], color='#e76f51', linewidth=0.8, label='G loss')
  axis.axhline(0.0, color='#555555', linewidth=0.7, alpha=0.5)
  axis.set_title(title)
  axis.set_ylabel('WGAN-GP loss')
  axis.grid(alpha=0.25)
  axis.legend(loc='best')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--wavegan-log', type=Path, default=PROJECT_DIR / 'train_click_torch_gpu.log')
  parser.add_argument('--specgan-log', type=Path, default=PROJECT_DIR / 'train_click_specgan_torch_gpu.log')
  parser.add_argument('--output', type=Path, default=PROJECT_DIR / 'training_loss_curves.png')
  args = parser.parse_args()

  wavegan_records = parse_loss_log(args.wavegan_log)
  specgan_records = parse_loss_log(args.specgan_log)

  figure, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
  draw_losses(axes[0], wavegan_records, 'WaveGAN: discriminator and generator losses')
  draw_losses(axes[1], specgan_records, 'SpecGAN: discriminator and generator losses')
  axes[1].set_xlabel('Training progress (epoch)')
  figure.suptitle('Training loss curves', fontsize=16)

  args.output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(str(args.output), dpi=180)
  plt.close(figure)
  print('Parsed {} WaveGAN and {} SpecGAN loss records.'.format(
      wavegan_records.shape[0], specgan_records.shape[0]))
  print('Saved figure to {}.'.format(args.output.resolve()))


if __name__ == '__main__':
  main()
