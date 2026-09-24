"""Plot separately normalized WaveGAN preview WAVs every N epochs in one figure."""

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


PREVIEW_PATTERN = re.compile(r'^preview_epoch_(\d+)\.wav$')


def collect_previews(preview_dir, interval):
  """Return (epoch, path) pairs whose epoch is an interval multiple."""
  previews = []
  for path in preview_dir.glob('preview_epoch_*.wav'):
    match = PREVIEW_PATTERN.match(path.name)
    if match is None:
      continue
    epoch = int(match.group(1))
    if epoch % interval == 0:
      previews.append((epoch, path))
  return sorted(previews)


def normalize_waveform(waveform):
  """Independently rescale one waveform to [-1, 1]."""
  waveform = np.asarray(waveform, dtype=np.float32)
  if waveform.ndim == 2:
    waveform = waveform.mean(axis=1)
  if waveform.ndim != 1:
    raise ValueError('Expected mono or stereo waveform, found shape {}.'.format(waveform.shape))
  peak = np.max(np.abs(waveform))
  return waveform / peak if peak > 0 else waveform


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--preview-dir',
      type=Path,
      default=Path(__file__).resolve().parent / 'train_click_torch_gpu',
      help='Directory containing preview_epoch_XXXX.wav files.')
  parser.add_argument('--interval', type=int, default=10, help='Epoch interval between plotted previews.')
  parser.add_argument(
      '--output',
      type=Path,
      default=None,
      help='Output PNG path. Defaults to preview_every_<interval>_epochs.png in preview-dir.')
  parser.add_argument('--columns', type=int, default=5, help='Number of subplot columns.')
  args = parser.parse_args()

  if args.interval <= 0:
    raise ValueError('--interval must be positive.')
  if args.columns <= 0:
    raise ValueError('--columns must be positive.')

  previews = collect_previews(args.preview_dir, args.interval)
  if not previews:
    raise FileNotFoundError(
        'No preview_epoch_XXXX.wav files divisible by {} found in {}.'.format(
            args.interval, args.preview_dir))

  output_path = args.output or args.preview_dir / 'preview_every_{}_epochs.png'.format(args.interval)
  rows = int(np.ceil(len(previews) / float(args.columns)))
  figure, axes = plt.subplots(rows, args.columns, figsize=(args.columns * 4.2, rows * 3.0), squeeze=False)

  for axis, (epoch, path) in zip(axes.flat, previews):
    _, waveform = wavfile.read(str(path))
    waveform = normalize_waveform(waveform)
    axis.plot(np.arange(waveform.size), waveform, linewidth=1.0, color='#1769aa')
    axis.axhline(0.0, color='#777777', linewidth=0.6)
    axis.set_title('Epoch {}'.format(epoch))
    axis.set_xlim(0, max(1, waveform.size - 1))
    axis.set_ylim(-1.05, 1.05)
    axis.set_xlabel('Sample index')
    axis.set_ylabel('Normalized amplitude')
    axis.grid(alpha=0.22)

  for axis in axes.flat[len(previews):]:
    axis.remove()

  figure.suptitle('WaveGAN previews: independently normalized to [-1, 1]', fontsize=16, y=0.995)
  figure.tight_layout(rect=(0, 0, 1, 0.985))
  output_path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(str(output_path), dpi=180)
  plt.close(figure)
  print('Saved {} normalized preview waveforms to {}.'.format(len(previews), output_path.resolve()))


if __name__ == '__main__':
  main()
