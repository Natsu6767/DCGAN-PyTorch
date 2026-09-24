"""Train and evaluate WaveGAN runs with peak-loss weights from 0.1 through 1.0.

Training transcripts and JSON metrics are written below ``log``.  Per-alpha
feature/PSD comparisons and an alpha summary are written below ``images``.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_ALPHAS = tuple(round(index / 10, 1) for index in range(1, 11))


def alpha_name(alpha):
  return '{:.1f}'.format(alpha)


def progress_bar(completed, total, label):
  width = 30
  filled = round(width * completed / total)
  bar = '#' * filled + '-' * (width - filled)
  print('Overall progress: [{}] {:6.2f}% | {}'.format(
      bar, 100.0 * completed / total, label), flush=True)


def run_and_log(command, log_path, run_index, run_count, epochs, alpha):
  """Run one command, retain its transcript, and print sweep-level progress."""
  epoch_pattern = re.compile(r'Epoch (\d+)/(\d+) \| batch (\d+)/(\d+)')
  last_reported_epoch = 0
  with log_path.open('w', encoding='utf-8') as log_handle:
    log_handle.write('Command: {}\n\n'.format(' '.join(map(str, command))))
    process = subprocess.Popen(
        command, cwd=str(SCRIPT_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace')
    assert process.stdout is not None
    for line in process.stdout:
      log_handle.write(line)
      match = epoch_pattern.search(line)
      if not match:
        continue
      epoch, total_epochs, batch, total_batches = map(int, match.groups())
      if batch == total_batches and (epoch == 1 or epoch == total_epochs or epoch - last_reported_epoch >= 10):
        last_reported_epoch = epoch
        completed = run_index * epochs + epoch
        progress_bar(completed, run_count * epochs, 'alpha {} training epoch {}/{}'.format(
            alpha_name(alpha), epoch, total_epochs))
    return_code = process.wait()
  if return_code:
    raise subprocess.CalledProcessError(return_code, command)


def plot_sweep_summary(results, output_path):
  """Visualize alpha-dependent fit errors and peak-amplitude statistics."""
  alphas = [result['alpha'] for result in results]
  metrics = [result['metrics'] for result in results]
  plots = [
      ('Peak amplitude Wasserstein distance',
       [metric['feature_wasserstein_distance']['peak_amplitude'] for metric in metrics]),
      ('RMS Wasserstein distance',
       [metric['feature_wasserstein_distance']['rms'] for metric in metrics]),
      ('Peak-to-peak Wasserstein distance',
       [metric['feature_wasserstein_distance']['peak_to_peak'] for metric in metrics]),
      ('Mean PSD absolute error', [metric['mean_psd_absolute_error'] for metric in metrics]),
      ('Mean log10 PSD absolute error', [metric['mean_log10_psd_absolute_error'] for metric in metrics]),
      ('Generated peak-amplitude mean',
       [metric['feature_summary']['generated']['peak_amplitude']['mean'] for metric in metrics]),
  ]
  real_peak_mean = metrics[0]['feature_summary']['real']['peak_amplitude']['mean']
  figure, axes = plt.subplots(2, 3, figsize=(15, 8))
  for axis, (title, values) in zip(axes.flat, plots):
    axis.plot(alphas, values, marker='o', color='#e76f51')
    axis.set_title(title)
    axis.set_xlabel('Peak-loss weight (alpha)')
    axis.grid(alpha=0.25)
  axes.flat[-1].axhline(real_peak_mean, color='#1769aa', linestyle='--', label='Real-data mean')
  axes.flat[-1].legend()
  figure.suptitle('WaveGAN peak-loss sweep: generated versus training-data statistics', fontsize=15)
  figure.tight_layout(rect=(0, 0, 1, 0.95))
  figure.savefig(output_path, dpi=180)
  plt.close(figure)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--epochs', type=int, default=200)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
  parser.add_argument('--seed', type=int, default=369)
  parser.add_argument('--checkpoint-every', type=int, default=200,
                      help='Saving only the final checkpoint avoids retaining 2,000 checkpoint files.')
  parser.add_argument('--alphas', type=float, nargs='+', default=DEFAULT_ALPHAS)
  parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data' / 'wav')
  args = parser.parse_args()

  if args.epochs < 1:
    raise ValueError('--epochs must be positive.')
  if not args.alphas or any(alpha < 0 for alpha in args.alphas):
    raise ValueError('--alphas must contain at least one non-negative value.')

  log_dir = SCRIPT_DIR / 'log'
  image_dir = SCRIPT_DIR / 'images'
  checkpoint_dir = SCRIPT_DIR / 'checkpoints'
  for directory in (log_dir, image_dir, checkpoint_dir):
    directory.mkdir(parents=True, exist_ok=True)

  results = []
  total_runs = len(args.alphas)
  progress_bar(0, total_runs * args.epochs, 'starting peak-loss sweep')
  for index, alpha in enumerate(args.alphas):
    name = alpha_name(alpha)
    run_dir = checkpoint_dir / 'alpha_{}'.format(name)
    image_output_dir = image_dir / 'alpha_{}'.format(name)
    checkpoint = run_dir / 'wavegan_epoch_{:04d}.pt'.format(args.epochs)
    train_log = log_dir / 'alpha_{}_training.log'.format(name)
    epoch_log = log_dir / 'alpha_{}_epochs.jsonl'.format(name)
    metrics_path = log_dir / 'alpha_{}_metrics.json'.format(name)
    print('\nStarting alpha={} ({}/{})'.format(name, index + 1, total_runs), flush=True)

    train_command = [
        sys.executable, str(SCRIPT_DIR / 'train_wavegan_torch.py'),
        '--data-dir', str(args.data_dir), '--output-dir', str(run_dir),
        '--epochs', str(args.epochs), '--batch-size', str(args.batch_size),
        '--device', args.device, '--seed', str(args.seed),
        '--checkpoint-every', str(args.checkpoint_every),
        '--peak-loss-weight', name, '--log-file', str(epoch_log),
    ]
    run_and_log(train_command, train_log, index, total_runs, args.epochs, alpha)

    evaluate_command = [
        sys.executable, str(SCRIPT_DIR / 'evaluate_wavegan_torch.py'),
        '--checkpoint', str(checkpoint), '--data-dir', str(args.data_dir),
        '--batch-size', str(args.batch_size), '--device', args.device,
        '--output-dir', str(image_output_dir), '--metrics-output', str(metrics_path),
    ]
    evaluation_log = log_dir / 'alpha_{}_evaluation.log'.format(name)
    run_and_log(evaluate_command, evaluation_log, index, total_runs, args.epochs, alpha)
    with metrics_path.open(encoding='utf-8') as handle:
      results.append({'alpha': alpha, 'metrics': json.load(handle)})
    progress_bar((index + 1) * args.epochs, total_runs * args.epochs,
                 'alpha {} training and evaluation complete'.format(name))

  summary_path = image_dir / 'peak_loss_alpha_sweep_summary.png'
  plot_sweep_summary(results, summary_path)
  summary_json = log_dir / 'peak_loss_alpha_sweep_summary.json'
  with summary_json.open('w', encoding='utf-8') as handle:
    json.dump(results, handle, indent=2, ensure_ascii=False)
  print('\nCompleted all {} alpha values.'.format(total_runs))
  print('Logs: {}'.format(log_dir.resolve()))
  print('Images: {}'.format(image_dir.resolve()))


if __name__ == '__main__':
  main()
