"""Compare baseline WaveGAN and peak-loss WaveGAN evaluations across alpha.

Every metric uses the label-free evaluation protocol in
``wavegan/compare_gan_evaluations.py``: feature Wasserstein distances, PSD
errors, and normalized waveform nearest-neighbour L2 distances.  Lower values
are better for all displayed rows.
"""

import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASELINE_METRICS = (
    PROJECT_ROOT / 'wavegan' / 'train_click_wavegan_torch_gpu' /
    'evaluation_wavegan_epoch_0200_nn' / 'metrics.json')

FEATURE_METRICS = (
    'peak_amplitude',
    'peak_to_peak',
    'rms',
    'peak_location_samples',
    'fwhm_samples',
    'dominant_frequency_hz',
    'spectral_centroid_hz',
    'spectral_bandwidth_hz',
)
METRICS = (
    *[(name, 'feature_wasserstein_distance.{}'.format(name)) for name in FEATURE_METRICS],
    ('mean_psd_absolute_error', 'mean_psd_absolute_error'),
    ('mean_log10_psd_absolute_error', 'mean_log10_psd_absolute_error'),
    ('NN L2: generated to real', 'normalized_waveform_nearest_neighbor_l2.generated_to_real.mean_l2'),
    ('NN L2: real to generated', 'normalized_waveform_nearest_neighbor_l2.real_to_generated.mean_l2'),
)


def read_json(path):
  with path.open(encoding='utf-8') as handle:
    return json.load(handle)


def nested_value(values, dotted_path):
  result = values
  for key in dotted_path.split('.'):
    result = result[key]
  return float(result)


def discover_alpha_metrics(log_dir):
  """Return alpha-to-metrics mappings, rejecting ambiguous or incomplete inputs."""
  metric_paths = {}
  pattern = re.compile(r'^alpha_(\d+\.\d+)_metrics\.json$')
  for path in log_dir.glob('alpha_*_metrics.json'):
    match = pattern.match(path.name)
    if match:
      alpha = float(match.group(1))
      metric_paths[alpha] = path
  if not metric_paths:
    raise FileNotFoundError('No alpha metrics files found in {}.'.format(log_dir))
  return {alpha: read_json(path) for alpha, path in sorted(metric_paths.items())}


def comparison_rows(baseline, alpha_metrics):
  models = [(0.0, baseline), *sorted(alpha_metrics.items())]
  rows = []
  for display_name, value_path in METRICS:
    values = {alpha: nested_value(metrics, value_path) for alpha, metrics in models}
    minimum = min(values.values())
    winners = [alpha for alpha, value in values.items() if np.isclose(value, minimum, rtol=1e-12, atol=1e-15)]
    rows.append({
        'metric': display_name,
        'lower_is_better': True,
        'values': values,
        'best_alpha': winners,
    })
  return rows


def write_machine_readable_outputs(rows, output_dir):
  output_dir.mkdir(parents=True, exist_ok=True)
  alphas = sorted(rows[0]['values'])
  csv_path = output_dir / 'peak_loss_alpha_metric_comparison.csv'
  fieldnames = ['metric', 'lower_is_better', 'best_alpha'] + ['alpha_{:.1f}'.format(alpha) for alpha in alphas]
  with csv_path.open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow({
          'metric': row['metric'],
          'lower_is_better': row['lower_is_better'],
          'best_alpha': ', '.join('{:.1f}'.format(alpha) for alpha in row['best_alpha']),
          **{'alpha_{:.1f}'.format(alpha): row['values'][alpha] for alpha in alphas},
      })
  json_path = output_dir / 'peak_loss_alpha_metric_comparison.json'
  serializable_rows = [{
      'metric': row['metric'],
      'lower_is_better': row['lower_is_better'],
      'best_alpha': row['best_alpha'],
      'values': {'{:.1f}'.format(alpha): value for alpha, value in row['values'].items()},
  } for row in rows]
  with json_path.open('w', encoding='utf-8') as handle:
    json.dump(serializable_rows, handle, indent=2, ensure_ascii=False)
  return csv_path, json_path


def plot_comparison_table(rows, output_path):
  alphas = sorted(rows[0]['values'])
  column_labels = ['Metric\n(lower is better)'] + [
      'α={:.1f}{}'.format(alpha, '\n(baseline)' if alpha == 0.0 else '') for alpha in alphas]
  cell_text = [[row['metric']] + [
      '{:.6g}'.format(row['values'][alpha]) for alpha in alphas
  ] for row in rows]

  figure, axis = plt.subplots(figsize=(22, 4.8))
  axis.axis('off')
  table = axis.table(
      cellText=cell_text, colLabels=column_labels, cellLoc='center', loc='upper center',
      colWidths=[0.22] + [0.071] * len(alphas))
  table.auto_set_font_size(False)
  table.set_fontsize(8)
  table.scale(1, 1.55)

  for column_index in range(len(column_labels)):
    cell = table[(0, column_index)]
    cell.set_facecolor('#1f4e78')
    cell.set_text_props(color='white', weight='bold')
  for row_index, row in enumerate(rows, start=1):
    table[(row_index, 0)].set_facecolor('#eaf2f8')
    for column_index, alpha in enumerate(alphas, start=1):
      cell = table[(row_index, column_index)]
      if alpha in row['best_alpha']:
        cell.set_facecolor('#fff2f2')
        cell.set_text_props(color='#d00000', weight='bold')

  axis.set_title('WaveGAN peak-loss sweep versus baseline (red = best / lowest)',
                 fontsize=15, weight='bold', pad=18)
  figure.tight_layout()
  figure.savefig(output_path, dpi=220, bbox_inches='tight')
  plt.close(figure)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--baseline-metrics', type=Path, default=BASELINE_METRICS)
  parser.add_argument('--alpha-log-dir', type=Path, default=SCRIPT_DIR / 'log')
  parser.add_argument('--output-dir', type=Path, default=SCRIPT_DIR / 'images')
  parser.add_argument('--summary-dir', type=Path, default=SCRIPT_DIR / 'log')
  args = parser.parse_args()

  baseline = read_json(args.baseline_metrics)
  alpha_metrics = discover_alpha_metrics(args.alpha_log_dir)
  rows = comparison_rows(baseline, alpha_metrics)
  csv_path, json_path = write_machine_readable_outputs(rows, args.summary_dir)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  image_path = args.output_dir / 'peak_loss_alpha_metric_comparison.png'
  plot_comparison_table(rows, image_path)

  print('Saved comparison image to {}.'.format(image_path.resolve()))
  print('Saved comparison data to {} and {}.'.format(csv_path.resolve(), json_path.resolve()))
  for row in rows:
    print('{}: best alpha={}'.format(
        row['metric'], ', '.join('{:.1f}'.format(alpha) for alpha in row['best_alpha'])))


if __name__ == '__main__':
  main()
