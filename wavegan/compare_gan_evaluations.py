"""Compare label-free evaluation JSON files from WaveGAN and SpecGAN."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def metric_rows(wavegan_metrics, specgan_metrics):
  names = sorted(wavegan_metrics['feature_wasserstein_distance'])
  names += ['mean_psd_absolute_error', 'mean_log10_psd_absolute_error']
  rows = []
  for name in names:
    if name in wavegan_metrics['feature_wasserstein_distance']:
      wavegan_value = wavegan_metrics['feature_wasserstein_distance'][name]
      specgan_value = specgan_metrics['feature_wasserstein_distance'][name]
    else:
      wavegan_value = wavegan_metrics[name]
      specgan_value = specgan_metrics[name]
    winner = 'WaveGAN' if wavegan_value < specgan_value else 'SpecGAN' if specgan_value < wavegan_value else 'Tie'
    rows.append({
        'metric': name,
        'wavegan': wavegan_value,
        'specgan': specgan_value,
        'winner_lower_is_better': winner,
        'specgan_to_wavegan_ratio': specgan_value / wavegan_value if wavegan_value else None,
    })
  for name in ('generated_to_real', 'real_to_generated'):
    wavegan_value = wavegan_metrics['normalized_waveform_nearest_neighbor_l2'][name]['mean_l2']
    specgan_value = specgan_metrics['normalized_waveform_nearest_neighbor_l2'][name]['mean_l2']
    winner = 'WaveGAN' if wavegan_value < specgan_value else 'SpecGAN' if specgan_value < wavegan_value else 'Tie'
    display_name = 'NN L2: generated to real' if name == 'generated_to_real' else 'NN L2: real to generated'
    rows.append({
        'metric': display_name,
        'wavegan': wavegan_value,
        'specgan': specgan_value,
        'winner_lower_is_better': winner,
        'specgan_to_wavegan_ratio': specgan_value / wavegan_value if wavegan_value else None,
    })
  return rows


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--wavegan-metrics', type=Path, required=True)
  parser.add_argument('--specgan-metrics', type=Path, required=True)
  parser.add_argument('--output-dir', type=Path, default=Path('gan_comparison'))
  args = parser.parse_args()

  with args.wavegan_metrics.open(encoding='utf-8') as metrics_file:
    wavegan_metrics = json.load(metrics_file)
  with args.specgan_metrics.open(encoding='utf-8') as metrics_file:
    specgan_metrics = json.load(metrics_file)
  rows = metric_rows(wavegan_metrics, specgan_metrics)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  with (args.output_dir / 'comparison.csv').open('w', newline='', encoding='utf-8') as comparison_file:
    writer = csv.DictWriter(comparison_file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  with (args.output_dir / 'comparison.json').open('w', encoding='utf-8') as comparison_file:
    json.dump(rows, comparison_file, indent=2, ensure_ascii=False)

  figure, axis = plt.subplots(figsize=(13, 0.48 * len(rows) + 2))
  axis.axis('off')
  table = axis.table(
      cellText=[[row['metric'], '{:.6g}'.format(row['wavegan']), '{:.6g}'.format(row['specgan']), row['winner_lower_is_better']]
                for row in rows],
      colLabels=['Metric (lower is better)', 'WaveGAN', 'SpecGAN', 'Better model'],
      cellLoc='center', loc='center')
  table.auto_set_font_size(False)
  table.set_fontsize(10)
  table.scale(1, 1.35)
  figure.suptitle('WaveGAN vs SpecGAN: label-free evaluation', fontsize=14)
  figure.tight_layout()
  figure.savefig(str(args.output_dir / 'comparison_table.png'), dpi=180, bbox_inches='tight')
  plt.close(figure)
  print('Saved comparison to {}.'.format(args.output_dir.resolve()))
  for row in rows:
    print('{}: WaveGAN={:.6g}, SpecGAN={:.6g}, better={}'.format(
        row['metric'], row['wavegan'], row['specgan'], row['winner_lower_is_better']))


if __name__ == '__main__':
  main()
