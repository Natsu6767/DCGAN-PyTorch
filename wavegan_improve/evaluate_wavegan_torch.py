"""Evaluate a PyTorch WaveGAN checkpoint against unlabeled real click waveforms.

This is the label-free counterpart to the original WaveGAN README's Inception
Score workflow. It compares real and generated waveform distributions rather
than requiring a domain-specific labeled audio classifier.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
import torch
from torch.utils.data import DataLoader

from pytorch_wavegan import WaveGANGenerator
from train_wavegan_torch import ClickWaveformDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def pulse_width_fwhm(waveform):
  """Return the absolute-amplitude FWHM of one waveform in samples."""
  envelope = np.abs(waveform)
  peak_index = int(np.argmax(envelope))
  peak = float(envelope[peak_index])
  if peak == 0.0:
    return 0.0
  half = peak / 2.0

  left = peak_index
  while left > 0 and envelope[left] >= half:
    left -= 1
  if left == 0 and envelope[left] >= half:
    left_crossing = 0.0
  else:
    left_crossing = left + (half - envelope[left]) / (envelope[left + 1] - envelope[left])

  right = peak_index
  last = envelope.size - 1
  while right < last and envelope[right] >= half:
    right += 1
  if right == last and envelope[right] >= half:
    right_crossing = float(last)
  else:
    right_crossing = (right - 1) + (half - envelope[right - 1]) / (
        envelope[right] - envelope[right - 1])
  return float(right_crossing - left_crossing)


def extract_features(waveforms, sample_rate):
  """Extract pulse and spectral descriptors from (N, 128) waveforms."""
  waveforms = np.asarray(waveforms, dtype=np.float32)
  if waveforms.ndim != 2 or waveforms.shape[1] != 128:
    raise ValueError('Expected waveform array with shape (N, 128), got {}.'.format(waveforms.shape))

  spectrum = np.fft.rfft(waveforms, axis=1)
  power = np.abs(spectrum) ** 2 / waveforms.shape[1]
  frequencies = np.fft.rfftfreq(waveforms.shape[1], d=1.0 / sample_rate)
  non_dc_power = power[:, 1:]
  dominant_indices = np.argmax(non_dc_power, axis=1) + 1
  total_power = power.sum(axis=1)
  safe_total_power = np.maximum(total_power, np.finfo(np.float32).eps)
  spectral_centroid = (power * frequencies[None, :]).sum(axis=1) / safe_total_power
  spectral_bandwidth = np.sqrt(
      (power * (frequencies[None, :] - spectral_centroid[:, None]) ** 2).sum(axis=1) / safe_total_power)

  return {
      'peak_amplitude': np.abs(waveforms).max(axis=1),
      'peak_to_peak': waveforms.max(axis=1) - waveforms.min(axis=1),
      'rms': np.sqrt(np.mean(waveforms ** 2, axis=1)),
      'peak_location_samples': np.argmax(np.abs(waveforms), axis=1),
      'fwhm_samples': np.asarray([pulse_width_fwhm(waveform) for waveform in waveforms]),
      'dominant_frequency_hz': frequencies[dominant_indices],
      'spectral_centroid_hz': spectral_centroid,
      'spectral_bandwidth_hz': spectral_bandwidth,
      'mean_psd': power.mean(axis=0),
      'frequencies_hz': frequencies,
  }


def normalize_waveforms_to_unit_range(waveforms):
  """Individually normalize (N, 128) waveforms to [-1, 1] for L2 matching."""
  waveforms = np.asarray(waveforms, dtype=np.float32)
  minimum = waveforms.min(axis=1, keepdims=True)
  maximum = waveforms.max(axis=1, keepdims=True)
  value_range = maximum - minimum
  normalized = np.zeros_like(waveforms)
  nonconstant = value_range[:, 0] > np.finfo(np.float32).eps
  normalized[nonconstant] = (
      2.0 * (waveforms[nonconstant] - minimum[nonconstant]) /
      value_range[nonconstant] - 1.0)
  return normalized


def nearest_neighbor_distance_stats(reference, query, device, query_batch_size, exclude_self=False):
  """Return exact L2 nearest-neighbor distance mean/std using bounded GPU blocks."""
  if exclude_self and reference.shape != query.shape:
    raise ValueError('Self exclusion requires identically shaped reference and query sets.')
  reference_tensor = torch.as_tensor(reference, dtype=torch.float32, device=device)
  reference_norms = (reference_tensor ** 2).sum(dim=1)
  distances = []
  with torch.no_grad():
    for start in range(0, query.shape[0], query_batch_size):
      end = min(start + query_batch_size, query.shape[0])
      query_tensor = torch.as_tensor(query[start:end], dtype=torch.float32, device=device)
      squared_distances = (
          (query_tensor ** 2).sum(dim=1, keepdim=True) + reference_norms.unsqueeze(0) -
          2.0 * query_tensor.matmul(reference_tensor.t()))
      if exclude_self:
        rows = torch.arange(end - start, device=device)
        columns = torch.arange(start, end, device=device)
        squared_distances[rows, columns] = torch.inf
      distances.append(torch.sqrt(squared_distances.clamp_min(0.0).min(dim=1).values).cpu())
  distances = torch.cat(distances).numpy()
  return {
      'mean_l2': float(distances.mean()),
      'std_l2': float(distances.std()),
  }


def normalized_waveform_nearest_neighbor_metrics(real_waveforms, generated_waveforms, device, query_batch_size):
  """Measure raw-waveform proximity after individual [-1, 1] normalization."""
  normalized_real = normalize_waveforms_to_unit_range(real_waveforms)
  normalized_generated = normalize_waveforms_to_unit_range(generated_waveforms)
  return {
      'normalization': 'per_waveform_minmax_to_minus1_plus1',
      'distance': 'l2_over_128_samples',
      'real_to_real_excluding_self': nearest_neighbor_distance_stats(
          normalized_real, normalized_real, device, query_batch_size, exclude_self=True),
      'generated_to_real': nearest_neighbor_distance_stats(
          normalized_real, normalized_generated, device, query_batch_size),
      'real_to_generated': nearest_neighbor_distance_stats(
          normalized_generated, normalized_real, device, query_batch_size),
  }


def load_real_waveforms(data_dir, sample_rate, batch_size, num_workers):
  dataset = ClickWaveformDataset(data_dir, sample_rate)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
  waveforms = [batch[:, 0].numpy() for batch in loader]
  return np.concatenate(waveforms, axis=0)


def generate_waveforms(generator, count, latent_dim, batch_size, device):
  generator.eval()
  waveforms = []
  with torch.no_grad():
    for start in range(0, count, batch_size):
      current_batch_size = min(batch_size, count - start)
      latent = torch.randn(current_batch_size, latent_dim, device=device)
      waveforms.append(generator(latent).cpu().numpy()[:, 0])
  return np.concatenate(waveforms, axis=0)


def plot_feature_distributions(real_features, generated_features, output_path):
  feature_names = [
      'peak_amplitude', 'peak_to_peak', 'rms', 'peak_location_samples',
      'fwhm_samples', 'dominant_frequency_hz', 'spectral_centroid_hz', 'spectral_bandwidth_hz',
  ]
  axis_labels = {
      'peak_amplitude': 'Peak amplitude (normalized)',
      'peak_to_peak': 'Peak-to-peak amplitude (normalized)',
      'rms': 'RMS amplitude (normalized)',
      'peak_location_samples': 'Peak location (samples)',
      'fwhm_samples': 'FWHM (samples)',
      'dominant_frequency_hz': 'Dominant frequency (Hz)',
      'spectral_centroid_hz': 'Spectral centroid (Hz)',
      'spectral_bandwidth_hz': 'Spectral bandwidth (Hz)',
  }
  figure, axes = plt.subplots(3, 3, figsize=(15, 11))
  for axis, name in zip(axes.flat, feature_names):
    axis.hist(real_features[name], bins=50, density=True, alpha=0.55, label='Real', color='#1769aa')
    axis.hist(generated_features[name], bins=50, density=True, alpha=0.55, label='Generated', color='#e76f51')
    axis.set_title(name.replace('_', ' '))
    axis.set_xlabel(axis_labels[name])
    axis.set_ylabel('Density')
    axis.grid(alpha=0.2)
    axis.legend()
  axes.flat[-1].axis('off')
  figure.suptitle('Real vs generated waveform feature distributions', fontsize=16)
  figure.tight_layout(rect=(0, 0, 1, 0.97))
  figure.savefig(str(output_path), dpi=180)
  plt.close(figure)


def plot_mean_psd(real_features, generated_features, output_path):
  figure, axis = plt.subplots(figsize=(10, 5))
  frequency_khz = real_features['frequencies_hz'] / 1000.0
  axis.semilogy(frequency_khz, real_features['mean_psd'] + 1e-12, label='Real', color='#1769aa')
  axis.semilogy(frequency_khz, generated_features['mean_psd'] + 1e-12, label='Generated', color='#e76f51')
  axis.set_title('Mean power spectral density')
  axis.set_xlabel('Frequency (kHz)')
  axis.set_ylabel('Power / bin')
  axis.grid(alpha=0.2)
  axis.legend()
  figure.tight_layout()
  figure.savefig(str(output_path), dpi=180)
  plt.close(figure)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', type=Path, required=True, help='PyTorch WaveGAN checkpoint (.pt).')
  parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data' / 'wav')
  parser.add_argument('--sample-rate', type=int, default=576000)
  parser.add_argument('--num-samples', type=int, default=None, help='Generated sample count; default matches real data.')
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument('--nn-query-batch-size', type=int, default=1024,
                      help='Nearest-neighbor query block size; lower it if GPU memory is limited.')
  parser.add_argument('--num-workers', type=int, default=0)
  parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
  parser.add_argument('--output-dir', type=Path, default=None)
  parser.add_argument('--metrics-output', type=Path,
                      help='Optional JSON metrics path; defaults to output-dir/metrics.json.')
  args = parser.parse_args()

  if args.device == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA was requested but is unavailable.')
  device_name = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
  if device_name == 'auto':
    device_name = 'cpu'
  device = torch.device(device_name)

  checkpoint = torch.load(str(args.checkpoint), map_location=device, weights_only=False)
  checkpoint_args = checkpoint['args']
  latent_dim = checkpoint_args['latent_dim']
  generator = WaveGANGenerator(
      latent_dim=latent_dim,
      dim=checkpoint_args['model_dim'],
      kernel_len=checkpoint_args['kernel_len']).to(device)
  generator.load_state_dict(checkpoint['generator'])

  real_waveforms = load_real_waveforms(args.data_dir, args.sample_rate, args.batch_size, args.num_workers)
  generated_count = args.num_samples or real_waveforms.shape[0]
  generated_waveforms = generate_waveforms(generator, generated_count, latent_dim, args.batch_size, device)
  real_features = extract_features(real_waveforms, args.sample_rate)
  generated_features = extract_features(generated_waveforms, args.sample_rate)

  output_dir = args.output_dir or args.checkpoint.parent / 'evaluation_{}'.format(args.checkpoint.stem)
  output_dir.mkdir(parents=True, exist_ok=True)
  feature_names = [name for name in real_features if name not in ('mean_psd', 'frequencies_hz')]
  wasserstein_metrics = {
      name: float(wasserstein_distance(real_features[name], generated_features[name]))
      for name in feature_names
  }
  psd_absolute_error = float(np.mean(np.abs(real_features['mean_psd'] - generated_features['mean_psd'])))
  psd_log_error = float(np.mean(np.abs(
      np.log10(real_features['mean_psd'] + 1e-12) - np.log10(generated_features['mean_psd'] + 1e-12))))
  nearest_neighbor_metrics = normalized_waveform_nearest_neighbor_metrics(
      real_waveforms, generated_waveforms, device, args.nn_query_batch_size)
  metrics = {
      'checkpoint': str(args.checkpoint.resolve()),
      'checkpoint_epoch': int(checkpoint['epoch']),
      'sample_rate_hz': args.sample_rate,
      'real_sample_count': int(real_waveforms.shape[0]),
      'generated_sample_count': int(generated_waveforms.shape[0]),
      'feature_wasserstein_distance': wasserstein_metrics,
      'mean_psd_absolute_error': psd_absolute_error,
      'mean_log10_psd_absolute_error': psd_log_error,
      'normalized_waveform_nearest_neighbor_l2': nearest_neighbor_metrics,
      'feature_summary': {
          'real': {
              name: {'mean': float(np.mean(real_features[name])), 'std': float(np.std(real_features[name]))}
              for name in feature_names
          },
          'generated': {
              name: {'mean': float(np.mean(generated_features[name])), 'std': float(np.std(generated_features[name]))}
              for name in feature_names
          },
      },
  }
  metrics_output = args.metrics_output or output_dir / 'metrics.json'
  metrics_output.parent.mkdir(parents=True, exist_ok=True)
  with metrics_output.open('w', encoding='utf-8') as metrics_file:
    json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)
  plot_feature_distributions(real_features, generated_features, output_dir / 'feature_distributions.png')
  plot_mean_psd(real_features, generated_features, output_dir / 'mean_psd_comparison.png')

  print('Evaluated checkpoint epoch {} using {} real and {} generated waveforms.'.format(
      metrics['checkpoint_epoch'], metrics['real_sample_count'], metrics['generated_sample_count']))
  print('Saved metrics and plots to {}.'.format(output_dir.resolve()))
  for name, value in wasserstein_metrics.items():
    print('Wasserstein {}: {:.6g}'.format(name, value))
  print('Mean PSD absolute error: {:.6g}'.format(psd_absolute_error))
  print('Mean log10 PSD absolute error: {:.6g}'.format(psd_log_error))
  for name in ('real_to_real_excluding_self', 'generated_to_real', 'real_to_generated'):
    values = nearest_neighbor_metrics[name]
    print('Normalized waveform NN {} L2: {:.6g} +- {:.6g}'.format(
        name, values['mean_l2'], values['std_l2']))


if __name__ == '__main__':
  main()
