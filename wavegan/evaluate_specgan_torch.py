"""Evaluate a PyTorch SpecGAN checkpoint using the WaveGAN label-free metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance
import torch

from evaluate_wavegan_torch import (
    PROJECT_ROOT, extract_features, load_real_waveforms,
    normalized_waveform_nearest_neighbor_metrics, plot_feature_distributions, plot_mean_psd,
)
from pytorch_specgan import SpecGANGenerator, invert_spectrogram


def generate_waveforms(generator, count, latent_dim, batch_size, device, mean, std, griffin_lim_iterations):
  generator.eval()
  waveforms = []
  with torch.no_grad():
    for start in range(0, count, batch_size):
      current_batch_size = min(batch_size, count - start)
      latent = torch.randn(current_batch_size, latent_dim, device=device)
      generated_spectrograms = generator(latent)
      generated_waveforms = invert_spectrogram(
          generated_spectrograms, mean, std, griffin_lim_iterations)
      waveforms.append(generated_waveforms.cpu().numpy()[:, 0])
  return np.concatenate(waveforms, axis=0)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', type=Path, required=True, help='PyTorch SpecGAN checkpoint (.pt).')
  parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data' / 'wav')
  parser.add_argument('--sample-rate', type=int, default=576000)
  parser.add_argument('--num-samples', type=int, default=None)
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument('--nn-query-batch-size', type=int, default=1024,
                      help='Nearest-neighbor query block size; lower it if GPU memory is limited.')
  parser.add_argument('--num-workers', type=int, default=0)
  parser.add_argument('--griffin-lim-iterations', type=int, default=16)
  parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
  parser.add_argument('--output-dir', type=Path, default=None)
  args = parser.parse_args()

  if args.device == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA was requested but is unavailable.')
  device_name = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
  device = torch.device('cpu' if device_name == 'auto' else device_name)
  checkpoint = torch.load(str(args.checkpoint), map_location=device, weights_only=False)
  if checkpoint.get('model_type') != 'specgan':
    raise ValueError('{} is not a SpecGAN checkpoint.'.format(args.checkpoint))
  checkpoint_args = checkpoint['args']
  generator = SpecGANGenerator(
      latent_dim=checkpoint_args['latent_dim'],
      dim=checkpoint_args['model_dim'],
      kernel_len=checkpoint_args['kernel_len']).to(device)
  generator.load_state_dict(checkpoint['generator'])
  mean = torch.tensor(checkpoint['spectrogram_mean'], dtype=torch.float32, device=device)
  std = torch.tensor(checkpoint['spectrogram_std'], dtype=torch.float32, device=device)

  real_waveforms = load_real_waveforms(args.data_dir, args.sample_rate, args.batch_size, args.num_workers)
  generated_count = args.num_samples or real_waveforms.shape[0]
  generated_waveforms = generate_waveforms(
      generator, generated_count, checkpoint_args['latent_dim'], args.batch_size, device,
      mean, std, args.griffin_lim_iterations)
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
      'model_type': 'specgan',
      'checkpoint': str(args.checkpoint.resolve()),
      'checkpoint_epoch': int(checkpoint['epoch']),
      'sample_rate_hz': args.sample_rate,
      'real_sample_count': int(real_waveforms.shape[0]),
      'generated_sample_count': int(generated_waveforms.shape[0]),
      'griffin_lim_iterations': args.griffin_lim_iterations,
      'feature_wasserstein_distance': wasserstein_metrics,
      'mean_psd_absolute_error': psd_absolute_error,
      'mean_log10_psd_absolute_error': psd_log_error,
      'normalized_waveform_nearest_neighbor_l2': nearest_neighbor_metrics,
  }
  with (output_dir / 'metrics.json').open('w', encoding='utf-8') as metrics_file:
    json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)
  plot_feature_distributions(real_features, generated_features, output_dir / 'feature_distributions.png')
  plot_mean_psd(real_features, generated_features, output_dir / 'mean_psd_comparison.png')
  print('Evaluated SpecGAN epoch {} using {} real and {} generated waveforms.'.format(
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
