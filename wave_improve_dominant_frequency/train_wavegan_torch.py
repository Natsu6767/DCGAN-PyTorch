"""Train WaveGAN with differentiable dominant-frequency distribution alignment."""

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.nn import functional as functional
from torch.optim import Adam
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_EXPERIMENT_DIR = PROJECT_ROOT / 'wavegan_improve'
sys.path.insert(0, str(BASE_EXPERIMENT_DIR))

from pytorch_wavegan import WaveGANDiscriminator, WaveGANGenerator, weights_init
from train_wavegan_torch import ClickWaveformDataset, gradient_penalty, save_checkpoint, save_preview


def extract_soft_dominant_frequency(audio, sample_rate, n_fft=128, hop_length=32, tau=0.05):
  """Return a differentiable dominant-frequency estimate for each waveform."""
  window = torch.hann_window(n_fft, device=audio.device)
  spectrum = torch.stft(
      audio.squeeze(1), n_fft=n_fft, hop_length=hop_length,
      win_length=n_fft, window=window, return_complex=True)
  mean_magnitude = torch.abs(spectrum).mean(dim=-1)[:, 1:]
  frequencies = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate, device=audio.device)[1:]
  normalized_magnitude = mean_magnitude / (
      mean_magnitude.max(dim=-1, keepdim=True).values * tau + 1e-7)
  weights = functional.softmax(normalized_magnitude, dim=-1)
  return torch.sum(weights * frequencies.unsqueeze(0), dim=-1)


def dominant_frequency_distribution_loss(real_audio, fake_audio, sample_rate):
  """Return the sorted-L1 1D Wasserstein loss between Batch frequency distributions."""
  real_frequencies = extract_soft_dominant_frequency(real_audio, sample_rate)
  fake_frequencies = extract_soft_dominant_frequency(fake_audio, sample_rate)
  wasserstein_distance = functional.l1_loss(
      torch.sort(fake_frequencies).values, torch.sort(real_frequencies).values)
  return wasserstein_distance / (sample_rate / 2.0)


def write_epoch_log(log_file, epoch, discriminator_losses, adversarial_losses,
                    dominant_frequency_losses, total_generator_losses, loss_weight):
  """Append one machine-readable training summary for an epoch."""
  def mean_or_none(values):
    return None if not values else float(np.mean(values))

  record = {
      'epoch': epoch,
      'dominant_frequency_loss_weight': loss_weight,
      'mean_discriminator_loss': mean_or_none(discriminator_losses),
      'mean_generator_adversarial_loss': mean_or_none(adversarial_losses),
      'mean_generator_dominant_frequency_loss': mean_or_none(dominant_frequency_losses),
      'mean_generator_total_loss': mean_or_none(total_generator_losses),
      'generator_update_count': len(total_generator_losses),
  }
  with log_file.open('a', encoding='utf-8') as handle:
    handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data' / 'wav')
  parser.add_argument('--output-dir', type=Path, default=Path('train_click_torch'))
  parser.add_argument('--epochs', type=int, default=200)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--latent-dim', type=int, default=100)
  parser.add_argument('--model-dim', type=int, default=64)
  parser.add_argument('--kernel-len', type=int, default=25)
  parser.add_argument('--sample-rate', type=int, default=576000)
  parser.add_argument('--n-critic', type=int, default=5)
  parser.add_argument('--gradient-penalty', type=float, default=10.0)
  parser.add_argument('--dominant-frequency-loss-weight', type=float, default=1.0)
  parser.add_argument('--learning-rate', type=float, default=1e-4)
  parser.add_argument('--num-workers', type=int, default=0)
  parser.add_argument('--seed', type=int, default=369)
  parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
  parser.add_argument('--resume', type=Path)
  parser.add_argument('--checkpoint-every', type=int, default=1)
  parser.add_argument('--log-file', type=Path,
                      help='Optional JSONL file receiving one loss summary per epoch.')
  args = parser.parse_args()

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

  if args.device == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA was requested but PyTorch cannot access a CUDA device.')
  device_name = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
  if device_name == 'auto':
    device_name = 'cpu'
  device = torch.device(device_name)
  if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    print('Using CUDA device: {}'.format(torch.cuda.get_device_name(device)))
  else:
    print('Using CPU. Install a CUDA-enabled PyTorch build to train on the RTX 4060.')

  dataset = ClickWaveformDataset(args.data_dir, args.sample_rate)
  loader = DataLoader(
      dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
      num_workers=args.num_workers, pin_memory=device.type == 'cuda')
  if len(loader) == 0:
    raise ValueError('Batch size {} exceeds dataset size {}.'.format(args.batch_size, len(dataset)))
  print('Loaded {} click waveforms from {}.'.format(len(dataset), args.data_dir.resolve()))

  generator = WaveGANGenerator(args.latent_dim, dim=args.model_dim, kernel_len=args.kernel_len).to(device)
  discriminator = WaveGANDiscriminator(dim=args.model_dim, kernel_len=args.kernel_len).to(device)
  generator.apply(weights_init)
  discriminator.apply(weights_init)
  generator_optimizer = Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
  discriminator_optimizer = Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
  start_epoch = 1

  if args.resume:
    checkpoint = torch.load(str(args.resume), map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
    discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    print('Resumed from {} at epoch {}.'.format(args.resume, start_epoch))

  args.output_dir.mkdir(parents=True, exist_ok=True)
  if args.log_file:
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open('w', encoding='utf-8') as handle:
      handle.write(json.dumps({
          'type': 'configuration',
          'dominant_frequency_n_fft': 128,
          'dominant_frequency_hop_length': 32,
          'dominant_frequency_tau': 0.05,
          'dominant_frequency_loss_weight': args.dominant_frequency_loss_weight,
          'epochs': args.epochs,
          'batch_size': args.batch_size,
          'seed': args.seed,
      }, ensure_ascii=False) + '\n')
  fixed_noise = torch.randn(16, args.latent_dim, device=device)
  global_step = 0
  for epoch in range(start_epoch, args.epochs + 1):
    generator.train()
    discriminator.train()
    generator_loss = None
    epoch_discriminator_losses = []
    epoch_adversarial_losses = []
    epoch_dominant_frequency_losses = []
    epoch_generator_total_losses = []
    for batch_index, real_waveforms in enumerate(loader, 1):
      real_waveforms = real_waveforms.to(device, non_blocking=True)
      batch_size = real_waveforms.size(0)

      fake_waveforms = generator(torch.randn(batch_size, args.latent_dim, device=device)).detach()
      discriminator_optimizer.zero_grad()
      discriminator_loss = (
          discriminator(fake_waveforms).mean() - discriminator(real_waveforms).mean()
          + args.gradient_penalty * gradient_penalty(discriminator, real_waveforms, fake_waveforms))
      discriminator_loss.backward()
      discriminator_optimizer.step()
      epoch_discriminator_losses.append(discriminator_loss.item())

      if global_step % args.n_critic == 0:
        generator_optimizer.zero_grad()
        generated_waveforms = generator(torch.randn(batch_size, args.latent_dim, device=device))
        adversarial_loss = -discriminator(generated_waveforms).mean()
        batch_dominant_frequency_loss = dominant_frequency_distribution_loss(
            real_waveforms, generated_waveforms, args.sample_rate)
        generator_loss = adversarial_loss + (
            args.dominant_frequency_loss_weight * batch_dominant_frequency_loss)
        generator_loss.backward()
        generator_optimizer.step()
        epoch_adversarial_losses.append(adversarial_loss.item())
        epoch_dominant_frequency_losses.append(batch_dominant_frequency_loss.item())
        epoch_generator_total_losses.append(generator_loss.item())

      global_step += 1
      if batch_index % 50 == 0 or batch_index == len(loader):
        generator_value = float('nan') if generator_loss is None else generator_loss.item()
        print(
            'Epoch {}/{} | batch {}/{} | D={:.5f} | G={:.5f}'.format(
                epoch, args.epochs, batch_index, len(loader), discriminator_loss.item(), generator_value),
            flush=True)

    if args.log_file:
      write_epoch_log(
          args.log_file, epoch, epoch_discriminator_losses, epoch_adversarial_losses,
          epoch_dominant_frequency_losses, epoch_generator_total_losses,
          args.dominant_frequency_loss_weight)

    if epoch % args.checkpoint_every == 0:
      save_checkpoint(
          args.output_dir / 'wavegan_epoch_{:04d}.pt'.format(epoch), epoch,
          generator, discriminator, generator_optimizer, discriminator_optimizer, args)
      save_preview(generator, fixed_noise, args.output_dir, epoch, args.sample_rate)


if __name__ == '__main__':
  main()
