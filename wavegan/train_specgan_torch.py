"""Train a short-pulse SpecGAN with PyTorch on the bundled click WAV files."""

import argparse
from pathlib import Path
import random

import numpy as np
from scipy.io import wavfile
import torch
from torch import autograd
from torch.optim import Adam
from torch.utils.data import DataLoader

from pytorch_specgan import (
    SpecGANDiscriminator, SpecGANGenerator, invert_spectrogram,
    normalized_spectrogram, waveform_to_log_magnitude, weights_init,
)
from train_wavegan_torch import ClickWaveformDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def compute_spectrogram_moments(loader, device):
  """Compute global log-magnitude moments from the normalized training waveforms."""
  sample_sum = torch.zeros((), device=device)
  squared_sum = torch.zeros((), device=device)
  count = 0
  with torch.no_grad():
    for waveforms in loader:
      log_magnitude = waveform_to_log_magnitude(waveforms.to(device, non_blocking=True))
      sample_sum += log_magnitude.sum()
      squared_sum += (log_magnitude ** 2).sum()
      count += log_magnitude.numel()
  mean = sample_sum / count
  variance = (squared_sum / count - mean ** 2).clamp_min(1e-8)
  return mean, variance.sqrt()


def gradient_penalty(discriminator, real_spectrograms, fake_spectrograms):
  batch_size = real_spectrograms.size(0)
  alpha = torch.rand(batch_size, 1, 1, 1, device=real_spectrograms.device)
  mixed = alpha * real_spectrograms + (1.0 - alpha) * fake_spectrograms
  mixed.requires_grad_(True)
  scores = discriminator(mixed)
  gradients = autograd.grad(
      outputs=scores, inputs=mixed, grad_outputs=torch.ones_like(scores),
      create_graph=True, retain_graph=True, only_inputs=True)[0]
  slopes = gradients.reshape(batch_size, -1).norm(2, dim=1)
  return ((slopes - 1.0) ** 2).mean()


def save_preview(generator, fixed_noise, output_dir, epoch, sample_rate, mean, std, griffin_lim_iterations):
  generator.eval()
  with torch.no_grad():
    spectrogram = generator(fixed_noise[:1])
    waveform = invert_spectrogram(spectrogram, mean, std, griffin_lim_iterations)[0, 0]
  generator.train()
  encoded = np.clip(waveform.cpu().numpy() * 32767.0, -32767.0, 32767.0).astype(np.int16)
  wavfile.write(str(output_dir / 'preview_epoch_{:04d}.wav'.format(epoch)), sample_rate, encoded)


def save_checkpoint(path, epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, args, mean, std):
  torch.save({
      'model_type': 'specgan',
      'epoch': epoch,
      'generator': generator.state_dict(),
      'discriminator': discriminator.state_dict(),
      'generator_optimizer': generator_optimizer.state_dict(),
      'discriminator_optimizer': discriminator_optimizer.state_dict(),
      'spectrogram_mean': float(mean.detach().cpu()),
      'spectrogram_std': float(std.detach().cpu()),
      'args': vars(args),
  }, str(path))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data' / 'wav')
  parser.add_argument('--output-dir', type=Path, default=Path('train_click_specgan_torch_gpu'))
  parser.add_argument('--epochs', type=int, default=200)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--latent-dim', type=int, default=100)
  parser.add_argument('--model-dim', type=int, default=32)
  parser.add_argument('--kernel-len', type=int, default=5)
  parser.add_argument('--sample-rate', type=int, default=576000)
  parser.add_argument('--n-critic', type=int, default=5)
  parser.add_argument('--gradient-penalty', type=float, default=10.0)
  parser.add_argument('--learning-rate', type=float, default=1e-4)
  parser.add_argument('--griffin-lim-iterations', type=int, default=16)
  parser.add_argument('--num-workers', type=int, default=0)
  parser.add_argument('--seed', type=int, default=369)
  parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
  parser.add_argument('--resume', type=Path)
  parser.add_argument('--checkpoint-every', type=int, default=1)
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
  moment_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
  print('Loaded {} click waveforms from {}.'.format(len(dataset), args.data_dir.resolve()))
  mean, std = compute_spectrogram_moments(moment_loader, device)
  print('Log-magnitude moments: mean={:.6f}, std={:.6f}.'.format(mean.item(), std.item()))

  generator = SpecGANGenerator(args.latent_dim, args.model_dim, args.kernel_len).to(device)
  discriminator = SpecGANDiscriminator(args.model_dim, args.kernel_len).to(device)
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
    mean = torch.tensor(checkpoint['spectrogram_mean'], device=device)
    std = torch.tensor(checkpoint['spectrogram_std'], device=device)
    start_epoch = checkpoint['epoch'] + 1
    print('Resumed from {} at epoch {}.'.format(args.resume, start_epoch))

  args.output_dir.mkdir(parents=True, exist_ok=True)
  fixed_noise = torch.randn(16, args.latent_dim, device=device)
  global_step = 0
  for epoch in range(start_epoch, args.epochs + 1):
    generator.train()
    discriminator.train()
    generator_loss = None
    for batch_index, real_waveforms in enumerate(loader, 1):
      real_waveforms = real_waveforms.to(device, non_blocking=True)
      real_spectrograms = normalized_spectrogram(real_waveforms, mean, std)
      batch_size = real_spectrograms.size(0)

      fake_spectrograms = generator(torch.randn(batch_size, args.latent_dim, device=device)).detach()
      discriminator_optimizer.zero_grad()
      discriminator_loss = (
          discriminator(fake_spectrograms).mean() - discriminator(real_spectrograms).mean()
          + args.gradient_penalty * gradient_penalty(discriminator, real_spectrograms, fake_spectrograms))
      discriminator_loss.backward()
      discriminator_optimizer.step()

      if global_step % args.n_critic == 0:
        generator_optimizer.zero_grad()
        generated_spectrograms = generator(torch.randn(batch_size, args.latent_dim, device=device))
        generator_loss = -discriminator(generated_spectrograms).mean()
        generator_loss.backward()
        generator_optimizer.step()
      global_step += 1
      if batch_index % 50 == 0 or batch_index == len(loader):
        generator_value = float('nan') if generator_loss is None else generator_loss.item()
        print('Epoch {}/{} | batch {}/{} | D={:.5f} | G={:.5f}'.format(
            epoch, args.epochs, batch_index, len(loader), discriminator_loss.item(), generator_value), flush=True)

    if epoch % args.checkpoint_every == 0:
      save_checkpoint(
          args.output_dir / 'specgan_epoch_{:04d}.pt'.format(epoch), epoch,
          generator, discriminator, generator_optimizer, discriminator_optimizer, args, mean, std)
      save_preview(generator, fixed_noise, args.output_dir, epoch, args.sample_rate, mean, std,
                   args.griffin_lim_iterations)


if __name__ == '__main__':
  main()
