"""Train a 128-sample WaveGAN with PyTorch and the bundled click WAV files."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
from scipy.io import wavfile
import torch
from torch import autograd
from torch.nn import functional as functional
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from pytorch_wavegan import WaveGANDiscriminator, WaveGANGenerator, weights_init


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def peak_loss(real_audio, fake_audio):
  """Return the L1 distance between per-waveform absolute peak amplitudes."""
  real_peaks = torch.max(torch.abs(real_audio), dim=-1).values
  fake_peaks = torch.max(torch.abs(fake_audio), dim=-1).values
  return functional.l1_loss(fake_peaks, real_peaks)


def decode_waveform(path, expected_sample_rate):
  """Read, mono-mix, normalize, and right-pad one waveform to 128 samples."""
  sample_rate, waveform = wavfile.read(str(path))
  if sample_rate != expected_sample_rate:
    raise ValueError('{} has {} Hz; expected {} Hz.'.format(path, sample_rate, expected_sample_rate))
  if waveform.ndim == 2:
    waveform = waveform.mean(axis=1)
  if waveform.ndim != 1:
    raise ValueError('{} has unsupported waveform shape {}.'.format(path, waveform.shape))

  if np.issubdtype(waveform.dtype, np.integer):
    waveform = waveform.astype(np.float32) / float(abs(np.iinfo(waveform.dtype).min))
  elif np.issubdtype(waveform.dtype, np.floating):
    waveform = waveform.astype(np.float32, copy=True)
  else:
    raise ValueError('{} has unsupported sample type {}.'.format(path, waveform.dtype))

  maximum = np.max(np.abs(waveform))
  if maximum > 0:
    waveform /= maximum
  if waveform.size > 128:
    raise ValueError('{} has {} samples; this 128-point model requires shorter inputs.'.format(path, waveform.size))
  return np.pad(waveform, (0, 128 - waveform.size)).astype(np.float32, copy=False)


class ClickWaveformDataset(Dataset):
  """One normalized and zero-padded 128-point example per WAV file."""

  def __init__(self, data_dir, sample_rate):
    self.paths = sorted(path for path in Path(data_dir).rglob('*') if path.suffix.lower() == '.wav')
    if not self.paths:
      raise FileNotFoundError('No WAV files found under {}.'.format(data_dir))
    self.sample_rate = sample_rate

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, index):
    waveform = decode_waveform(self.paths[index], self.sample_rate)
    return torch.from_numpy(waveform).unsqueeze(0)


def gradient_penalty(discriminator, real_waveforms, fake_waveforms):
  batch_size = real_waveforms.size(0)
  interpolation = torch.rand(batch_size, 1, 1, device=real_waveforms.device)
  mixed = interpolation * real_waveforms + (1.0 - interpolation) * fake_waveforms
  mixed.requires_grad_(True)
  scores = discriminator(mixed)
  gradients = autograd.grad(
      outputs=scores,
      inputs=mixed,
      grad_outputs=torch.ones_like(scores),
      create_graph=True,
      retain_graph=True,
      only_inputs=True)[0]
  slopes = gradients.reshape(batch_size, -1).norm(2, dim=1)
  return ((slopes - 1.0) ** 2).mean()


def save_preview(generator, fixed_noise, output_dir, epoch, sample_rate):
  generator.eval()
  with torch.no_grad():
    waveform = generator(fixed_noise[:1])[0, 0].detach().cpu().numpy()
  generator.train()
  encoded = np.clip(waveform * 32767.0, -32767.0, 32767.0).astype(np.int16)
  wavfile.write(str(output_dir / 'preview_epoch_{:04d}.wav'.format(epoch)), sample_rate, encoded)


def save_checkpoint(path, epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, args):
  torch.save({
      'epoch': epoch,
      'generator': generator.state_dict(),
      'discriminator': discriminator.state_dict(),
      'generator_optimizer': generator_optimizer.state_dict(),
      'discriminator_optimizer': discriminator_optimizer.state_dict(),
      'args': vars(args),
  }, str(path))


def write_epoch_log(log_file, epoch, alpha, discriminator_losses,
                    adversarial_losses, peak_losses, total_generator_losses):
  """Append one machine-readable training summary for an epoch."""
  def mean_or_none(values):
    return None if not values else float(np.mean(values))

  record = {
      'epoch': epoch,
      'peak_loss_weight': alpha,
      'mean_discriminator_loss': mean_or_none(discriminator_losses),
      'mean_generator_adversarial_loss': mean_or_none(adversarial_losses),
      'mean_generator_peak_loss': mean_or_none(peak_losses),
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
  parser.add_argument('--peak-loss-weight', type=float, default=0.0,
                      help='Alpha multiplying per-waveform peak-amplitude L1 loss in the generator objective.')
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
  if args.device == 'auto':
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
  else:
    device_name = args.device
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
          'peak_loss_weight': args.peak_loss_weight,
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
    epoch_peak_losses = []
    epoch_generator_total_losses = []
    for batch_index, real_waveforms in enumerate(loader, 1):
      real_waveforms = real_waveforms.to(device, non_blocking=True)
      batch_size = real_waveforms.size(0)

      noise = torch.randn(batch_size, args.latent_dim, device=device)
      fake_waveforms = generator(noise).detach()
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
        batch_peak_loss = peak_loss(real_waveforms, generated_waveforms)
        generator_loss = adversarial_loss + args.peak_loss_weight * batch_peak_loss
        generator_loss.backward()
        generator_optimizer.step()
        epoch_adversarial_losses.append(adversarial_loss.item())
        epoch_peak_losses.append(batch_peak_loss.item())
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
          args.log_file, epoch, args.peak_loss_weight, epoch_discriminator_losses,
          epoch_adversarial_losses, epoch_peak_losses, epoch_generator_total_losses)

    if epoch % args.checkpoint_every == 0:
      save_checkpoint(
          args.output_dir / 'wavegan_epoch_{:04d}.pt'.format(epoch), epoch,
          generator, discriminator, generator_optimizer, discriminator_optimizer, args)
      save_preview(generator, fixed_noise, args.output_dir, epoch, args.sample_rate)


if __name__ == '__main__':
  main()
