"""PyTorch SpecGAN modules and short-waveform spectrogram inversion helpers."""

import torch
from torch import nn
from torch.nn import functional as functional


N_FFT = 32
HOP_LENGTH = 8
FREQUENCY_BINS = N_FFT // 2 + 1
TIME_BINS = 13
SPECTROGRAM_SIZE = 32
LOG_EPS = 1e-6
CLIP_NSTD = 3.0


def weights_init(module):
  if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
    nn.init.normal_(module.weight, mean=0.0, std=0.02)
    if module.bias is not None:
      nn.init.zeros_(module.bias)


def waveform_to_log_magnitude(waveforms):
  """Transform (N, 1, 128) waveform batches to (N, 17, 13) log magnitudes."""
  window = torch.ones(N_FFT, dtype=waveforms.dtype, device=waveforms.device)
  spectra = torch.stft(
      waveforms[:, 0], n_fft=N_FFT, hop_length=HOP_LENGTH,
      center=False, return_complex=True, window=window)
  return torch.log(spectra.abs() + LOG_EPS)


def normalized_spectrogram(waveforms, mean, std):
  """Create a clipped (N, 1, 32, 32) SpecGAN input image."""
  log_magnitude = waveform_to_log_magnitude(waveforms)
  normalized = ((log_magnitude - mean) / (CLIP_NSTD * std)).clamp(-1.0, 1.0)
  image = torch.full(
      (waveforms.size(0), 1, SPECTROGRAM_SIZE, SPECTROGRAM_SIZE), -1.0,
      dtype=waveforms.dtype, device=waveforms.device)
  image[:, 0, :FREQUENCY_BINS, :TIME_BINS] = normalized
  return image


def invert_spectrogram(images, mean, std, iterations=16):
  """Recover 128-sample waveforms from normalized generated spectrograms."""
  normalized = images[:, 0, :FREQUENCY_BINS, :TIME_BINS]
  magnitude = torch.exp(normalized * (CLIP_NSTD * std) + mean)
  window = torch.ones(N_FFT, dtype=images.dtype, device=images.device)
  phase = torch.rand_like(magnitude) * (2.0 * torch.pi)
  spectrum = torch.polar(magnitude, phase)
  for _ in range(iterations):
    waveforms = torch.istft(
        spectrum, n_fft=N_FFT, hop_length=HOP_LENGTH, center=False, length=128, window=window)
    estimate = torch.stft(
        waveforms, n_fft=N_FFT, hop_length=HOP_LENGTH, center=False, return_complex=True, window=window)
    phase = estimate / estimate.abs().clamp_min(1e-8)
    spectrum = magnitude * phase
  waveforms = torch.istft(
      spectrum, n_fft=N_FFT, hop_length=HOP_LENGTH, center=False, length=128, window=window)
  return waveforms.unsqueeze(1)


class SpecGANGenerator(nn.Module):
  """Generate normalized 32x32 magnitude-spectrogram images from latent vectors."""

  def __init__(self, latent_dim=100, dim=32, kernel_len=5):
    super().__init__()
    self.dim = dim
    self.project = nn.Sequential(
        nn.Linear(latent_dim, 4 * 4 * dim * 8),
        nn.ReLU(inplace=True),
    )
    self.upconv_0 = nn.Sequential(
        nn.ConvTranspose2d(dim * 8, dim * 4, kernel_len, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
    )
    self.upconv_1 = nn.Sequential(
        nn.ConvTranspose2d(dim * 4, dim * 2, kernel_len, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
    )
    self.upconv_2 = nn.ConvTranspose2d(dim * 2, 1, kernel_len, stride=2, padding=2, output_padding=1)

  def forward(self, latent):
    output = self.project(latent).view(latent.size(0), self.dim * 8, 4, 4)
    output = self.upconv_0(output)
    output = self.upconv_1(output)
    return torch.tanh(self.upconv_2(output))


class SpecGANDiscriminator(nn.Module):
  """Score normalized 32x32 magnitude-spectrogram images for WGAN-GP."""

  def __init__(self, dim=32, kernel_len=5):
    super().__init__()
    self.downconv_0 = nn.Conv2d(1, dim, kernel_len, stride=2, padding=2)
    self.downconv_1 = nn.Conv2d(dim, dim * 2, kernel_len, stride=2, padding=2)
    self.downconv_2 = nn.Conv2d(dim * 2, dim * 4, kernel_len, stride=2, padding=2)
    self.output = nn.Linear(dim * 4 * 4 * 4, 1)

  def forward(self, spectrograms):
    output = functional.leaky_relu(self.downconv_0(spectrograms), negative_slope=0.2)
    output = functional.leaky_relu(self.downconv_1(output), negative_slope=0.2)
    output = functional.leaky_relu(self.downconv_2(output), negative_slope=0.2)
    return self.output(output.flatten(start_dim=1)).squeeze(1)
