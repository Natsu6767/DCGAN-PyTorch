"""PyTorch WaveGAN modules for 128-sample, single-channel pulse waveforms."""

import torch
from torch import nn
from torch.nn import functional as functional


def weights_init(module):
  """Initialize convolutional and linear layers with the DCGAN convention."""
  if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
    nn.init.normal_(module.weight, mean=0.0, std=0.02)
    if module.bias is not None:
      nn.init.zeros_(module.bias)


def phase_shuffle(inputs, radius):
  """Apply the WaveGAN phase-shuffle augmentation along the temporal axis."""
  if radius <= 0:
    return inputs

  phase = int(torch.randint(-radius, radius + 1, (), device=inputs.device))
  if phase == 0:
    return inputs
  if phase > 0:
    padded = functional.pad(inputs, (phase, 0), mode='reflect')
    return padded[:, :, :inputs.size(2)]
  padded = functional.pad(inputs, (0, -phase), mode='reflect')
  return padded[:, :, -phase:-phase + inputs.size(2)]


class WaveGANGenerator(nn.Module):
  """Generate 128-point normalized mono waveforms from latent vectors."""

  def __init__(self, latent_dim=100, channels=1, dim=64, kernel_len=25):
    super().__init__()
    self.dim = dim
    self.project = nn.Sequential(
        nn.Linear(latent_dim, 16 * dim * 4),
        nn.ReLU(inplace=True),
    )
    # These parameters reproduce TensorFlow SAME transpose-convolution lengths:
    # 16 -> 64 (stride 4), then 64 -> 128 (stride 2).
    self.upconv_0 = nn.Sequential(
        nn.ConvTranspose1d(
            dim * 4, dim * 2, kernel_size=kernel_len, stride=4,
            padding=11, output_padding=1),
        nn.ReLU(inplace=True),
    )
    self.upconv_1 = nn.ConvTranspose1d(
        dim * 2, channels, kernel_size=kernel_len, stride=2,
        padding=12, output_padding=1)

  def forward(self, latent):
    output = self.project(latent)
    output = output.view(latent.size(0), self.dim * 4, 16)
    output = self.upconv_0(output)
    return torch.tanh(self.upconv_1(output))


class WaveGANDiscriminator(nn.Module):
  """Score 128-point waveform batches for Wasserstein-GP training."""

  def __init__(self, channels=1, dim=64, kernel_len=25, phase_shuffle_radius=2):
    super().__init__()
    self.phase_shuffle_radius = phase_shuffle_radius
    self.downconv_0 = nn.Conv1d(
        channels, dim, kernel_size=kernel_len, stride=4, padding=11)
    self.downconv_1 = nn.Conv1d(
        dim, dim * 2, kernel_size=kernel_len, stride=4, padding=11)
    self.output = nn.Linear(dim * 2 * 8, 1)

  def forward(self, waveforms):
    output = functional.leaky_relu(self.downconv_0(waveforms), negative_slope=0.2)
    output = phase_shuffle(output, self.phase_shuffle_radius)
    output = functional.leaky_relu(self.downconv_1(output), negative_slope=0.2)
    return self.output(output.flatten(start_dim=1)).squeeze(1)
