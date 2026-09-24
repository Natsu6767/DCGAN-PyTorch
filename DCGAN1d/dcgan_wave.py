import torch
import torch.nn as nn


def weights_init(module):
    """Initialize convolution and batch-normalization layers as in DCGAN."""
    class_name = module.__class__.__name__.lower()
    if "conv" in class_name and hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "batchnorm" in class_name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


class Generator1D(nn.Module):
    """Generate normalized waveforms with shape (batch, channels, 128)."""

    def __init__(self, latent_dim=100, channels=1, ngf=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, ngf * 4 * 16),
            nn.ReLU(True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose1d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(ngf, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), -1, 16)
        return self.net(x)


class Discriminator1D(nn.Module):
    """Return LSGAN scores for normalized waveforms with shape (batch, channels, 128)."""

    def __init__(self, channels=1, ndf=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(ndf * 4, 1, kernel_size=16, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(-1)
