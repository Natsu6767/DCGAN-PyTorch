"""Pure-BCE DCGAN baseline for two-channel waveform and spectrum features.

This preserves the training parameters and DCGAN modules used by
train_wave_baseline_epoch15.py.  The required input files are (2, 128) .npy
arrays created under data/threewav by extract_three_channel_wav_features.py.
"""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from 一维DCGAN.dcgan_wave import Discriminator1D, Generator1D, weights_init


class TwoChannelNpyDataset(Dataset):
    """Load one (waveform, magnitude spectrum) feature array per sample."""

    def __init__(self, root: str | Path, channels: int, signal_length: int):
        root_path = Path(root)
        self.feature_paths = sorted(path for path in root_path.rglob("*.npy") if path.is_file())
        if not self.feature_paths:
            raise FileNotFoundError(f"No .npy feature files found under {root_path.resolve()}.")

        expected_shape = (channels, signal_length)
        first_shape = np.load(self.feature_paths[0], mmap_mode="r").shape
        if first_shape != expected_shape:
            raise ValueError(
                f"Expected feature shape {expected_shape}, but {self.feature_paths[0]} has {first_shape}."
            )
        self.expected_shape = expected_shape

    def __len__(self) -> int:
        return len(self.feature_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.feature_paths[index]
        features = np.load(path).astype(np.float32, copy=False)
        if features.shape != self.expected_shape:
            raise ValueError(f"Expected {self.expected_shape}, but {path} has {features.shape}.")
        return torch.from_numpy(features)


def get_npy_dataloader(params: dict) -> DataLoader:
    dataset = TwoChannelNpyDataset(
        root=params["data_root"],
        channels=params["channels"],
        signal_length=params["signal_length"],
    )
    return DataLoader(dataset, batch_size=params["bsize"], shuffle=True, num_workers=0)


def show_batch_progress(epoch: int, total_epochs: int, batch: int, total_batches: int) -> None:
    """Show overall training progress without changing the training procedure."""
    completed = (epoch - 1) * total_batches + batch
    total = total_epochs * total_batches
    width = 30
    filled = int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\rTraining [{bar}] {completed}/{total} batches", end="", flush=True)


seed = 369
random.seed(seed)
torch.manual_seed(seed)

params = {
    "bsize": 256,
    "channels": 2,
    "signal_length": 128,
    "nz": 100,
    "ngf": 64,
    "ndf": 32,
    "nepochs": 15,
    "lr_d": 0.00001,
    "lr_g": 0.00005,
    "beta1": 0.5,
    "save_epoch": 1,
    "data_root": "data/threewav",
    "run_name": "two_channel_baseline_epoch15_bce_only",
}

if params["signal_length"] != 128:
    raise ValueError("Generator1D and Discriminator1D require signal_length=128.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataloader = get_npy_dataloader(params)

net_g = Generator1D(params["nz"], params["channels"], params["ngf"]).to(device)
net_d = Discriminator1D(params["channels"], params["ndf"]).to(device)
net_g.apply(weights_init)
net_d.apply(weights_init)

criterion = nn.BCELoss()
optimizer_d = optim.Adam(net_d.parameters(), lr=params["lr_d"], betas=(params["beta1"], 0.999))
optimizer_g = optim.Adam(net_g.parameters(), lr=params["lr_g"], betas=(params["beta1"], 0.999))
fixed_noise = torch.randn(16, params["nz"], device=device)

run_dir = Path("model") / params["run_name"]
run_dir.mkdir(parents=True, exist_ok=True)
g_losses, d_losses = [], []

print(f"Using {device}; loaded {len(dataloader.dataset)} two-channel feature samples.")
print("Starting BCE-only two-channel 1D DCGAN baseline training...")

for epoch in range(params["nepochs"]):
    for batch_index, real_waveforms in enumerate(dataloader):
        real_waveforms = real_waveforms.to(device)
        batch_size = real_waveforms.size(0)
        real_labels = torch.ones(batch_size, dtype=torch.float32, device=device)
        fake_labels = torch.zeros(batch_size, dtype=torch.float32, device=device)

        net_d.zero_grad()
        output_real = net_d(real_waveforms)
        loss_d_real = criterion(output_real, real_labels)
        loss_d_real.backward()

        noise = torch.randn(batch_size, params["nz"], device=device)
        fake_waveforms = net_g(noise)
        output_fake = net_d(fake_waveforms.detach())
        loss_d_fake = criterion(output_fake, fake_labels)
        loss_d_fake.backward()
        loss_d = loss_d_real + loss_d_fake
        optimizer_d.step()

        net_g.zero_grad()
        output_for_g = net_d(fake_waveforms)
        loss_g = criterion(output_for_g, real_labels)
        loss_g.backward()
        optimizer_g.step()

        d_losses.append(loss_d.item())
        g_losses.append(loss_g.item())

        if batch_index % 50 == 0:
            print(
                f"\n[{epoch + 1}/{params['nepochs']}][{batch_index}/{len(dataloader)}] "
                f"Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f} "
                f"D(x): {output_real.mean().item():.4f} "
                f"D(G(z)): {output_fake.mean().item():.4f}/{output_for_g.mean().item():.4f}"
            )
        show_batch_progress(epoch + 1, params["nepochs"], batch_index + 1, len(dataloader))

    print()

    if (epoch + 1) % params["save_epoch"] == 0:
        torch.save(
            {"generator": net_g.state_dict(), "discriminator": net_d.state_dict(), "params": params},
            run_dir / f"model_wave_epoch_{epoch + 1}.pth",
        )

torch.save(
    {"generator": net_g.state_dict(), "discriminator": net_d.state_dict(), "params": params},
    run_dir / "model_wave_final.pth",
)

plt.figure(figsize=(9, 4))
plt.plot(d_losses, label="D")
plt.plot(g_losses, label="G")
plt.xlabel("Iteration")
plt.ylabel("Binary cross-entropy loss")
plt.legend()
plt.tight_layout()
plt.show()

net_g.eval()
with torch.no_grad():
    generated_features = net_g(fixed_noise).cpu().numpy()

figure, axes = plt.subplots(4, 4, figsize=(12, 9), sharex=True, sharey=True)
figure.suptitle("16 generated waveforms (channel 0)")
for index, axis in enumerate(axes.flat):
    axis.plot(generated_features[index, 0], linewidth=0.8)
    axis.set_title(f"Waveform {index + 1}")
    axis.set_xlim(0, params["signal_length"] - 1)
    axis.set_ylim(-1.05, 1.05)
figure.tight_layout()
plt.show()
