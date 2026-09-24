# 加入相关性损失函数、标准差损失函数、eval模式
# 最优位置停在了18轮
from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from DCGAN1d.dcgan_wave import Discriminator1D, Generator1D, weights_init
from DCGAN1d.utils_wave import get_wav_dataloader


seed = 369
random.seed(seed)
torch.manual_seed(seed)

params = {
    "bsize": 256,
    "channels": 1,
    "signal_length": 128,
    "nz": 100,
    "ngf": 64,
    "ndf": 32,
    "nepochs": 18,
    "lr_d": 0.00001,
    "lr_g": 0.00005,
    "beta1": 0.5,
    "lambda_lag1": 5.0,
    "lambda_std": 50.0,
    "run_name": "correlation_optimized",
    "save_epoch": 1,
    "data_root": "data/wav",
}


def lag1_autocorrelation(waveforms):
    """Return the batch lag-1 autocorrelation used for waveform realism checks."""
    numerator = (waveforms[:, :, :-1] * waveforms[:, :, 1:]).mean()
    denominator = waveforms.square().mean().clamp_min(1e-8)
    return numerator / denominator


if params["signal_length"] != 128:
    raise ValueError("The current Generator1D and Discriminator1D are designed for signal_length=128.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device}.")
dataloader = get_wav_dataloader(params)
print(f"Loaded {len(dataloader.dataset)} waveform segments.")

net_g = Generator1D(params["nz"], params["channels"], params["ngf"]).to(device)
net_d = Discriminator1D(params["channels"], params["ndf"]).to(device)
net_g.apply(weights_init)
net_d.apply(weights_init)

criterion = nn.BCELoss()
optimizer_d = optim.Adam(net_d.parameters(), lr=params["lr_d"], betas=(params["beta1"], 0.999))
optimizer_g = optim.Adam(net_g.parameters(), lr=params["lr_g"], betas=(params["beta1"], 0.999))
fixed_noise = torch.randn(16, params["nz"], device=device)

g_losses, d_losses = [], []
generated_history = []
lag1_losses = []
run_dir = Path("model") / params["run_name"]
run_dir.mkdir(parents=True, exist_ok=True)
print("Starting 1D waveform GAN training...")

for epoch in range(params["nepochs"]):
    for batch_index, real_waveforms in enumerate(dataloader):
        real_waveforms = real_waveforms.to(device)
        batch_size = real_waveforms.size(0)

        net_d.zero_grad()
        real_labels = torch.ones(batch_size, dtype=torch.float32, device=device)
        output_real = net_d(real_waveforms)
        loss_d_real = criterion(output_real, real_labels)
        loss_d_real.backward()

        noise = torch.randn(batch_size, params["nz"], device=device)
        fake_waveforms = net_g(noise)
        fake_labels = torch.zeros(batch_size, dtype=torch.float32, device=device)
        output_fake = net_d(fake_waveforms.detach())
        loss_d_fake = criterion(output_fake, fake_labels)
        loss_d_fake.backward()
        loss_d = loss_d_real + loss_d_fake
        optimizer_d.step()

        net_g.zero_grad()
        output_for_g = net_d(fake_waveforms)
        loss_g_adv = criterion(output_for_g, real_labels)
        real_lag1 = lag1_autocorrelation(real_waveforms).detach()
        real_std = real_waveforms.std().detach()
        net_g.eval()
        eval_fake_waveforms = net_g(noise)
        fake_lag1 = lag1_autocorrelation(eval_fake_waveforms)
        fake_std = eval_fake_waveforms.std()
        loss_lag1 = (fake_lag1 - real_lag1).square()
        loss_std = (fake_std - real_std).square()
        net_g.train()
        loss_g = (
            loss_g_adv
            + params["lambda_lag1"] * loss_lag1
            + params["lambda_std"] * loss_std
        )
        loss_g.backward()
        optimizer_g.step()

        d_losses.append(loss_d.item())
        g_losses.append(loss_g.item())
        lag1_losses.append(loss_lag1.item())

        if batch_index % 50 == 0:
            print(
                f"[{epoch + 1}/{params['nepochs']}][{batch_index}/{len(dataloader)}] "
                f"Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f} "
                f"Lag1(real/fake): {real_lag1.item():.4f}/{fake_lag1.item():.4f} "
                f"Std(real/fake): {real_std.item():.4f}/{fake_std.item():.4f} "
                f"D(x): {output_real.mean().item():.4f} "
                f"D(G(z)): {output_fake.mean().item():.4f}/{output_for_g.mean().item():.4f}"
            )

    with torch.no_grad():
        generated_history.append(net_g(fixed_noise).cpu())

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

plt.figure(figsize=(10, 6))
for index, waveform in enumerate(generated_history[-1][:8]):
    plt.subplot(4, 2, index + 1)
    plt.plot(waveform[0].numpy())
    plt.ylim(-1.05, 1.05)
    plt.title(f"Generated waveform {index + 1}")
plt.tight_layout()
plt.show()
