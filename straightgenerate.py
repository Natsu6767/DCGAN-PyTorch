import numpy as np
import torch
import matplotlib.pyplot as plt

from DCGAN1d.dcgan_wave import Generator1D

checkpoint_path = r"D:\Project_Github\DCGAN-PyTorch\DCGAN1d\model\baseline_epoch15_lsgan_weak_discriminator\model_wave_epoch_75.pth"

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
params = checkpoint["params"]

generator = Generator1D(
    latent_dim=params["nz"],
    channels=params["channels"],
    ngf=params["ngf"],
)
generator.load_state_dict(checkpoint["generator"])
generator.eval()

with torch.no_grad():
    noise = torch.randn(16, params["nz"])
    generated = generator(noise).cpu().numpy()  # shape: (16, 1, 128)

fig, axes = plt.subplots(4, 4, figsize=(12, 10), constrained_layout=True)
for index, axis in enumerate(axes.flat):
    axis.plot(generated[index, 0])
    axis.set_title(f"Generated waveform {index + 1}")
    axis.set_xlabel("Sample")
    axis.set_ylabel("Amplitude")
    axis.set_ylim(-1.0, 1.0)

figure_path = r"D:\Project_Github\DCGAN-PyTorch\images\new_generated_waveforms.png"
fig.savefig(figure_path, dpi=300)
plt.show()
