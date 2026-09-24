"""Train a 1D waveform DCGAN and evaluate it on a held-out waveform set."""

from __future__ import annotations

import csv
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.stats import wasserstein_distance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dcgan_wave import Discriminator1D, Generator1D, weights_init
from utils_wave import WavSegmentDataset


seed = 369
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
print(f"Random seed: {seed}")

params = {
    "bsize": 256,
    "channels": 1,
    "signal_length": 128,
    "nz": 100,
    "ngf": 64,
    "ndf": 32,
    "nepochs": 100,
    "lr_d": 1e-5,
    "lr_g": 1e-4,
    "beta1": 0.5,
    "lambda_peak": 1.0,
    "peak_to_peak_quantile": 0.99,
    "train_fraction": 0.80,
    "evaluation_interval": 5,
    "data_root": "data/wav",
    "run_name": "baseline_epoch15_lsgan_weak_discriminator",
}

if params["signal_length"] != 128:
    raise ValueError("Generator1D and Discriminator1D require signal_length=128.")


def get_sample_rate_khz(wav_paths: list[Path]) -> float:
    """Return the common WAV sample rate in kHz, rejecting mixed-rate datasets."""
    sample_rates = {wavfile.read(path, mmap=True)[0] for path in wav_paths}
    if len(sample_rates) != 1:
        raise ValueError(
            "All WAV files must have one sample rate to compare peak frequency and PSD. "
            f"Found: {sorted(sample_rates)} Hz."
        )
    return sample_rates.pop() / 1000.0


def waveform_psd(waveforms: np.ndarray, sample_rate_khz: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute one-sided periodograms in power/kHz for waveforms of shape (N, L)."""
    signal_length = waveforms.shape[-1]
    spectra = np.fft.rfft(waveforms, axis=-1)
    psd = np.abs(spectra) ** 2 / (sample_rate_khz * signal_length)
    if signal_length % 2 == 0:
        psd[:, 1:-1] *= 2.0
    else:
        psd[:, 1:] *= 2.0
    frequencies_khz = np.fft.rfftfreq(signal_length, d=1.0 / sample_rate_khz)
    return frequencies_khz, psd


def pulse_width_fwhm(waveform: np.ndarray) -> float:
    """Measure absolute-amplitude FWHM in samples around the maximum-amplitude pulse."""
    envelope = np.abs(waveform)
    peak_index = int(np.argmax(envelope))
    peak_value = float(envelope[peak_index])
    if peak_value == 0.0:
        return 0.0

    half_height = peak_value / 2.0
    left = peak_index
    while left > 0 and envelope[left] >= half_height:
        left -= 1
    if left == 0 and envelope[left] >= half_height:
        left_crossing = 0.0
    else:
        left_crossing = left + (half_height - envelope[left]) / (envelope[left + 1] - envelope[left])

    right = peak_index
    last_index = envelope.size - 1
    while right < last_index and envelope[right] >= half_height:
        right += 1
    if right == last_index and envelope[right] >= half_height:
        right_crossing = float(last_index)
    else:
        right_crossing = (right - 1) + (half_height - envelope[right - 1]) / (
            envelope[right] - envelope[right - 1]
        )
    return float(right_crossing - left_crossing)


def extract_statistics(waveforms: np.ndarray, sample_rate_khz: float) -> dict[str, np.ndarray]:
    """Extract dominant frequency, FWHM, and mean PSD from a waveform batch."""
    frequencies_khz, psd = waveform_psd(waveforms, sample_rate_khz)
    # Ignore the DC component: an offset is not a waveform peak frequency.
    peak_indices = np.argmax(psd[:, 1:], axis=1) + 1
    return {
        "frequencies_khz": frequencies_khz,
        "peak_frequencies_khz": frequencies_khz[peak_indices],
        "pulse_widths_samples": np.asarray([pulse_width_fwhm(waveform) for waveform in waveforms]),
        "mean_psd": psd.mean(axis=0),
    }


def dataset_statistics(dataset, batch_size: int, sample_rate_khz: float) -> dict[str, np.ndarray]:
    """Evaluate a dataset without retaining every waveform in memory."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    peak_frequencies, pulse_widths = [], []
    psd_sum = None
    sample_count = 0
    frequencies_khz = None
    for waveforms in loader:
        statistics = extract_statistics(waveforms[:, 0].numpy(), sample_rate_khz)
        peak_frequencies.append(statistics["peak_frequencies_khz"])
        pulse_widths.append(statistics["pulse_widths_samples"])
        weighted_psd = statistics["mean_psd"] * waveforms.size(0)
        psd_sum = weighted_psd if psd_sum is None else psd_sum + weighted_psd
        sample_count += waveforms.size(0)
        frequencies_khz = statistics["frequencies_khz"]
    return {
        "frequencies_khz": frequencies_khz,
        "peak_frequencies_khz": np.concatenate(peak_frequencies),
        "pulse_widths_samples": np.concatenate(pulse_widths),
        "mean_psd": psd_sum / sample_count,
    }


def generated_statistics(
    net_g, count: int, batch_size: int, latent_dim: int, device, sample_rate_khz: float
) -> dict[str, np.ndarray]:
    """Generate exactly ``count`` samples and return their aggregate statistics."""
    peak_frequencies, pulse_widths = [], []
    psd_sum = None
    generated_count = 0
    frequencies_khz = None
    net_g.eval()
    with torch.no_grad():
        while generated_count < count:
            current_batch_size = min(batch_size, count - generated_count)
            noise = torch.randn(current_batch_size, latent_dim, device=device)
            waveforms = net_g(noise).cpu().numpy()[:, 0]
            statistics = extract_statistics(waveforms, sample_rate_khz)
            peak_frequencies.append(statistics["peak_frequencies_khz"])
            pulse_widths.append(statistics["pulse_widths_samples"])
            weighted_psd = statistics["mean_psd"] * current_batch_size
            psd_sum = weighted_psd if psd_sum is None else psd_sum + weighted_psd
            generated_count += current_batch_size
            frequencies_khz = statistics["frequencies_khz"]
    net_g.train()
    return {
        "frequencies_khz": frequencies_khz,
        "peak_frequencies_khz": np.concatenate(peak_frequencies),
        "pulse_widths_samples": np.concatenate(pulse_widths),
        "mean_psd": psd_sum / generated_count,
    }


def save_psd_comparison(test_stats, generated_stats, epoch: int, output_dir: Path) -> None:
    """Save the test-versus-generated mean PSD curve for one epoch."""
    plt.figure(figsize=(8, 4.5))
    plt.semilogy(test_stats["frequencies_khz"], test_stats["mean_psd"], label="Test set")
    plt.semilogy(generated_stats["frequencies_khz"], generated_stats["mean_psd"], label="Generated set")
    plt.title(f"Mean PSD comparison (epoch {epoch})")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("PSD (power/kHz)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / f"mean_psd_comparison_epoch_{epoch:03d}.png", dpi=160)
    plt.close()


def save_metric_plots(metrics: list[dict[str, float]], output_dir: Path) -> None:
    """Save all scalar metric trajectories and test/generated feature comparisons."""
    epochs = [row["epoch"] for row in metrics]
    scalar_metrics = [
        ("generator_loss", "Generator total loss", "Loss"),
        ("generator_adversarial_loss", "Generator adversarial loss", "LSGAN loss"),
        ("peak_to_peak_p99_loss", "Peak-to-peak P99 auxiliary loss", "Amplitude"),
        ("discriminator_loss", "Discriminator loss", "LSGAN loss"),
        ("peak_frequency_wasserstein_khz", "Peak-frequency Wasserstein distance", "Distance (kHz)"),
        ("pulse_width_wasserstein_samples", "Pulse-width Wasserstein distance", "Distance (samples)"),
        ("psd_mean_absolute_error", "Mean PSD absolute error", "PSD (power/kHz)"),
        ("score", "Normalized composite Score", "Score"),
    ]
    for column, title, ylabel in scalar_metrics:
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, [row[column] for row in metrics], marker="o")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / f"{column}_vs_epoch.png", dpi=160)
        plt.close()

    feature_comparisons = [
        ("mean_peak_frequency_khz", "Mean peak frequency", "Frequency (kHz)"),
        ("mean_pulse_width_samples", "Mean pulse width", "Width (samples)"),
    ]
    for column, title, ylabel in feature_comparisons:
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, [row[f"test_{column}"] for row in metrics], label="Test set", linestyle="--")
        plt.plot(epochs, [row[f"generated_{column}"] for row in metrics], label="Generated set", marker="o")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / f"{column}_comparison_vs_epoch.png", dpi=160)
        plt.close()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device}.")

# Deterministic random 80/20 split at the level of 128-point waveform segments.
dataset = WavSegmentDataset(root=params["data_root"], segment_length=params["signal_length"])
sample_rate_khz = get_sample_rate_khz(dataset.wav_paths)
train_size = int(len(dataset) * params["train_fraction"])
test_size = len(dataset) - train_size
split_generator = torch.Generator().manual_seed(seed)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=split_generator)
train_loader = DataLoader(train_dataset, batch_size=params["bsize"], shuffle=True, num_workers=0)
print(
    f"Loaded {len(dataset)} waveform segments: {len(train_dataset)} training, "
    f"{len(test_dataset)} testing. Sample rate: {sample_rate_khz:g} kHz."
)

# The held-out distribution is fixed and reused after every epoch.
test_statistics = dataset_statistics(test_dataset, params["bsize"], sample_rate_khz)

net_g = Generator1D(params["nz"], params["channels"], params["ngf"]).to(device)
net_d = Discriminator1D(params["channels"], params["ndf"]).to(device)
net_g.apply(weights_init)
net_d.apply(weights_init)

criterion = nn.MSELoss()
optimizer_d = optim.Adam(net_d.parameters(), lr=params["lr_d"], betas=(params["beta1"], 0.999))
optimizer_g = optim.Adam(net_g.parameters(), lr=params["lr_g"], betas=(params["beta1"], 0.999))

script_dir = Path(__file__).resolve().parent
model_dir = script_dir / "model" / params["run_name"]
results_dir = script_dir / "results" / params["run_name"]
model_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)
metrics = []
score_reference = None
best_score = float("inf")
best_score_epoch = None

print("Starting LSGAN training loop...")
for epoch in range(params["nepochs"]):
    epoch_g_losses, epoch_g_adversarial_losses, epoch_peak_losses, epoch_d_losses = [], [], [], []
    for batch_index, real_waveforms in enumerate(train_loader):
        real_waveforms = real_waveforms.to(device)
        batch_size = real_waveforms.size(0)
        labels = torch.full((batch_size,), 1.0, dtype=torch.float32, device=device)

        net_d.zero_grad()
        output_real = net_d(real_waveforms)
        loss_d_real = criterion(output_real, labels)
        loss_d_real.backward()
        d_x = output_real.mean().item()

        noise = torch.randn(batch_size, params["nz"], device=device)
        fake_waveforms = net_g(noise)
        labels.fill_(0.0)
        output_fake = net_d(fake_waveforms.detach())
        loss_d_fake = criterion(output_fake, labels)
        loss_d_fake.backward()
        d_g_z1 = output_fake.mean().item()
        loss_d = loss_d_real + loss_d_fake
        optimizer_d.step()

        net_g.zero_grad()
        labels.fill_(1.0)
        output_for_g = net_d(fake_waveforms)
        loss_g_adversarial = criterion(output_for_g, labels)
        # Match high-amplitude transient behavior through the 99th percentile of peak-to-peak values.
        real_peak_to_peak = (real_waveforms.amax(dim=2) - real_waveforms.amin(dim=2)).flatten()
        fake_peak_to_peak = (fake_waveforms.amax(dim=2) - fake_waveforms.amin(dim=2)).flatten()
        real_peak_to_peak_p99 = torch.quantile(real_peak_to_peak, params["peak_to_peak_quantile"])
        fake_peak_to_peak_p99 = torch.quantile(fake_peak_to_peak, params["peak_to_peak_quantile"])
        loss_peak = torch.abs(real_peak_to_peak_p99 - fake_peak_to_peak_p99)
        loss_g = loss_g_adversarial + params["lambda_peak"] * loss_peak
        loss_g.backward()
        d_g_z2 = output_for_g.mean().item()
        optimizer_g.step()

        if batch_index % 50 == 0:
            print(
                f"[{epoch + 1}/{params['nepochs']}][{batch_index}/{len(train_loader)}] "
                f"Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f} "
                f"(adv: {loss_g_adversarial.item():.4f}, p2p P99: {loss_peak.item():.4f}) "
                f"D(x): {d_x:.4f} D(G(z)): {d_g_z1:.4f}/{d_g_z2:.4f} "
                f"fake[min, max]: [{fake_waveforms.min().item():.4f}, {fake_waveforms.max().item():.4f}]"
            )
        epoch_g_losses.append(loss_g.item())
        epoch_g_adversarial_losses.append(loss_g_adversarial.item())
        epoch_peak_losses.append(loss_peak.item())
        epoch_d_losses.append(loss_d.item())

    # Save a checkpoint after every epoch, including optimizer state for resumption.
    torch.save(
        {
            "epoch": epoch + 1,
            "generator": net_g.state_dict(),
            "discriminator": net_d.state_dict(),
            "optimizerG": optimizer_g.state_dict(),
            "optimizerD": optimizer_d.state_dict(),
            "params": params,
        },
        model_dir / f"model_wave_epoch_{epoch + 1}.pth",
    )

    if (epoch + 1) % params["evaluation_interval"] != 0:
        print(
            f"Progress [{'#' * (epoch + 1)}{'.' * (params['nepochs'] - epoch - 1)}] "
            f"{epoch + 1}/{params['nepochs']} | checkpoint saved; "
            f"comparison every {params['evaluation_interval']} epochs"
        )
        continue

    # Generate exactly as many samples as in the test subset and compare every five epochs.
    generated_set_statistics = generated_statistics(
        net_g, len(test_dataset), params["bsize"], params["nz"], device, sample_rate_khz
    )
    peak_wasserstein = wasserstein_distance(
        test_statistics["peak_frequencies_khz"], generated_set_statistics["peak_frequencies_khz"]
    )
    width_wasserstein = wasserstein_distance(
        test_statistics["pulse_widths_samples"], generated_set_statistics["pulse_widths_samples"]
    )
    psd_mae = np.mean(np.abs(test_statistics["mean_psd"] - generated_set_statistics["mean_psd"]))

    if score_reference is None:
        score_reference = {
            "peak_frequency_wasserstein_khz": peak_wasserstein,
            "pulse_width_wasserstein_samples": width_wasserstein,
            "psd_mean_absolute_error": psd_mae,
        }
        if any(value <= 0.0 for value in score_reference.values()):
            raise ValueError("The epoch-5 Score reference contains zero and cannot be used as a divisor.")

    score = (
        peak_wasserstein / score_reference["peak_frequency_wasserstein_khz"]
        + width_wasserstein / score_reference["pulse_width_wasserstein_samples"]
        + psd_mae / score_reference["psd_mean_absolute_error"]
    )
    metric_row = {
        "epoch": epoch + 1,
        "generator_loss": float(np.mean(epoch_g_losses)),
        "generator_adversarial_loss": float(np.mean(epoch_g_adversarial_losses)),
        "peak_to_peak_p99_loss": float(np.mean(epoch_peak_losses)),
        "discriminator_loss": float(np.mean(epoch_d_losses)),
        "test_mean_peak_frequency_khz": float(np.mean(test_statistics["peak_frequencies_khz"])),
        "generated_mean_peak_frequency_khz": float(np.mean(generated_set_statistics["peak_frequencies_khz"])),
        "peak_frequency_wasserstein_khz": float(peak_wasserstein),
        "test_mean_pulse_width_samples": float(np.mean(test_statistics["pulse_widths_samples"])),
        "generated_mean_pulse_width_samples": float(np.mean(generated_set_statistics["pulse_widths_samples"])),
        "pulse_width_wasserstein_samples": float(width_wasserstein),
        "psd_mean_absolute_error": float(psd_mae),
        "score": float(score),
    }
    metrics.append(metric_row)
    save_psd_comparison(test_statistics, generated_set_statistics, epoch + 1, results_dir)

    if score < best_score:
        best_score = score
        best_score_epoch = epoch + 1
        torch.save(
            {
                "epoch": epoch + 1,
                "score": score,
                "score_reference": score_reference,
                "generator": net_g.state_dict(),
                "discriminator": net_d.state_dict(),
                "optimizerG": optimizer_g.state_dict(),
                "optimizerD": optimizer_d.state_dict(),
                "params": params,
            },
            model_dir / "model_wave_best_score.pth",
        )

    print(
        f"Progress [{'#' * (epoch + 1)}{'.' * (params['nepochs'] - epoch - 1)}] "
        f"{epoch + 1}/{params['nepochs']} | peak W={peak_wasserstein:.5f} kHz | "
        f"width W={width_wasserstein:.5f} samples | PSD MAE={psd_mae:.5e} | Score={score:.5f}"
    )

metrics_csv_path = results_dir / "epoch_metrics.csv"
with metrics_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=list(metrics[0]))
    writer.writeheader()
    writer.writerows(metrics)
save_metric_plots(metrics, results_dir)

torch.save(
    {
        "epoch": params["nepochs"],
        "generator": net_g.state_dict(),
        "discriminator": net_d.state_dict(),
        "optimizerG": optimizer_g.state_dict(),
        "optimizerD": optimizer_d.state_dict(),
        "params": params,
    },
    model_dir / "model_wave_final.pth",
)
print(f"Saved checkpoints to {model_dir.resolve()}")
print(f"Saved metrics and figures to {results_dir.resolve()}")
print(f"Best Score: {best_score:.5f} at epoch {best_score_epoch}")
