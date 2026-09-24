"""Create 128-point waveform and magnitude-spectrum features.

The script first displays one random WAV sample.  Close that window to start
batch processing.  Every input WAV produces a float32 .npy file with shape
(2, 128): [waveform, magnitude spectrum].
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "wav"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "threewav"
FEATURE_POINTS = 128


def read_wav_as_mono(path: Path) -> tuple[int, np.ndarray]:
    """Read a WAV file, select its first channel, and convert it to [-1, 1]."""
    sample_rate, samples = wavfile.read(path)
    if samples.ndim == 2:
        samples = samples[:, 0]
    if samples.ndim != 1:
        raise ValueError(f"Unsupported WAV shape {samples.shape}.")
    if samples.size == 0:
        raise ValueError("The WAV file is empty.")

    if np.issubdtype(samples.dtype, np.floating):
        waveform = samples.astype(np.float32, copy=False)
    elif samples.dtype == np.uint8:
        waveform = (samples.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(samples.dtype, np.signedinteger):
        waveform = samples.astype(np.float32) / float(abs(np.iinfo(samples.dtype).min))
    else:
        raise ValueError(f"Unsupported sample type {samples.dtype}.")

    return sample_rate, np.clip(waveform, -1.0, 1.0)


def resample_to_points(values: np.ndarray, points: int = FEATURE_POINTS) -> np.ndarray:
    """Linearly resample a one-dimensional sequence to a fixed point count."""
    if values.size == 1:
        return np.full(points, values[0], dtype=np.float32)
    old_positions = np.linspace(0.0, 1.0, num=values.size, dtype=np.float64)
    new_positions = np.linspace(0.0, 1.0, num=points, dtype=np.float64)
    return np.interp(new_positions, old_positions, values).astype(np.float32)


def normalize_to_minus_one_to_one(values: np.ndarray) -> np.ndarray:
    """Min-max normalize one feature sequence to [-1, 1]."""
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if np.isclose(maximum, minimum):
        return np.zeros_like(values, dtype=np.float32)
    normalized = 2.0 * (values - minimum) / (maximum - minimum) - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def build_features(waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return 128-point waveform and magnitude spectrum."""
    waveform_feature = resample_to_points(waveform)

    # rFFT covers the non-negative frequencies from 0 Hz to the Nyquist frequency.
    spectrum = np.fft.rfft(waveform)
    magnitude_feature = normalize_to_minus_one_to_one(resample_to_points(np.abs(spectrum)))

    return waveform_feature, magnitude_feature


def show_random_sample(wav_paths: list[Path]) -> None:
    """Display a random original-resolution waveform and its spectra."""
    selected_path = random.choice(wav_paths)
    sample_rate, waveform = read_wav_as_mono(selected_path)
    spectrum = np.fft.rfft(waveform)
    frequencies = np.fft.rfftfreq(waveform.size, d=1.0 / sample_rate)
    time_axis = np.arange(waveform.size) / sample_rate

    figure, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    figure.suptitle(f"Random sample: {selected_path.name}\nClose this window to start processing.")
    axes[0].plot(time_axis, waveform, linewidth=0.7)
    axes[0].set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")

    normalized_magnitude = normalize_to_minus_one_to_one(np.abs(spectrum))
    axes[1].plot(frequencies, normalized_magnitude, linewidth=0.7)
    axes[1].set(
        title="Magnitude spectrum (per-sample normalized)",
        xlabel="Frequency (Hz)",
        ylabel="Normalized magnitude",
        ylim=(-1.05, 1.05),
    )

    plt.show(block=True)


def update_progress(completed: int, total: int) -> None:
    """Write a dependency-free overall progress bar to the console."""
    width = 40
    filled = int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\rProcessing [{bar}] {completed}/{total}", end="", flush=True)


def process_wavs(input_root: Path, output_root: Path) -> None:
    wav_paths = sorted(path for path in input_root.rglob("*") if path.is_file() and path.suffix.lower() == ".wav")
    if not wav_paths:
        raise FileNotFoundError(f"No WAV files were found under {input_root}.")

    print(f"Found {len(wav_paths)} WAV files under: {input_root}")
    print("A random sample window will now open. Close it to begin processing.")
    show_random_sample(wav_paths)

    output_root.mkdir(parents=True, exist_ok=True)
    errors: list[tuple[Path, str]] = []
    for index, wav_path in enumerate(wav_paths, start=1):
        try:
            _, waveform = read_wav_as_mono(wav_path)
            features = np.stack(build_features(waveform), axis=0).astype(np.float32, copy=False)
            destination = output_root / wav_path.relative_to(input_root).with_suffix(".npy")
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination, features)
        except (OSError, ValueError) as error:
            errors.append((wav_path, str(error)))
        update_progress(index, len(wav_paths))

    print()
    print(f"Completed: {len(wav_paths) - len(errors)} files saved to: {output_root}")
    if errors:
        print(f"Skipped {len(errors)} unreadable/invalid WAV files:")
        for path, message in errors[:20]:
            print(f"  {path}: {message}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract two-channel 128-point features from WAV files.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="Recursive WAV input directory.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory for per-WAV .npy features.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    process_wavs(arguments.input_root.resolve(), arguments.output_root.resolve())
