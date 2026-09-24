from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import DataLoader, Dataset


def discover_wav_paths(root):
    """Return a deterministic list of WAV files beneath ``root``."""
    root_path = Path(root)
    wav_paths = sorted(
        path for path in root_path.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"
    )
    if not wav_paths:
        raise FileNotFoundError(f"No .wav files found under {root_path.resolve()}.")
    return wav_paths


def _read_wav(path):
    """Read PCM or IEEE-float WAV and return its first channel in [-1, 1]."""
    _, samples = wavfile.read(path)
    if samples.ndim == 2:
        samples = samples[:, 0]
    elif samples.ndim != 1:
        raise ValueError(f"Unsupported WAV shape {samples.shape} in {path}.")

    if np.issubdtype(samples.dtype, np.floating):
        samples = samples.astype(np.float32)
    elif samples.dtype == np.uint8:
        samples = (samples.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(samples.dtype, np.signedinteger):
        info = np.iinfo(samples.dtype)
        samples = samples.astype(np.float32) / float(abs(info.min))
    else:
        raise ValueError(f"Unsupported WAV sample dtype {samples.dtype} in {path}.")

    return np.clip(samples, -1.0, 1.0)


class WavSegmentDataset(Dataset):
    """Expose every WAV file as 128-point segments; the final segment is zero-padded."""

    def __init__(self, root=None, segment_length=128, wav_paths=None):
        self.segment_length = segment_length
        if wav_paths is None:
            if root is None:
                raise ValueError("Provide either root or wav_paths.")
            self.wav_paths = discover_wav_paths(root)
        else:
            self.wav_paths = [Path(path) for path in wav_paths]
            if not self.wav_paths:
                raise ValueError("wav_paths must contain at least one WAV file.")

        self.segments = []
        for path in self.wav_paths:
            waveform = _read_wav(path)
            if waveform.size == 0:
                continue
            for start in range(0, waveform.size, segment_length):
                segment = waveform[start:start + segment_length]
                if segment.size < segment_length:
                    segment = np.pad(segment, (0, segment_length - segment.size))
                self.segments.append(segment)

        if not self.segments:
            raise ValueError("All discovered WAV files are empty.")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        return torch.from_numpy(self.segments[index]).unsqueeze(0)


def get_wav_dataloader(params):
    dataset = WavSegmentDataset(
        root=params["data_root"],
        segment_length=params["signal_length"],
    )
    return DataLoader(dataset, batch_size=params["bsize"], shuffle=True, num_workers=0)
