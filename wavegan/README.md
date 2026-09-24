# WaveGAN (v2)

Official implementation of **WaveGAN**, a machine learning algorithm which learns to generate raw audio waveforms.

**UPDATE (2/2/19)**: We have made substantial improvements to this repository in response to common requests:
- **Added streaming data loader allowing you to train a WaveGAN on MP3s/WAVs/OGGs/etc. without preprocessing**
- **Added ability to train WaveGANs capable of generating longer audio examples (up to 4 seconds at 16kHz)**
- Added support for any audio sample rate
- Added support for multi-channel audio
- Compatibility with Python 3 and Tensorflow 1.12.0
- Old (v1) version still available at [this tag](https://github.com/chrisdonahue/wavegan/tree/v1)

<img src="static/wavegan.png"/>

This is the official TensorFlow implementation of WaveGAN (Donahue et al. 2018) ([paper](https://arxiv.org/abs/1802.04208)) ([demo](https://chrisdonahue.com/wavegan)) ([sound examples](http://wavegan-v1.s3-website-us-east-1.amazonaws.com)). WaveGAN is a machine learning algorithm which learns to synthesize raw waveform audio by observing many examples of real audio. WaveGAN is comparable to the popular DCGAN approach (Radford et al. 2016) for learning to generate images.

In this repository, we include an implementation of WaveGAN capable of learning to generate up to 4 seconds of audio at 16kHz. For comparison, we also include an implementation of SpecGAN, an approach to audio generation which applies image-generating GANs to image-like audio spectrograms.

<img src="static/results.png"/>

WaveGAN is capable of learning to synthesize audio in many different sound domains. In the above figure, we visualize real and WaveGAN-generated audio of speech, bird vocalizations, drum sound effects, and piano excerpts. These sound examples and more can be heard [here](http://wavegan-v1.s3-website-us-east-1.amazonaws.com).

## Legacy TensorFlow requirements

```
pip install tensorflow-gpu==1.12.0
pip install scipy==1.0.0
pip install matplotlib==3.0.2
pip install librosa==0.6.2
```

## PyTorch implementation for current NVIDIA GPUs

`train_wavegan_torch.py` is the maintained training entry point for the
bundled 128-point click waveforms. It uses PyTorch, automatically selects CUDA
when a CUDA-enabled PyTorch build is installed, and implements WGAN-GP with
the same 128-point generator/discriminator structure.

Install the CUDA-enabled PyTorch dependencies listed in `../requirements.txt`,
then run this from the `wavegan` directory:

```
python train_wavegan_torch.py --device cuda --output-dir ./train_click_wavegan_torch_gpu
```

The original TensorFlow 1.12 scripts remain for reference only; they are not
compatible with the RTX 4060.

### Evaluate an unlabeled click-waveform model

The original README's Inception Score requires a labeled 16 kHz classification
dataset and cannot be applied to the bundled 576 kHz click waveforms. Use the
PyTorch evaluator to compare real and generated distributions instead. It
reports Wasserstein distances for pulse amplitude, RMS, peak position, FWHM,
dominant frequency, spectral centroid, and spectral bandwidth, plus mean-PSD
errors. It also saves feature-distribution and PSD comparison figures.

```
python evaluate_wavegan_torch.py \
    --checkpoint ./train_click_wavegan_torch_gpu/wavegan_epoch_0200.pt \
    --device cuda
```

### Train and evaluate SpecGAN for comparison

`train_specgan_torch.py` is the PyTorch adaptation of SpecGAN for 128-point
click waveforms. It trains on padded 32×32 log-magnitude spectra and uses
Griffin–Lim to reconstruct generated waveforms. The run uses the same dataset,
200 epochs, WGAN-GP loss, and label-free metrics as WaveGAN.

```
python train_specgan_torch.py --device cuda --epochs 200 \
    --output-dir ./train_click_specgan_torch_gpu

python evaluate_specgan_torch.py \
    --checkpoint ./train_click_specgan_torch_gpu/specgan_epoch_0200.pt \
    --device cuda

python compare_gan_evaluations.py \
    --wavegan-metrics ./train_click_wavegan_torch_gpu/evaluation_wavegan_epoch_0200/metrics.json \
    --specgan-metrics ./train_click_specgan_torch_gpu/evaluation_specgan_epoch_0200/metrics.json \
    --output-dir ./wavegan_specgan_comparison
```

## Datasets

WaveGAN can now be trained on datasets of arbitrary audio files (previously required preprocessing). You can use any folder containing audio, but here are a few example datasets to help you get started:

- [Speech Commands Zero through Nine (SC09)](https://drive.google.com/file/d/1WmlNdi_lMH13d08_0GymNcTpWBKo4MYn/view?usp=sharing)
- [Drum sound effects](https://drive.google.com/file/d/1552QqU_MYCaxHK71psNisG4-w9VpUEli/view?usp=sharing)
- [Bach piano performances](https://drive.google.com/file/d/1DPGRm1Di-ITAcFnmQtnr5nw-BnMzwb6y/view?usp=sharing)

## Train a WaveGAN

Here is how you would begin (or resume) training a WaveGAN on random clips from a directory containing longer audio, i.e., more than a few seconds per file:

```
export CUDA_VISIBLE_DEVICES="0"
python train_wavegan.py train ./train \
	--data_dir ./data/dir_with_longer_audio_files
```

If you are instead training on datasets of short sound effects (e.g., SC09 or drum sound effects), you want to use this command:

```
export CUDA_VISIBLE_DEVICES="0"
python train_wavegan.py train ./train \
	--data_dir ./data/sc09/train \
	--data_first_slice \
	--data_pad_end \
	--data_fast_wav
```

### Train on the bundled click-waveform data

The repository's `../data/wav` directory contains short, 576 kHz click
waveforms in nested directories. The default WaveGAN configuration has been
adapted to this data: it recursively finds WAV files, normalizes each pulse,
zero-pads it to 128 samples, and generates 128-sample waveforms. From the
`wavegan` directory, run:

```
python train_wavegan.py train ./train_click
```

Use `--data_dir` to override the default dataset directory.
The bundled TensorFlow 1.12 environment defaults to CPU execution because its
GPU runtime is not compatible with the current RTX 4060 setup. Use
`--runtime_device gpu` only after providing a compatible TensorFlow/CUDA stack.

Because our codebase buffers audio clips directly from files, it is important to change the data-related command line arguments to be appropriate for your dataset (see [#data-considerations](data considerations below)).

We currently do not support training on multiple GPUs. If your machine has multiple GPUs, make sure to set the `CUDA_VISIBLE_DEVICES` flag as shown above.

While you can *technically* train a WaveGAN on CPU, it is prohibitively slow and not recommended. If you do attempt this, add the flag `--data_prefetch_gpu_num -1`.

### Data considerations

The checked-in defaults target the bundled short click-waveform dataset. For longer audio or a different dataset, set the data-related command line arguments explicitly. For short sound effects, use `--data_first_slice` to only extract the first slice from each audio file and `--data_pad_end` to zero-pad clips shorter than the selected slice length.

If your dataset consists exclusively of "standard" WAV files (16-bit signed PCM or 32-bit float), you can use the flag `--data_fast_wav` which will use `scipy` (faster) to decode your audio instead of `librosa`. This may slightly increase training speed.

If you want to change the generation length, set `--data_slice_len` to `128`, `16384`, `32768`, or `65536` to generate that many audio samples. If you choose a larger generation length, you will likely want to reduce the number of model parameters to train more quickly (e.g. `--wavegan_dim 32`). You can also adjust the sampling rate using `--data_sample_rate` which will effectively change the generation length.

If you have stereo (or multi-channel) audio, adjust `--data_num_channels` as needed. If you are modeling more than 2 channels, each audio file must have the exact number of channels specified.

If you want to normalize each audio file before training, set `--data_normalize`.

### Quality considerations

If your results are too noisy, try adding a post-processing filter with `--wavegan_genr_pp`. You may also want to change the amount of or remove phase shuffle using `--wavegan_disc_phaseshuffle 0`. Increasing either the model size (`--wavegan_dim`) or filter length (`--wavegan_kernel_len`) may improve results but will increase training time.

### Monitoring

To run a script that will dump a preview of fixed latent vectors at each checkpoint on the CPU

```
export CUDA_VISIBLE_DEVICES="-1"
python train_wavegan.py preview ./train
```

To back up checkpoints every hour (GAN training may occasionally collapse so it's good to have backups)

```
python backup.py ./train 60
```

To monitor training via tensorboard, use

```
tensorboard --logdir=./train
```

If you are training on the SC09 dataset, this command will (slowly) calculate inception score at each checkpoint

```
export CUDA_VISIBLE_DEVICES="-1"
python train_wavegan.py incept ./train
```

## Train a SpecGAN

The primary focus of this repository is on WaveGAN, our raw audio generation method. For comparison, we also include an implementation of SpecGAN, an approach to generating audio by applying image-generating GANs on image-like audio spectrograms. This implementation only generates spectrograms of one second in length at 16khz.

Before training a SpecGAN, we must first compute mean and variance of each spectrogram bin to use for normalization. This may take a while(you can also measure these statistics on a subset of the data)

```
python train_specgan.py moments ./train \
	--data_dir ./data/dir_with_mp3s \
	--data_moments_fp ./train/moments.pkl
```

To begin (or resume) training on GPU:

```
python train_specgan.py train ./train \
	--data_dir ./data/dir_with_mp3s \
	--data_moments_fp ./train/moments.pkl
```

### Monitoring

To run a script that will dump a preview of fixed latent vectors at each checkpoint on the CPU

```
export CUDA_VISIBLE_DEVICES="-1"
python train_specgan.py preview ./train \
	--data_moments_fp ./train/moments.pkl
```

To back up checkpoints every hour (GAN training will occasionally collapse)

```
python backup.py ./train 60
```

To monitor training via tensorboard, use

```
tensorboard --logdir=./train
```

If you are training on the SC09 dataset, this command will (slowly) calculate inception score at each checkpoint

```
export CUDA_VISIBLE_DEVICES="-1"
python train_specgan.py incept ./train \
	--data_moments_fp ./train/moments.pkl
```

## Generation

The training scripts for both WaveGAN and SpecGAN create simple TensorFlow MetaGraphs for generating audio waveforms, located in the training directory. An example usage is below; see [this Colab notebook](https://colab.research.google.com/drive/1e9o2NB2GDDjadptGr3rwQwTcw-IrFOnm) for additional features.

```py
import tensorflow as tf
from IPython.display import display, Audio

# Load the graph
tf.reset_default_graph()
saver = tf.train.import_meta_graph('infer.meta')
graph = tf.get_default_graph()
sess = tf.InteractiveSession()
saver.restore(sess, 'model.ckpt')

# Create 50 random latent vectors z
_z = (np.random.rand(50, 100) * 2.) - 1

# Synthesize G(z)
z = graph.get_tensor_by_name('z:0')
G_z = graph.get_tensor_by_name('G_z:0')
_G_z = sess.run(G_z, {z: _z})

# Play audio in notebook
display(Audio(_G_z[0, :, 0], rate=16000))
```

## Evaluation

Our [paper](https://arxiv.org/abs/1802.04208) uses Inception score to (roughly) measure model performance. If you plan to *directly* compare to our reported numbers, you should run [this script](https://github.com/chrisdonahue/wavegan/blob/master/eval/inception/score.py) on a directory of 50,000 16-bit PCM WAV files with 16384 samples each.

```
python score.py --audio_dir wavs
```

To reproduce our paper results (9.18 +- 0.04) for the SC09 ([download](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/sc09.tar.gz)) training dataset, run

```
python score.py --audio_dir sc09/train  --fix_length --n 18620
```

## Web code

Under `web`, we also include a JavaScript implementation of WaveGAN (generation only). Using this implementation, we created a [procedural drum machine](https://chrisdonahue.com/wavegan) powered by a WaveGAN trained on drum sound effects.

### Attribution

If you use this code in your research, cite via the following BibTeX:

```
@inproceedings{donahue2019wavegan,
  title={Adversarial Audio Synthesis},
  author={Donahue, Chris and McAuley, Julian and Puckette, Miller},
  booktitle={ICLR},
  year={2019}
}
```
