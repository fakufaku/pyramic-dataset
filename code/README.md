Code of Pyramic Dataset
=======================

This folder contains the code and scripts used for

1. Automating the experiment
2. Processing the recordings into a usable form
3. Compress/Uncompress the samples

Here's a short description for each file here:

| File                  | Description                                                                    |
|-----------------------|--------------------------------------------------------------------------------|
| `calibration.py`      | Routines for the calibration of microphones and sources locations              |
| `compress.sh`         | Compress a bunch of wav files to tta format using ffmpeg                       |
| `compute_tdoa.py`     | Compute the time difference of arrivals between a reference mic and the others |
| `deconvolution.py`    | Routine for the Wiener deconvolution                                           |
| `ffmpeg_audio.py`     | Reads an audio file with FFMPEG                                                |
| `gen_sweeps.py`       | Generates linear and exponential sweeps                                        |
| `run_calibration.py`  | Produce a calibration file for the microphones and sources locations           |
| `run_deconvolution.py`| Produce all the impulse responses for the dataset                              |
| `segment.py`          | Segment the raw recordings into individual samples                             |
| `stitch_samples.py`   | Stitch the different audio samples before playback                             |
| `uncompress.sh`       | Uncompress tar-ed compressed audio to uncompressed audio (directly)            |

All this code was written by Robin Scheibler and is under the MIT License.
