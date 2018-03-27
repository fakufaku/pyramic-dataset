'''
This script takes all the samples in the 'samples' folder and
stitch them into a single wav file.
A short pause is added between the samples and samples that are
at a different sampling frequency are resampled to 48 kHz.

History:
    2017 Robin Scheibler: initial script
'''
from __future__ import division, print_function
import numpy as np
import os
from scipy.io import wavfile
from samplerate import resample

output_name = u'all_samples.wav'
samples_dir = u'samples'
files = sorted(os.listdir(samples_dir))

if output_name in files:
    files.remove(output_name)

fs = 48000
n_silence = 1500

signals = [np.zeros(fs)]  # with a sec of silence at beginning
silence = np.zeros(n_silence)

ordered_files = ['silence']

for fname in files[::-1]:
    name = '/'.join([samples_dir, fname])

    # .DS_store shit
    if os.path.splitext(name)[1] != '.wav':
        continue

    ordered_files.append(os.path.splitext(fname)[0])

    rate, buf = wavfile.read(name)

    buf = np.array(buf, dtype=np.float64) / np.max(np.abs(buf)) * 0.99

    if rate != fs:
        print(fname, 'resample')
        buf = resample(buf, fs / rate, 'sinc_best')
    else:
        print(fname)

    signals.append(buf)
    signals.append(silence)

signals[-1] = np.zeros(fs)

standard_devs = np.r_[[np.std(a) for a in signals[1::2]]]
ref_std = np.max(standard_devs)

'''
for signal, sd in zip(signals[1::2], standard_devs):
    signal /= np.std(signal)
    signal *= ref_std
'''

print('this is usefule for segmentation:')
print('lengths=', [len(signal) for signal in signals])
print('labels =', ordered_files)

all_samples = np.concatenate(signals)
all_samples *= 0.5 / np.max(np.abs(all_samples))

all_samples_i16 = (all_samples).astype(np.float32)

wavfile.write(os.path.join(samples_dir, output_name), fs, all_samples_i16)
