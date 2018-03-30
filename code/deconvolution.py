'''
Deconvolution
=============

This file contains the routine to perform Wiener deconvolution in the Fourier domain.

Author: 2018 (c) Robin Scheibler
License: MIT License
'''
from __future__ import division, print_function

import numpy as np
from scipy import linalg as la
from scipy.interpolate import interp1d

try:
    # optionaly: https://github.com/IntelPython/mkl_fft
    # can be used to accelerate FFT
    import mkl_fft._numpy_fft as fft
    print('Using mkl_fft')
except ImportError:
    import numpy.fft as fft

def plot_spectrum(x, fs, *args, **kwargs):
    n = x.shape[0]
    f = np.linspace(0, 0.5, n // 2 + 1) * fs
    X = np.abs(np.fft.rfft(x, axis=0)) / np.sqrt(n)
    plt.plot(f, 20*np.log10(X), *args, **kwargs)

def periodogram(data, p_len, n=None):
    '''
    Compute the Periodogram of the input data

    Parameters
    ----------
    data: ndarray
        The input data
    p_len: int
        The number of points in the analyisis periodogram
    n: int, optional
        Optionaly, the periodogram is linearly resampled to this number of point
    '''

    n_frames = (data.shape[0] // p_len)
    if data.ndim == 1:
        frames = data[:n_frames*p_len].reshape((n_frames,p_len))
    elif data.ndim == 2:
        frames = data[:n_frames*p_len].reshape((n_frames,p_len,-1))
    else:
        raise ValueError('Dimension larger than 2 not supported')

    P = np.mean(np.abs(np.fft.rfft(frames, axis=1) / np.sqrt(p_len))**2, axis=0)

    if n is not None:
        f = np.arange(p_len // 2 + 1) / p_len
        interpolator = interp1d(f, P, kind='linear', axis=0)
        f_long = np.arange(n // 2 + 1) / n
        P = interpolator(f_long)

    return P


def deconvolve(y, s, noise, periodogram_len=64):
    '''
    Wiener deconvolution of an excitation signal from an impulse response

    Parameters
    ----------

    y : ndarray
        The recording
    s : ndarray
        The excitation signal
    noise: ndarray
        The noise reference signal

    Returns
    -------
    h_len : int
        The length of the impulse response
    '''

    # FFT length including zero padding
    n = y.shape[0] + s.shape[0] - 1

    # let FFT size be even for convenience
    if n % 2 != 0:
        n += 1

    # Compute the periodogram of the noise signal
    N = periodogram(noise, periodogram_len, n=n)

    # Forward transforms
    Y  = fft.rfft(y, n=n, axis=0) / np.sqrt(n)
    S = fft.rfft(s, n=n, axis=0) / np.sqrt(n)

    # Wiener deconvolution
    if Y.ndim > 1:
        S = S[:,None]

    H = Y * np.conj(S) / (np.abs(S)**2 + N)

    # Inverse transform
    h = fft.irfft(H, n=n, axis=0)

    return h

if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt
    from scipy.io import wavfile
    import pyroomacoustics as pra

    parser = argparse.ArgumentParser(description='Deconvolve an impulse response')
    parser.add_argument('recording', type=str, help='The recording file to deconvolve')
    parser.add_argument('signal', type=str, help='The known test signal used at recording time')
    parser.add_argument('noise', type=str, help='A template of the noise signal')
    parser.add_argument('-l', '--length', type=int, help='The number of samples to which we limit the length of impulse response')
    args = parser.parse_args()


    r_rec, data_rec = wavfile.read(args.recording)
    r_test, data_test = wavfile.read(args.signal)
    r_noise, data_noise = wavfile.read(args.noise)

    pre_proc = lambda x: x.astype(np.float32) - np.mean(x, axis=0)
    data_rec = pre_proc(data_rec)
    data_test = pre_proc(data_test)
    data_noise = pre_proc(data_noise)

    if r_rec != r_test:
        raise ValueError('The two signals need to have the same sampling frequency')

    fs = r_rec

    h = deconvolve(data_rec, data_test, data_noise, periodogram_len=64)

    if args.length is not None:
        h = h[:args.length,:]

    plt.figure()

    y_lim = [-50, 70]

    plt.subplot(2,2,1)
    plot_spectrum(data_rec, fs)
    plt.title('Recording')
    plt.ylim(y_lim)

    plt.subplot(2,2,2)
    plot_spectrum(data_test, fs)
    plt.title('Test Signal')
    plt.ylim(y_lim)

    plt.subplot(2,2,3)
    Pn = periodogram(data_noise, 64)
    f = np.linspace(0, 0.5, len(Pn)) * fs
    plot_spectrum(data_noise, fs, 'g')
    plt.plot(f, 10*np.log10(Pn), 'r')
    plt.title('Noise')
    plt.ylim(y_lim)

    plt.subplot(2,2,4)
    plot_spectrum(h, fs)
    plt.title('Deconvolved Signal')
    plt.ylim(y_lim)

    plt.tight_layout(pad=0.1)

    plt.figure()
    time = np.arange(len(h)) / r_rec * 1000
    plt.plot(time, h)
    plt.title('Impulse Responses')
    plt.xlabel('Time [ms]')

    plt.tight_layout(pad=0.1)

    plt.show()
