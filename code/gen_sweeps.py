# Experiment from 2016/08/31
from __future__ import division, print_function

import numpy as np
import sounddevice as sd
import time
from scipy.io import wavfile

def window(signal, n_win):
    ''' window the signal at beginning and end with window of size n_win/2 '''

    win = np.hanning(2*n_win)

    sig_copy = signal.copy()

    sig_copy[:n_win] *= win[:n_win]
    sig_copy[-n_win:] *= win[-n_win:]

    return sig_copy

def exponential_sweep(T, fs, f_low=50., f_high=22000.):

    f1 = f_low     # Start frequency in [Hz]
    f2 = f_high  # End frequency in [Hz]
    Ts = 1./fs   # Sampling period in [s]
    N = np.floor(T/Ts)
    n  = np.arange(0, N, dtype='float64')  # Sample index

    om1 = 2*np.pi*f1
    om2 = 2*np.pi*f2

    x_exp = 0.95*np.sin(om1*N*Ts / np.log(om2/om1) * (np.exp(n/N*np.log(om2/om1)) - 1))
    x_exp = x_exp[::-1]

    return x_exp

def linear_sweep(T, fs, f_low=50., f_high=22000.):

    f1 = f_low     # Start frequency in [Hz]
    f2 = f_high  # End frequency in [Hz]
    Ts = 1./fs   # Sampling period in [s]
    N = np.floor(T/Ts)
    n  = np.arange(0, N, dtype='float64')  # Sample index

    om1 = 2*np.pi*f1
    om2 = 2*np.pi*f2

    x_exp = 0.95*np.sin(2 * np.pi * 0.5 * (f1 + (f2 - f1) * n / N) * n / fs)
    x_exp = x_exp[::-1]

    return x_exp


if __name__ == "__main__":

    # Signal parameters
    fs = 48000
    f_lo = 20
    f_hi = fs / 2 - 50.
    T = 3.
    
    # Setup device
    sd.default.device = 2
    sd.default.samplerate = fs
    sd.default.channels = (8,1)

    # create sweep signal
    sweep = np.zeros(int(T*fs))

    # save the sweep to deconvolve later
    sweep_exp = exponential_sweep(T, fs, f_low=f_lo, f_high=f_hi)
    sweep_exp_win = window(sweep_exp, int(fs*0.01))

    wavfile.write("sweep_exp.wav", fs, sweep_exp)
    wavfile.write("sweep_exp_win.wav", fs, sweep_exp_win)

    sweep_lin = linear_sweep(T, fs, f_low=f_lo, f_high=f_hi)
    sweep_lin_win = window(sweep_lin, int(fs*0.01))

    wavfile.write('sweep_lin.wav', fs, sweep_lin)
    wavfile.write('sweep_lin_win.wav', fs, sweep_lin_win)

