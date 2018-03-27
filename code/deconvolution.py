'''
This file perform deconvolution in the Fourier domain.

History:
    2017 Robin Scheibler
'''
from __future__ import division, print_function

import numpy as np
from scipy import linalg as la

try:
    import mkl_fft as fft
    print('Using mkl')
except ImportError:
    import numpy.fft as fft

def deconvolve(y, s, thresh=0.05):
    '''
    Deconvolve an excitation signal from an impulse response

    Parameters
    ----------

    y : ndarray
        The recording
    s : ndarray
        The excitation signal
    h_len : int
        The length of the impulse response
    '''

    # FFT length including zero padding
    n = y.shape[0] + s.shape[0] - 1

    # let FFT size be even for convenience
    if n % 2 != 0:
        n += 1

    # Forward transforms
    Y  = fft.rfft(np.array(y, dtype=np.float32), n=n) / np.sqrt(n)
    S = fft.rfft(np.array(s, dtype=np.float32), n=n) / np.sqrt(n)

    # Only do the division where S is large enough
    H = np.zeros(*Y.shape, dtype=Y.dtype)
    I = np.where(np.abs(S) > thresh)
    H[I] = Y[I] / S[I]

    # Inverse transform
    h = fft.irfft(H, n=n)

    return h

