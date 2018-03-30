'''
FFMPEG AUDIO
============

This is a thin wrapper around the ffmpeg software. It can be used
to read probably very many audio file formats into python. It was
tested with TTA and WAV formats. It only supports signed 16 bit audio
as of now.

Author: 2018 (c) Robin Scheibler
License: MIT License
'''

import numpy as np
import re
import subprocess as sp

def read(filename, ffmpeg_bin='ffmpeg', debug=False):
    '''
    Read an audio file into python using FFMPEG. The syntax
    is similar to `scipy.io.wavfile.read`.

    Note: only supports signed 16 bit audio

    Parameters
    ----------
    filename: str
        The audio filename
    ffmpeg_bin: str, optional
        The name of the ffmpeg executable
    debug: bool, optional
        Print some debug information

    Returns
    -------
    samplerate: The samplerate of the audio signal.
    audio: An ndarray containing the audio samples. For multichannel audio it returns
        a 2D array with every column corresponding to a channel.
    '''
    
    command = [ ffmpeg_bin, '-i', filename ]
    with sp.Popen(command, stdout = sp.PIPE, stderr = sp.PIPE, bufsize=100000) as pipe:
        _, stderr = pipe.communicate()

        for l in stderr.decode('utf-8').split('\n'):
            if debug:
                print(l)

            words = l.split()
            if len(words) >= 1 and words[0] != 'Stream':
                continue

            fmt = re.search('Audio: (.*?) ', l).group(1)
            samplerate = re.search(', (\d+) Hz,', l).group(1)
            n_channels, sampleformat, n_bits = re.search(', (\d+ channels|mono|stereo), ([a-z]+)([0-9]*)', l).group(1,2,3)

            break

    if n_channels == 'mono':
        n_channels = 1
    elif n_channels == 'stereo':
        n_channels = 2
    else:
        n_channels = int(re.search('(\d+) channels', n_channels).group(1))

    samplerate = int(samplerate)

    if sampleformat == 'flt':

        dtype = np.float32
        out_format = 'f32le'
        n_bits = 32

    elif sampleformat == 's':

        n_bits = int(n_bits)

        if n_bits == 16:
            dtype = np.int16
            out_format = 's16le'
        elif n_bits == 32:
            dtype = np.int32
            out_format = 's32le'
        else:
            raise ValueError('For now only signed 16/32 bit audio is supported. Sorry')

    else:
        raise ValueError('For now only signed 16/32 or float 32 bit audio is supported. Sorry')

    n_bytes = n_bits // 8

    # chunks of a second of audio
    n_chunk = samplerate

    # now read the samples
    command = [ ffmpeg_bin, '-i', filename, '-f', out_format, '-' ]
    with sp.Popen(command, stdout = sp.PIPE, stderr = sp.PIPE, bufsize=n_channels * n_bytes * n_chunk) as pipe:

        raw_audio, _ = pipe.communicate()
        audio = np.frombuffer(raw_audio, dtype=dtype).reshape((-1,n_channels))

    return samplerate, audio

