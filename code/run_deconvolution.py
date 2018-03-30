'''
Run Deconvolution
=================

This script will run the deconvolution on every exponential sweep in the
dataset to produce the impulse response of every source location recorded.

Author: 2018 (c) Robin Scheibler
License: MIT License
'''
import sys, argparse, os
from scipy.io import wavfile
import numpy as np

sys.path.append('./code')
from deconvolution import deconvolve

angles = np.arange(0, 360, 2)
spkrs = [0, 1, 2]

filename_fmt = '{sweep}/{sweep}_spkr{spkr}_angle{angle}.wav'
noise_ref_fmt = 'silence/silence_spkr{spkr}_angle{angle}.wav'

def pre_proc(x):
    ''' Convert to floating points and remove the DC component '''
    return x.astype(np.float32) - np.mean(x, axis=0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=(
          'Deconvolve the response of the array for all' 
        + 'of the recordings and stores the impulse responses')
        )
    parser.add_argument('test_signal', type=str, help='The test signal used for the recordings')
    parser.add_argument('segmented_dir', type=str, help='Location of segmented recordings')
    parser.add_argument('output_dir', type=str, help='Location where to store the impulse responses')
    parser.add_argument('-q', '--qc', action='store_true', help='Create plots for quality control')
    parser.add_argument('-l', '--length', type=int, help='cut-off length of the impulse response')
    parser.add_argument('-f', '--force', action='store_true', help='Overwrite output file if existing')
    args = parser.parse_args()

    r_sweep, data_sweep = wavfile.read(args.test_signal)

    sweep_name = os.path.basename(args.test_signal)
    sweep_name = os.path.splitext(sweep_name)[0]

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.qc:
        import matplotlib.pyplot as plt
        qc_dir = os.path.join(args.output_dir, 'qc')
        if not os.path.exists(qc_dir):
            os.mkdir(qc_dir)

    for spkr in spkrs:
        for angle in angles:

            # output file name
            bname = 'ir_spkr{}_angle{}.wav'.format(spkr, angle)
            fn_out = os.path.join(args.output_dir, bname)

            if os.path.exists(fn_out):
                print('spkr={} angle={}: file exists. Skip'.format(spkr, angle))
                continue

            # open file with noise reference
            bname = noise_ref_fmt.format(spkr=spkr, angle=angle)
            fn_noise = os.path.join(args.segmented_dir, bname)
            r_noise, data_noise = wavfile.read(fn_noise)
            data_noise = pre_proc(data_noise)

            # Resample noise if necessary
            if r_noise != r_sweep:
                from samplerate import resample
                data_noise = resample(data_noise, r_sweep / r_noise, 'sinc_best')

            # open recording file
            bname = filename_fmt.format(sweep=sweep_name, spkr=spkr, angle=angle)
            fn_rec = os.path.join(args.segmented_dir, bname)
            r_rec, data_rec = wavfile.read(fn_rec)
            data_rec = pre_proc(data_rec)

            if r_rec != r_sweep:
                raise ValueError('Some mismatch in the sampling frequencies')

            ir = deconvolve(data_rec, data_sweep, data_noise, periodogram_len=64)

            if args.length is not None:
                ir = ir[:args.length,:]

            wavfile.write(fn_out, r_rec, ir)

            if args.qc:
                plt.figure(1)
                time = np.arange(ir.shape[0]) / r_rec * 1000
                plt.plot(time, ir)
                plt.xlabel('time [ms]')
                plt.title('IR speaker {} angle {}'.format(spkr, angle))
                plt.tight_layout(pad=0.1)
                bname = 'ir_spkr{}_angle{}.png'.format(spkr, angle)
                fn_out = os.path.join(qc_dir, bname)
                plt.savefig(fn_out)
                plt.clf()

            print('Speaker', spkr, 'Angle', angle, '... done.')


            





