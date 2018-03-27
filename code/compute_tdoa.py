"""
Compute TDOA
===========

This script uses the white noise signal recording to compute the
TDOA between every pair of microphones.
"""

import numpy as np
import sys, json, os, argparse
from scipy.io import wavfile
import pyroomacoustics as pra
import matplotlib.pyplot as plt

sys.path.append('./code')

from calibration import sph2cart


# These are the valid angles and speakers
angles = list(range(0, 360, 2))
spkr_list = {'middle':0, 'low':1, 'high':2}
spkrs = sorted(spkr_list.keys())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Read the noise recordings and use them to compute TDOA between the microphones')
    parser.add_argument('-o', '--output', type=str, default='pyramic_tdoa.json',
            help='The file where to write all the computed delays')
    parser.add_argument('-p', '--protocol', type=str, default='protocol.json',
            help='The protocol JSON file containing the experiment details.')
    parser.add_argument('-r', '--ref', type=int, default=0, help='The reference microphone')
    args = parser.parse_args()

    delays_filename = args.output
    protocol_filename = args.protocol
    mic_ref = args.ref

    # STEP 0 : Read in the experimental data

    # We should make this the default structure
    # it can be applied by copying/downloading the data or creating a symbolic link
    rec_file = './segmented/noise/noise_spkr{spkr}_angle{angle}.wav'

    # Open the protocol json file
    with open(protocol_filename) as fd:
        exp_data = json.load(fd)

    # Experiment related parameters
    temp = 0.5 * (exp_data['conditions']['start']['temperature'] +
                  exp_data['conditions']['end']['temperature'])
    hum = 0.5 * (exp_data['conditions']['start']['humidity'] +
                 exp_data['conditions']['end']['humidity'])

    # speed of sound (100 kpa pressure)
    sound_speed = pra.experimental.calculate_speed_of_sound(temp, hum, 100.)

    # the microphone array geometry
    mic_array = np.array(exp_data['geometry']['microphones']['locations']).T
    num_mic = mic_array.shape[1]  # number of microphones

    # compute the colatitude of the speakers
    array_center = np.mean(mic_array, axis=1)
    spkr_colatitude = dict()
    pyramic_height = np.abs(array_center[2] - np.min(mic_array[2,:])) + \
                     exp_data['geometry']['microphones']['height']  # offset of pyramic measurement
    for spkr in spkr_list.keys():
        spkr_height = exp_data['geometry']['speakers']['location'][spkr]['height']
        baffle_corr = exp_data['geometry']['speakers']['baffles_correction']['woofer']
        dh = pyramic_height - spkr_height - baffle_corr
        spkr_colatitude[spkr] = \
            np.pi / 2 + np.arctan2(dh, exp_data['geometry']['speakers']['location'][spkr]['distance'])

    # These are the unit vectors of DOA from the speakers locations, the sound sources
    sources = np.zeros((3, len(angles), len(spkrs)))

    ### STEP 1 : recover the time of arrival of all sources by generalized cross-correlation ###

    mic_no_ref = list(range(mic_array.shape[1]))
    mic_no_ref.remove(mic_ref)

    delays = np.zeros((mic_array.shape[1] - 1, len(angles), len(spkr_list)))

    for i_a, angle in enumerate(angles):
        for i_s, spkr in enumerate(spkrs):
            print('angle={} spkr={}'.format(angle, spkr))

            # record the unit vector repr of the source DOA
            sources[:,i_a,i_s] = sph2cart(1., spkr_colatitude[spkr], np.radians(angle))

            # read the recording
            rate, audio = wavfile.read(rec_file.format(angle=angle, spkr=spkr_list[spkr]))

            # deconvolve and find dominant delay for all channels
            for i, channel in enumerate(mic_no_ref):
                delays[i,i_a,i_s] = \
                    pra.experimental.tdoa(audio[:,channel], audio[:,mic_ref], fs=rate, interp=4)

    ### STEP 2 : Compile and save the important information

    pyramic_tdoa_info = {
            'sound_speed_mps' : sound_speed,
            'mic_ref' : mic_ref,
            'delays':delays.tolist(),
            'sources' : sources.tolist(),
            'microphones' : mic_array.tolist(),
            }


    with open(delays_filename, 'w') as f:
        json.dump(pyramic_tdoa_info, f)

