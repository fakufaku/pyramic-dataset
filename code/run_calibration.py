"""
Calibration
===========

This script performs calibration of the array and the sources locations using
Thrun's affine structure from sound algorithm.
"""

import numpy as np
import json, os, argparse, sys
from scipy.io import wavfile
import pyroomacoustics as pra
import matplotlib.pyplot as plt

sys.path.append('./code')

from calibration import cart2sph, joint_calibration_gd, joint_calibration_sgd, structure_from_sound

if __name__ == '__main__':

    methods = ['gd', 'sgd', 'svd']

    parser = argparse.ArgumentParser(description='Calibrate the pyramic measurements')
    parser.add_argument('-f', '--file', type=str, default='pyramic_tdoa.json',
            help='The file containing the TDOA measurements')
    parser.add_argument('-m', '--method', type=str, choices=methods,
            help='The calibration method to use ''gd'' or ''svd''')
    args = parser.parse_args()

    with open(args.file, 'r') as f:
        pyramic_tdoa = json.load(f)

    sound_speed = pyramic_tdoa['sound_speed_mps']
    delta = sound_speed * np.array(pyramic_tdoa['delays'])
    shape = delta.shape[1:]
    delta = delta.reshape((delta.shape[0], -1))
    mic_ref = pyramic_tdoa['mic_ref']
    mic_array = np.array(pyramic_tdoa['microphones'])
    sources = np.array(pyramic_tdoa['sources'])

    # mask a few outliers
    I = np.where(np.abs(delta) > 1)
    mask = np.ones(delta.shape)
    mask[I] = 0

    ### STEP 2 : start from the known locations of microphones and sources ###
    mic_no_ref = list(range(mic_array.shape[1]))
    mic_no_ref.remove(mic_ref)

    X0 = mic_array[:,mic_no_ref] - mic_array[:,mic_ref,np.newaxis]
    P0 = sources.reshape((sources.shape[0], -1))

    # we replace missing values by estimates from measured parameters
    for m,n in zip(*I):
        print('mic num:', m, 'direction:', np.degrees(P0[:,n]))
        delta[m,n] = np.inner(X0[:,m], -P0[:,n])

    if args.method == 'gd':
        # Direct method
        X, P, convergence_curve = joint_calibration_gd(delta,
                mask=mask, 
                gd_step_size=0.003, gd_n_steps=10000,
                X=X0, P=P0,
                enable_convergence_curve=True, verbose=True)

    if args.method == 'sgd':
        # Direct method
        X, P, convergence_curve = joint_calibration_sgd(delta,
                mask=mask, 
                gd_step_size=0.005, gd_n_steps=1500,
                X=X0, P=P0,
                enable_convergence_curve=True, verbose=True)

    elif args.method == 'svd':
        # Thrun's method
        X, P, convergence_curve = structure_from_sound(-delta,
                gd_step_size=1e-1, gd_n_steps=2000, stop_tol=1e-10,
                enable_convergence_curve=True, verbose=True)

    # We realign with the manually measured microphone locations
    u,s,v = np.linalg.svd(np.dot(X0, X.T))
    R = np.dot(u,v)
    X = np.dot(R, X)
    P = np.dot(R, P)

    # compute init/final error
    err_vec = mask * (np.dot(X0.T, P0) + delta)
    err = np.sqrt(np.mean(err_vec**2))
    print('The initial error:', err)
    err_vec = mask * (np.dot(X.T, P) + delta)
    err = np.sqrt(np.mean(err_vec**2))
    print('The final error:', err)

    # Post process microphone location
    if mic_ref != 0:
        raise ValueError('mic ref not zero')
    mic_array_cal = np.concatenate((np.zeros((3,1)), X), axis=1) + \
                    mic_array[:,mic_ref,np.newaxis]

    # center arrays
    mic_array_cal -= mic_array_cal.mean(axis=1)[:,np.newaxis]
    mic_array -= mic_array.mean(axis=1)[:,np.newaxis]


    # Plot stuff
    plt.figure()
    plt.semilogy(convergence_curve)
    plt.title('Convergence')

    sources = pra.experimental.PointCloud(X=P0)
    sources_cal = pra.experimental.PointCloud(X=P)

    mics = pra.experimental.PointCloud(X=mic_array)
    mics_cal = pra.experimental.PointCloud(X=mic_array_cal)

    ax = mics.plot()
    mics_cal.plot(show_labels=False, axes=ax, c='r')
    ax.set_aspect('equal')

    max_range = 0.5 * (X.max() - X.min())
    mid = (P.max(axis=1) + P.min(axis=1)) * 0.5
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax2 = sources.plot(show_labels=False)
    sources_cal.plot(show_labels=False, axes=ax2, c='r')
    ax2.set_aspect('equal')
    ax.set_aspect('equal')

    max_range = 0.5 * (P.max() - P.min())
    mid = (P.max(axis=1) + P.min(axis=1)) * 0.5
    ax2.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax2.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax2.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.figure()

    r0, col0, az0 = cart2sph(*P0)
    I = np.argsort(az0[::3])
    col0 = col0.reshape(shape)[I,:]
    az0 = az0.reshape(shape)[I,:]
    r, col, az = cart2sph(*P)
    col = col.reshape(shape)[I,:]
    az = az.reshape(shape)[I,:]

    plt.subplot(1,2,1)
    plt.plot(np.degrees(az0), np.degrees(col))
    plt.plot(np.degrees(az0), np.degrees(col0))
    plt.legend(['high', 'low', 'middle'])
    plt.xlabel('colatitude set')
    plt.ylabel('error')

    plt.subplot(1,2,2)
    plt.plot(np.degrees(az0), np.degrees(az - az0))
    plt.legend(['high', 'low', 'middle'])
    plt.xlabel('azimuth set')
    plt.ylabel('error')


    plt.show()
