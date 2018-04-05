'''
Evaluate Direction of Arrival Algorithms
========================================

This example evaluate the performance of three direction of arrival (DOA)
algorithms on the recorded samples. It compares the discrepancy between the
output of the DOA algorithm and the calibrated locations (manual and
optimized).

The script requires `numpy`, `scipy`, `pyroomacoustics`, and `joblib` to run.

The three DOA algorithms are `MUSIC` [1], `SRP-PHAT` [2], and `WAVES` [3].

References
----------

.. [1] R. Schmidt, *Multiple emitter location and signal parameter estimation*, 
    IEEE Trans. Antennas Propag., Vol. 34, Num. 3, pp 276--280, 1986

.. [2] J. H. DiBiase, *A high-accuracy, low-latency technique for talker localization 
    in reverberant environments using microphone arrays*, PHD Thesis, Brown University, 2000

.. [3] E. D. di Claudio, R. Parisi, *WAVES: Weighted average of signal subspaces for 
    robust wideband direction finding*, IEEE Trans. Signal Process., Vol. 49, Num. 10, 
    2179--2191, 2001
'''
import json
import numpy as np
from scipy.io import wavfile
import pyroomacoustics as pra
from joblib import Parallel, delayed

# We choose the sample to use
sample_name = 'fq_sample0'
fn = 'segmented/{name}/{name}_spkr{spkr}_angle{angle}.wav'

fs = 16000  # This needs to be changed to 48000 for 'noise', 'sweep_lin', 'sweep_exp'
nfft = 256
stft_hop = 256

# We use a spherical grid with 30000 points
algorithms = {
        "SRP-PHAT" : { "algo_obj" : "SRP",   "n_grid" : 30000 },
        "MUSIC" :    { "algo_obj" : "MUSIC", "n_grid" : 30000 },
        "WAVES" :    { "algo_obj" : "WAVES", "n_grid" : 30000, "num_iter" : 10 },
        }

locate_kwargs = {
        "MUSIC" : {     "freq_range" : [1500.0, 3500.0], "n_bands" : 20 },
        "SRP-PHAT" : {  "freq_range" : [1500.0, 3500.0], "n_bands" : 20 },
        "WAVES" : {     "freq_range" : [1500.0, 3500.0], "n_bands" : 20 },
        }

def run_doa(angle, h, algo, doa_kwargs, freq_bins, speakers_numbering):
    ''' Run the doa localization for one source location and one algorithm '''

    # Prepare the DOA localizer object
    algo_key = doa_kwargs['algo_obj']
    doa = pra.doa.algorithms[algo_key](mic_array, fs, nfft, c=c, num_src=1, dim=3, **doa_kwargs)

    # get the loudspeaker index from its name
    spkr = speakers_numbering[h]

    # open the recording file
    filename = fn.format(name=sample_name, spkr=spkr, angle=angle)
    fs_data, data = wavfile.read(filename)

    if fs_data != fs:
        raise ValueError('Sampling frequency mismatch')

    # do time-freq decomposition
    X = np.array([ 
        pra.stft(signal, nfft, stft_hop, transform=np.fft.rfft).T 
        for signal in data.T ])

    # run doa
    doa.locate_sources(X, freq_bins=freq_bins)
    col = float(doa.colatitude_recon[0])
    az = float(doa.azimuth_recon[0])

    # manual calibration groundtruth
    col_gt_man = locations['speakers_manual_colatitude'][h]
    az_gt_man = np.radians(int(angle))
    error_man = pra.doa.great_circ_dist(1., col, az, col_gt_man, az_gt_man)

    # optimized calibration groundtruth
    col_gt_opt = locations['sources'][h]['colatitude'][angle]
    az_gt_opt = locations['sources'][h]['azimuth'][angle]
    error_opt = pra.doa.great_circ_dist(1., col, az, col_gt_opt, az_gt_opt)

    print(algo, h, angle, ': Err Man=', error_man, 'Opt=', error_opt)

    return {
            'algo' : algo,
            'angle' : angle,
            'spkr_height' : h,
            'loc_man' : (col_gt_man, az_gt_man),
            'loc_opt' : (col_gt_opt, az_gt_opt),
            'loc_doa' : (col, az),
            'error_man' : float(error_man),
            'error_opt' : float(error_opt),
            }


if __name__ == '__main__':

    with open('calibration/calibrated_locations.json', 'r') as f:
        locations = json.load(f)

    c = locations['sound_speed_mps']

    # microphone locations
    mic_array = np.array(locations['microphones']).T

    # Recover the list of all sources locations
    spkr_azimuths = list(locations['sources']['low']['azimuth'].keys())
    spkr_height = list(locations['sources'].keys())

    errors = {
            'manual' : dict(zip(spkr_height, [[],[],[]])),
            'opt' : dict(zip(spkr_height, [[],[],[]]))
            }


    all_args = []
    for algo, doa_kwargs in algorithms.items():
        # select frequency bins uniformly in the range
        freq_hz = np.linspace(
            locate_kwargs[algo]['freq_range'][0],
            locate_kwargs[algo]['freq_range'][1],
            locate_kwargs[algo]['n_bands']
        )

        freq_bins = np.unique(
            np.array([int(np.round(f / fs * nfft))
                      for f in freq_hz])
        )

        # This will loop over all sources locations
        for h in spkr_height:
            for angle in spkr_azimuths:

                all_args.append(
                    (angle, h, algo, doa_kwargs, freq_bins, locations['speakers_numbering'])
                    )

    # Now run this in parallel with joblib
    results = Parallel(n_jobs=18)(delayed(run_doa)(*args) for args in all_args)

    with open('doa_results.json', 'w') as f:
        json.dump(results, f)



