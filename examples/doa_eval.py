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

def main_run(args):

    with open(args.calibration_file, 'r') as f:
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

    with open(args.output, 'w') as f:
        json.dump(results, f)


def main_plot(args):
    ''' Plot the result of the Evaluation '''

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_json(args.result)

    # Compute the average error with the grid used
    grid = pra.doa.GridSphere(n_points=30000)
    avg_error = np.degrees(grid.min_max_distance()[2])

    # remove the location columns, only keep error
    df = df[['algo', 'angle', 'spkr_height', 'error_man', 'error_opt']]
    df2 = pd.melt(df, value_vars=['error_man', 'error_opt'],
            value_name='Error', var_name='Calibration',
            id_vars=['algo','angle','spkr_height'])

    df2['Error'] = df2['Error'].apply(np.degrees)
    df2['Calibration'] = df2['Calibration'].replace({'error_man' : 'Manual', 'error_opt' : 'Optimized'})

    # Ignore WAVES as the algorithm does not work so well
    df2 = df2[df2.algo != 'WAVES']
    df2 = df2.rename(index=str, columns={'algo' : 'Algorithms'})

    sns.set_style('white')
    sns.set_context('paper')

    fig, ax = plt.subplots(figsize=(3.39, 1.6))

    pal = sns.cubehelix_palette(2, start=-0.5, rot=0.1, dark=0.4, light=.7, reverse=True)

    sns.boxplot(ax=ax, x="Algorithms", y="Error", hue="Calibration", data=df2, palette=pal, whis=[5, 95])
    plt.legend(framealpha=0.8, frameon=True, loc='upper right')
    plt.xlabel('')
    plt.ylabel('Error [$^\circ$]')
    plt.axhline(y=avg_error)
    sns.despine(ax=ax, offset=5)
    plt.tight_layout(pad=0.1)

    if args.save is not None:
        plt.savefig(args.save, dpi=150)

    plt.show()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a few DOA algorithms")
    subparsers = parser.add_subparsers()

    parser_run = subparsers.add_parser('run', description='Run the DOA algorithms on the measurements')
    parser_run.set_defaults(func=main_run)
    parser_run.add_argument('calibration_file', type=str, help='The JSON file containing the calibrated locations')
    parser_run.add_argument('--output', '-o', type=str, default='doa_results.json', help='The JSON file where to save the results')

    parser_plot = subparsers.add_parser('plot', description='Plot the results of the evaluation')
    parser_plot.set_defaults(func=main_plot)
    parser_plot.add_argument('result', type=str, help='The JSON file containing the results')
    parser_plot.add_argument('-s', '--save', metavar='FILE', type=str, help='Save plot')

    args = parser.parse_args()
    args.func(args)

