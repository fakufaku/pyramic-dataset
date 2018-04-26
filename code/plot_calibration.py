
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':


    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a few DOA algorithms")
    parser.add_argument('calibration_file', type=str, help='The JSON file containing the calibrated locations')
    parser.add_argument('-s', '--save', metavar='FILE', type=str, help='Save plot')
    args = parser.parse_args()

    with open(args.calibration_file, 'r') as f:
        locations = json.load(f)

    c = locations['sound_speed_mps']

    # microphone locations
    mic_array = np.array(locations['microphones']).T

    # Recover the list of all sources locations
    spkr_azimuths = list(locations['sources']['low']['azimuth'].keys())
    spkr_height = ['low', 'middle', 'high']

    pal = sns.color_palette("PRGn", 7)
    pal = sns.dark_palette("muted purple", input="xkcd")
    pal = sns.cubehelix_palette(3, start=2, rot=0, dark=0, light=.5, reverse=True)

    sns.set_context('paper')
    plt.figure(figsize=(3.38846, 0.9))

    angles = np.arange(0, 360, 2)

    for i, height in enumerate(spkr_height):

        az = [np.degrees(locations['sources'][height]['azimuth'][str(a)]) % 360 for a in angles]
        az[0] -= 360.  # fix first element
        co = [np.degrees(locations['sources'][height]['colatitude'][str(a)]) for a in angles]

        man_co = np.ones(angles.shape) * np.degrees(locations['speakers_manual_colatitude'][height])

        plt.plot(angles, man_co, color=pal[i], linewidth=0.7)
        plt.plot(az, co, '.', markersize=1.75, color=pal[i], label='height')

    plt.gca().invert_yaxis()

    plt.xlabel('Azimuth', fontsize='x-small')
    plt.ylabel('Colatitude [$^\circ$]', fontsize='x-small')
    plt.xticks([])
    plt.yticks([np.round(np.degrees(locations['speakers_manual_colatitude'][h])) for h in spkr_height], fontsize='x-small')

    sns.despine(offset=2, bottom=True)

    plt.tight_layout(pad=0.1)

    if args.save is not None:
        plt.savefig(args.save, dpi=300)

    plt.show()


