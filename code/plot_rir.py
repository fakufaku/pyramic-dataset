
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile


if __name__ == '__main__':

    angles = np.arange(0, 358, 2)
    height = 0  # middle

    filename = lambda a : './pyramic_ir/ir_spkr0_angle{}.wav'.format(a)

    fs, data = wavfile.read(filename(10))
    time = np.arange(data.shape[0]) / fs

    sns.set_context('paper')

    # First figure: a single rir
    plt.figure(figsize=(3.38846 / 2, 1.))
    plt.plot(time * 1000, data[:,0])
    plt.xlim([0, 20])
    plt.xticks([0, 5, 10, 15, 20], fontsize='x-small')
    plt.yticks([])
    plt.xlabel('Time [ms]', fontsize='x-small')
    sns.despine(offset=5, left=True)
    plt.tight_layout(pad=0.1)

    plt.savefig('rir.pdf')
    
    plt.show()

