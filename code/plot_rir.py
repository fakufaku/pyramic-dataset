
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile


if __name__ == '__main__':

    angles = np.arange(0, 358, 2)
    height = 0  # middle

    filename = lambda a : './pyramic_ir/ir_spkr0_angle{}.wav'.format(a)

    _, data = wavfile.read(filename(0))

    rir_matrix = np.zeros((data.shape[0], angles.shape[0]))

    sns.set_context('paper')

    # First figure: a single rir
    plt.figure(figsize=(3.38846 / 2, 1.))
    plt.plot(data[:,0])
    sns.despine(offset=5)
    
    # Second figure: a full rotation of the array
    plt.figure()
    for i,angle in enumerate(angles):
        _, data = wavfile.read(filename(angle))
        rir_matrix[:,i] = data[:,0]

    plt.imshow(rir_matrix)

    plt.show()

