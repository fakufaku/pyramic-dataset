from __future__ import division, print_function

import argparse, os, re, sys
import numpy
from scipy.io import wavfile
from samplerate import resample

basedir = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(basedir)
import ffmpeg_audio

# need that when no display is available
if 'DISPLAY' not in os.environ:
    import matplotlib as mpl
    mpl.use('Agg')
    need_agg = True
else:
    need_agg = False

from matplotlib import pyplot

# import reference file globally
path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
all_samples_file = path + '/all_samples.wav'
fs, all_samples = wavfile.read(all_samples_file)

# these are the lengths in samples of all the files concatenated during playback
# This list is printed by the stitch script when 'all_samples.wav' is created.
lengths= [48000, 144000, 1500, 144000, 1500, 144000, 1500, 151451, 
        1500, 113357, 1500, 106598, 1500, 152987, 1500, 190463, 48000]

# these are the names of the individual segments in the order they were played
labels = ['silence', 'sweep_lin', 'sweep_exp', 'noise', 'fq_sample4', 
        'fq_sample3', 'fq_sample2', 'fq_sample1', 'fq_sample0']

# filename pattern
pattern = re.compile('pyramic_spkr[012]_all_samples_\d+.[wav|tta]')

def compute_cost(audio, noise_mean, noise_est, noise_thresh):
    import numpy
    return numpy.mean(numpy.abs(audio - noise_mean[None,:]), axis=1) - noise_thresh * noise_est

def find_segments(audio, lengths, noise_mean, noise_est, noise_thresh):
    import numpy

    # end recursion
    if len(lengths) < 2:
        return []

    # find beginning of segment
    cost = compute_cost(audio[:lengths[0]+lengths[1],:], noise_mean, noise_est, noise_thresh)
    start_offset = int(numpy.min( numpy.where(cost > 0)[0] ))

    # find end of segment
    L = start_offset + lengths[1]
    cost = compute_cost(audio[L:L+lengths[2]*5,:], noise_mean, noise_est, noise_thresh)
    end_offset = 0
    q = cost > 0
    chunk = 50
    jump = 25
    while numpy.sum(q[end_offset:end_offset+chunk]) > 0:
        end_offset += jump

    end_offset += L

    # recursively search segments
    beyond = [x + end_offset for x in find_segments(audio[end_offset:,:], lengths[2:], noise_mean, noise_est, noise_thresh)]

    return [start_offset, end_offset] + beyond

def open_segment(filename, noise_thresh=3, off_lo=50, off_hi=50, plot=False):
    global lengths, labels

    import re, os
    import numpy
    from scipy.io import wavfile

    from matplotlib import pyplot

    # the file to segment
    rate, audio = wavfile.read(filename)

    # find offset here
    noise_mean = numpy.mean(audio[:500, :], axis=0)
    noise_est = numpy.mean(numpy.std(audio[:500,:] - noise_mean[None,:], axis=0))

    # find the boundary of first sweep, with length[1]
    boundaries = [0] + find_segments(audio, lengths, noise_mean, noise_est, noise_thresh)

    # now extract
    signals = [audio[:boundaries[1]-off_lo,:]]

    # list of views
    for i in range(1, len(boundaries)-1, 2):
        b_lo, b_hi = boundaries[i], boundaries[i+1]
        signals.append(audio[b_lo-off_lo:b_hi+off_hi])

    # make a dictionary
    d = dict(zip(labels, signals))
    if plot:
        for label, signal in d.items():
            pyplot.figure()
            pyplot.plot(signal[:,0])
            pyplot.title(label)
        pyplot.show()

    # add the rate
    d['rate'] = rate

    return d

def open_segment_rigid(filename, noise_thresh=3, off_lo=150, off_hi=150, plot=False):
    global lengths, labels, basedir

    import re, os, sys
    import numpy
    from scipy.io import wavfile

    from matplotlib import pyplot

    sys.path.append(basedir)
    import ffmpeg_audio

    # the file to segment
    if os.path.splitext(filename)[1] == '.wav':
        rate, audio = wavfile.read(filename)
    else:
        rate, audio = ffmpeg_audio.read(filename)

    # find offset here
    noise_mean = numpy.mean(audio[:500, :], axis=0)
    noise_est = numpy.mean(numpy.std(audio[:500,:] - noise_mean[None,:], axis=0))

    # recursive code (method 2)
    boundaries = [0] + find_segments(audio, lengths[:3], noise_mean, noise_est, noise_thresh)
    boundaries[2] = boundaries[1] + lengths[1]

    for i in range(2, len(lengths)-1):
        boundaries.append(boundaries[-1] + lengths[i])

    # now extract
    signals = [audio[:boundaries[1]-off_lo,:]]

    # list of views
    for i in range(1, len(boundaries)-1, 2):
        b_lo, b_hi = boundaries[i], boundaries[i+1]
        signals.append(audio[b_lo-off_lo:b_hi+off_hi])

    # make a dictionary
    d = dict(zip(labels, signals))
    if plot:
        for label, signal in d.items():
            pyplot.figure()
            pyplot.plot(signal[:,0])
            pyplot.title(label)
        pyplot.show()

    # add the rate
    d['rate'] = rate

    return d


def save_samples(filename):
    global pattern, output_dir, qc_images

    import re, os
    import numpy
    from scipy.io import wavfile
    from samplerate import resample

    from matplotlib import pyplot



    _, fname = os.path.split(filename)

    if not pattern.match(fname):
        return None

    spkr, angle = [int(i) for i in re.findall(r'\d+', fname)]
    signals = open_segment_rigid(filename, noise_thresh=3, off_lo=150, off_hi=150, plot=False)

    out_name = '{}_spkr{}_angle{}.{}'

    for name, signal in signals.items():

        if name == 'rate':
            continue

        if 'fq_sample' in name:
            # we resample the speech signals to 16kHz because this was the
            # orignal rate of the samples played 
            # this saves some storage and load speed is faster
            signal = resample(signal, 16000. / signals['rate'], 'sinc_best')
            rate = 16000
        else:
            rate = signals['rate']

        # save in folder with name of the sample
        folder = os.path.join(output_dir, name)
        if not os.path.exists(folder):
            os.mkdir(folder)

        # format filename with angle and speaker location
        filename = out_name.format(name, spkr, (360 - angle) % 360, 'wav')

        signal_float = signal.astype(numpy.int16)
        wavfile.write(os.path.join(folder, filename), rate, signal_float)

        if qc_images:
            # save a spectrogram for later inspection

            folder = os.path.join(output_dir, 'qc_images', name)
            if not os.path.exists(folder):
                os.mkdir(folder)

            filename = out_name.format(name, spkr, (360 - angle) % 360, 'png')

            pyplot.specgram(signal[:,0].astype(numpy.float32) / (2**15+1), Fs=rate, NFFT=1024, noverlap=512)
            pyplot.savefig(os.path.join(folder, filename))
            pyplot.clf()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Segments files')
    parser.add_argument('path', type=str,
                        help='path to file to segment, or directory containing all files when using the option --all')
    parser.add_argument('-o', '--output', type=str, default='segmented',
                        help='output folder for the segemented recordings')
    parser.add_argument('-a', '--all', action='store_true',
                        help='segment all files in a directory')
    parser.add_argument('-q', '--qc', action='store_true',
                        help='produce spectrograms of all samples for quality control')

    args = parser.parse_args()

    # create necessary directories if not existing
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    if args.qc and not os.path.exists(args.output + '/qc_images'):
        os.mkdir(args.output + '/qc_images')

    if args.all:

        if not os.path.isdir(args.path):
            raise ValueError('When segmenting multiple files, the argument should be a folder name')
            sys.exit()

        print('All files in {} will be segmented'.format(args.path))

        import ipyparallel as ip

        # Start the parallel processing
        c = ip.Client()
        NC = len(c.ids)
        print(NC,'workers on the job')

        # need that because no display is available
        if need_agg:
            c[:].map_sync(mpl.use, ['Agg'] * len(c))

        # push necessary global variables
        c[:].push(
                dict(
                    pattern=pattern, 
                    output_dir=args.output, 
                    qc_images=args.qc, 
                    lengths=lengths, 
                    labels=labels, 
                    open_segment=open_segment,
                    open_segment_rigid=open_segment_rigid,
                    find_segments=find_segments,
                    compute_cost=compute_cost,
                    basedir=basedir,
                    ),
                block=True
                )
        print('pushing global variables to engines.')

        # all the files to process
        all_filenames = [fname for fname in os.listdir(args.path) if fname[0] != '.']

        # delegate to workers
        print('Dispatching {} files to engines.'.format(len(all_filenames)))
        try:
            c[:].map_sync(save_samples, [args.path + '/' + fname for fname in all_filenames])
        except:
            c.abort(block=True)

    else:
        open_segment_rigid(args.path, plot=True)
        output_dir = 'segmented'
        qc_images = True
        save_samples(args.path)
