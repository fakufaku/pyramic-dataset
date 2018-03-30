'''
Uncompress
==========

Read a tar file containing compressed audio files and directly write the
uncompressed audio files in a new folder.

Author: 2018 (c) Robin Scheibler
License: MIT License
'''

import numpy as np
import tarfile, argparse
import subprocess as sp
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            description='Uncompress a bunch of TTA recordings that are in a TAR archive.')
    parser.add_argument('archive', type=str, help='The archive filename')
    parser.add_argument('output_dir', type=str, default='recordings', help='Output directory')
    parser.add_argument('-f', '--force', action='store_true', help='Overwrite output file if existing')
    args = parser.parse_args()

    # configuration
    out_dir = args.output_dir

    # audio information
    fs = 48000
    n_channels = 48
    dtype = np.int16


    ### START ###
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    FFMPEG_BIN = 'ffmpeg'
    command = [ FFMPEG_BIN,
            '-f', 'tta',        # input format
            '-i', '-',          # The imput comes from a pipe
            '-f', 'wav',  # output format
            ''                 # output filename here
            ]

    archive = tarfile.open(args.archive, 'r')

    filenames = archive.getnames()

    chunksize = 1024 * 1024  # read 1MB of data at a time

    for filename in filenames:


        if not filename.lower().endswith('tta'):
            print('Skip', filename)
            continue
        else:
            print(filename)

        f = archive.extractfile(filename)

        outname = out_dir + '/' + os.path.splitext(os.path.basename(filename))[0] + '.wav'

        if not args.force and os.path.exists(outname):
            raise ValueError('File {} exists already. Use -f to overwrite.'.format(outname))

        command[-1] = outname

        with sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE, bufsize=5*chunksize) as pipe:
            raw_data = f.read()
            pipe.communicate(input=raw_data)

    print('Done!!')

