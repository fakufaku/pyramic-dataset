'''
Run Experiment
==============

This is the script automating the recording of the dataset.
Please find more info in the PROTOCOL.md file

Author: 2018 (c) Robin Scheibler
License: MIT License
'''

import sys, os, datetime
import numpy as np
import argparse
import sounddevice as sd
import time
from scipy.io import wavfile

#import pythonaudio as pa

'''
GLOBAL CONFIG
'''
EASYDSP_PATH = '/Users/scheibler/PHD/Projects/Hardware/easy-dsp'
TURNTABLE_PATH = '/Volumes/DATA/TurntableDriver'

TURNTABLE_DEVICE_NAME = '???'
AUDIO_DEVICE_OUT = '???'
AUDIO_DEVICE_IN = '192.168.2.11'

right_now = datetime.datetime.now()
RECORD_FOLDER = 'recordings-{}'.format(right_now.strftime('%Y%m%d-%H%M%S'))
RECORD_FILENAME = '{array}_spkr{spkr}_{sample}_{angle}.wav'

SAMPLES_FOLDER = '.'
SAMPLES_FILENAME = 'samples/all_samples.wav'
SPEAKERS = [0, 1, 2]
# SPEAKERS = [0]  # this was set like this to record 310 only from spkr 0

SAMPLING_FREQUENCY = 48000
BUFFER_SIZE = 9600
MIC_VOLUME = 100

# import local files
sys.path.append(EASYDSP_PATH)
sys.path.append(TURNTABLE_PATH)

from turntable import TurnTable
import browserinterface

# use this to communicate file info to save_audio callback
current_record = dict(sample='', array='', spkr=0, duration=0, angle=0, done=False)

# Record call back func
def save_audio(buf):
    global current_record, SAMPLING_FREQUENCY, RECORD_FOLDER, RECORD_FILENAME
    filename = '/'.join([RECORD_FOLDER, RECORD_FILENAME.format(**current_record)])
    wavfile.write(filename, SAMPLING_FREQUENCY, buf)
    print(filename, 'done with', len(buf), 'samples')
    current_record['done'] = True
    
''' Main starts here '''
if __name__ == "__main__":

    # parse CLI arguments
    parser = argparse.ArgumentParser(description='Perform measurements for DOA evaluation')
    parser.add_argument('ip', type=str,
                        help='ip address of the microphone array')
    parser.add_argument('-a', '--array', type=str, default='pyramic',
                        help='microphone array to use [pyramic or compactsix]')
    parser.add_argument('-r', '--angles', type=str, default='1',
                        help='range of angles, can be a slice <begin>:<end>:<step>')
    parser.add_argument('-s', '--simulate', action='store_true',
                        help='simulate turntable movement')
    parser.add_argument('-o', '--output_device', type=int, default=2,
                        help='output device number')
    parser.add_argument('-t', '--turntable', type=str, default='ASRL4::INSTR',
                        help='turntable instrument identifier')
    args = parser.parse_args()

    # setup the output device
    sd.default.device = args.output_device
    sd.default.channels = (8, len(SPEAKERS))

    # fix number of channels
    if args.array == 'pyramic':
        channels = 48
    elif args.array == 'compactsix':
        channels = 6
    else:
        raise ValueError('Array can be ''pyramic'' or ''compactsix''')

    # angle range
    try:
        angle_range = [int(_) for _ in args.angles.split(':')]
    except:
        raise ValueError('Angles should be a slice <begin>:<end>:<step>')

    ''' Do all the good stuff here '''

    # create the folder to save all the recordings
    os.mkdir(RECORD_FOLDER)

    # init turntable
    if not args.simulate:
        table = TurnTable(args.turntable)

    # setup sounddevice

    # init the browser interface
    browserinterface.inform_browser = False
    browserinterface.bi_board_ip = args.ip
    browserinterface.start()
    browserinterface.change_config(
            rate=SAMPLING_FREQUENCY, 
            channels=channels, 
            buffer_frames=BUFFER_SIZE, 
            volume=MIC_VOLUME
            )

    # open the audio file
    filename = '/'.join([SAMPLES_FOLDER, SAMPLES_FILENAME])
    rate, audio = wavfile.read(filename)

    audio = np.concatenate([np.zeros(rate//2), audio])

    # set interface rate
    sd.default.samplerate = rate

    # now loop for all the angles
    for angle in range(*angle_range):

        # position turntable
        print('Setting angle', angle)
        if not args.simulate:
            table.turn_abs(angle)
            time.sleep(2.0)

        for spkr in SPEAKERS:

            # prepare the sample
            current_record['sample'] = os.path.splitext(SAMPLES_FILENAME)[0]
            current_record['array'] = args.array
            current_record['spkr'] = spkr
            current_record['duration'] = audio.shape[0] / rate * 1000.
            current_record['angle'] = angle
            current_record['done'] = False

            # start a new recording
            browserinterface.record_audio(current_record['duration'], save_audio)

            # play the sound sample
            sd.play(audio, samplerate=rate, mapping=[spkr+1])
            status = sd.wait()

            # and wait for the recording to finish
            while not current_record['done']:
                # there should be only one call back coming: save_audio
                browserinterface.process_callbacks()

            time.sleep(0.5)

    # go back to zero angle
    if not args.simulate:
        table.turn_abs(0)
