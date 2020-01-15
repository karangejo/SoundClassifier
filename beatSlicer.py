import librosa
import numpy as np
import os
import subprocess

file = '/home/karang/Documents/AudioClassification/beatbox.wav'
slice_folder = '/home/karang/Documents/AudioClassification/slices/'

# function that uses the aubio-tools command aubiocut
# to beat slice a .wav file and save slices to a folder
def aubio_cut_subprocess(file,slice_folder):
    output = subprocess.check_output(["aubiocut","-i",file,'-c','-o',slice_folder],universal_newlines=True)
    return(output)

# use librosa to slice a .wav file
# does not work as well as the code above
def slice_audio_sample(file,slice_folder):

    y, sr = librosa.load(file)
    #o_env = librosa.onset.onset_strength(y, feature=librosa.feature.zero_crossing_rate)
    #o_env = librosa.onset.onset_strength(y, sr=sr, feature=librosa.feature.chroma_cqt)
    o_env = librosa.onset.onset_strength(y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)

    onset_samples = list(librosa.frames_to_samples(onset_frames))
    onset_samples = np.concatenate(onset_samples, len(y))
    starts = onset_samples[0:-1]
    stops = onset_samples[1:]


    for i, (start, stop) in enumerate(zip(starts, stops)):
            audio = y[start:stop]
            filename = str(i) + '.wav'
            librosa.output.write_wav(slice_folder + filename, audio, sr)
