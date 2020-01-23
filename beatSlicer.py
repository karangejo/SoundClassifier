import librosa
import numpy as np
import os
import subprocess
import random
import soundfile as sf

file = '/home/karang/Documents/AudioClassification/beatbox.wav'
slice_folder = '/home/karang/Documents/AudioClassification/slices/'

# function that uses the aubio-tools command aubiocut
# to beat slice a .wav file and save slices to a folder
# returns the list of times (in seconds) where cuts were made
def aubio_cut_subprocess(file,slice_folder,threshold):
    output = subprocess.check_output(["aubiocut","-i",file,'-c','-t',threshold,'-o',slice_folder]).splitlines()
    # decode the bytes into string and then convert to float
    decoded_output = [float(x.decode()) for x in output]
    return(decoded_output)

def aubio_onset_subprocess(file,threshold):
    output = subprocess.check_output(["aubioonset","-i",file,'-t',threshold]).splitlines()
    # decode the bytes into string and then convert to float
    decoded_output = [float(x.decode()) for x in output]
    return(decoded_output)


# use librosa to slice a .wav file
def slice_audio_sample(file,slice_folder):

    y, sr = librosa.load(file)
    # this one uses a precalculated envelope
    o_env = librosa.onset.onset_strength(y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, backtrack=True)
    num_cuts = len(onset_frames) -1

    onset_samples = list(librosa.frames_to_samples(onset_frames))
    onset_samples = np.concatenate(onset_samples, len(y))
    starts = onset_samples[0:-1]
    stops = onset_samples[1:]


    for i, (start, stop) in enumerate(zip(starts, stops)):
            audio = y[start:stop]
            filename = str(i) + '.wav'
            librosa.output.write_wav(slice_folder + '/' + filename, audio, sr)
    return(num_cuts)

# same as above except here we do not use a precalculated envelope
def slice_audio_sample_no_env(file,slice_folder):

    y, sr = librosa.load(file)

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    num_cuts = len(onset_frames) -1

    onset_samples = list(librosa.frames_to_samples(onset_frames))
    onset_samples = np.concatenate(onset_samples, len(y))
    starts = onset_samples[0:-1]
    stops = onset_samples[1:]


    for i, (start, stop) in enumerate(zip(starts, stops)):
            audio = y[start:stop]
            filename = str(i) + '.wav'
            librosa.output.write_wav(slice_folder + '/' + filename, audio, sr)
    return(num_cuts)


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

def join_random_samples_from_dataset(dataset_path,number_of_joins):
    # get a list of all files from dataset directory
    wav_files = getListOfFiles(dataset_path)
    # take a random sample depending on the number of joins desired
    selected_wavs = random.sample(wav_files, number_of_joins)

    # loop through and load all the files into np.arrays
    outfiles = []
    sr = 0
    for file in selected_wavs:
        X, sample_rate = librosa.load(file)
        sr = sample_rate
        outfiles.append(X)

    # concatenate the loaded files and write them to disk
    joined_file = np.concatenate(outfiles, axis=0)
    sf.write('joined.wav',joined_file,sr)
