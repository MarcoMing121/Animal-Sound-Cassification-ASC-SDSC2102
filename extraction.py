import numpy as np
import librosa
import noisereduce as nr
import librosa.display
import pandas as pd

import AnimalDetect as ad

FRAME_SIZE = 2048
HOP_LENGTH = 128
sr = 22050

df = pd.DataFrame.from_dict({
    'filename':[],
    'duration':[],
    'label':[],
    'AUC':[],
    'max_freq':[], 
    'spectral_centroid':[],
    'band_energy_ratio_800':[],
    'band_energy_ratio_1600':[],
    'one_ratio_band_energy_freq':[],
    'power_bandwidth':[],
    'kurtosis':[], 
    'skewness':[], 
    'zcr':[], 
    'spectral_entropy':[],
    'spectral_skewness':[],
    'spectral_spread':[],
    'fundamental_frequency':[], 
    'interquartile_range':[],
    'turning_rate':[],
    'mfcc1':[], 
    'mfcc2':[], 
    'mfcc3':[], 
    'mfcc4':[], 
    'mfcc5':[]
    })

audio_dir = 'Audio_processing/Animal/esc50-dataset/dataset/'
metadata_file = 'Audio_processing/Animal/esc50-dataset/esc50.csv'
animal_labels = ['dog', 'cat', 'rooster', 'hen', 'pig', 'frog', 'cow', 'crow']
metadata = pd.read_csv(metadata_file)
animal_metadata = metadata[metadata['category'].isin(animal_labels)]
filename = animal_metadata['filename'].tolist()
animal_files = pd.DataFrame({'animal': animal_metadata['category'], 'file': filename})
animal_files.reset_index(drop=True, inplace=True)

for i in range(0, len(animal_files)):
    x, sr = librosa.load(audio_dir + animal_files.iloc[i]['file'])
    reduced_noise = nr.reduce_noise(y=x, sr=sr)
    start_time, end_time, clips = ad.onset_detection(reduced_noise, False, False, 2048, 128, 20, 10)
    clips = ad.remove_clips(clips)
    # Second cliping
    for j in range(len(clips)):
        clips[j] = ad.second_cliping(clips[j], SIGMA = .5, SHOW_PLOT = False)
        clip = clips[j]
        features = ad.extract_features(clip, sr, animal_files.iloc[i]['file'], animal_files.iloc[i]['animal'])
        print(features)
        df.loc[len(df)] = features

df.to_csv('animal_features.csv', index=False)



