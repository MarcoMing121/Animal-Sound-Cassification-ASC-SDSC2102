

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tsfel
import pandas as pd

__all__ = ["max_curve", 
           "rms_curve", 
           "moving_average", 
           "weighted_moving_average", 
           "min_max_signal", 
           "find_maxima", 
           "remove_close_maxima", 
           "find_end_minima", 
           "find_start_minima", 
           "remove_same_start_end", 
           "convert_to_time", 
           "delay_clip",
]

# -- Statistics function -- #

#compute maximum for each frame
def max_curve(y, frame_size, hop_length) -> np.ndarray:
    maxes = []
    y = y**2
    for i in range(0, len(y), hop_length):
        frame = y[i:i+frame_size]
        maxes.append(max(frame))
    return maxes

#compute root mean square for each frame
def rms_curve(y, frame_size, hop_length) -> np.ndarray:
    rms = []
    for i in range(0, len(y), hop_length):
        frame = y[i:i+frame_size]
        rms.append(np.sqrt(np.mean(frame**2)))
    return rms

#compute the moving average
def moving_average(signal, frame_length) -> np.ndarray:
    moving_average = [0]
    for i in range(0, len(signal)):
        frame = signal[i:i+frame_length]
        moving_average.append(np.mean(frame))
    return moving_average

#compute the weighted moving average
def weighted_moving_average(signal, frame_length) -> np.ndarray:
    weighted_moving_average = [0]
    for i in range(0, len(signal)):
        frame = signal[i:i+frame_length]
        weighted_moving_average.append(np.average(frame, weights=np.arange(1, len(frame)+1)))
    return weighted_moving_average

#perform min-max normalization
def min_max_signal(signal) -> np.ndarray:
    normalized_signal = []
    min_value = min(signal)
    max_value = max(signal)
    for i in range(0, len(signal)):
        normalized_signal.append((signal[i] - min_value) / (max_value - min_value))
    return normalized_signal

# -- Animal detection module. Step 1 -- #

# Finding all local maxima in the curve above mean
def find_maxima(signal) -> np.ndarray:
    local_maxima = []
    signal.append(0)
    for i in range(1, len(signal)-1):
        if signal[i] >= signal[i-1] and signal[i] >= signal[i+1] and signal[i] >= np.mean(signal):
            local_maxima.append(i)
    return local_maxima

#if two maxima are too close, keep only the one with the highest value
def remove_close_maxima(maxima, distance) -> np.ndarray:
    new_maxima = []
    for i in range(0, len(maxima)):
        if i == 0:
            new_maxima.append(maxima[i])
        else:
            if maxima[i] - maxima[i-1] > distance:
                new_maxima.append(maxima[i])
    return new_maxima

# Finding the first local minima after each local maxima as the end of the note
def find_end_minima(signal, maxima) -> np.ndarray:
    local_minima = []
    for i in maxima:
        for j in range(i+1, len(signal)-1):
            if signal[j] <= signal[j-1] and signal[j] <= signal[j+1] and signal[j] <= np.mean(signal):
                local_minima.append(j)
                break
    return local_minima

# Finding the first local minima before each local maxima as the start of the note
def find_start_minima(signal, maxima) -> np.ndarray:
    local_minima = []
    signal = np.insert(signal, 0, 0)
    for i in maxima:
        for j in range(i-1, 0, -1):
            if signal[j] <= signal[j-1] and signal[j] <= signal[j+1] and signal[j] <= np.mean(signal):
                local_minima.append(j)
                break
    return local_minima

#if there are two same start and end points pair, keep only one of them
def remove_same_start_end(start_minima, end_minima) -> np.ndarray:
    start = [start_minima[0]]
    end = [end_minima[0]]
    for i in range(1, len(start_minima)):
        if start_minima[i] - start_minima[i-1] > 5 or end_minima[i] - end_minima[i-1] > 5:
            start.append(start_minima[i])
            end.append(end_minima[i])
    return start, end

#convert the local maxima and minima to time
def convert_to_time(signal, maxima, minima, hop_length) -> np.ndarray:
    start_time = []
    end_time = []
    for i in range(len(maxima)):
        if maxima[i] < 5:
            start_time.append(0)
        else:
            start_time.append(max(0, int(maxima[i]*hop_length)))
    if start_time == []:
        start_time.append(0)
    for i in range(len(minima)):
        end_time.append(min(len(signal), int(minima[i]*hop_length)))
    return start_time, end_time

#if two segments are too close, delay the end of the first segment, and delay the start of the second segment
def delay_clip(start_time, end_time, distance=500, delay=2205) -> np.ndarray:
    new_start_time = []
    new_end_time = []
    for i in range(len(start_time)):
        if i == 0:
            new_start_time.append(start_time[i])
            new_end_time.append(end_time[i])
        else:
            if start_time[i] - end_time[i-1] < distance:
                new_start_time.append(start_time[i] + delay)
                new_end_time[i-1] += delay
                new_end_time.append(end_time[i])
            else:
                new_start_time.append(start_time[i])
                new_end_time.append(end_time[i])
    return new_start_time, new_end_time


    # Save the clips
def save_clips(start_time, end_time, sourse) -> np.ndarray:
    clips = []
    for i in range(len(start_time)):
        clips.append(sourse[start_time[i]:end_time[i]])
    return clips


def onset_detection(
        signal, 
        DEV_MODE=False, 
        SHOW_PLOT=False,
        FRAME_SIZE=2048, 
        HOP_LENGTH=128, 
        MOVING_SIZE=20, 
        MAXIMA_SIZE=10
) -> np.ndarray:
    
    #compute the RMS curve
    curve = rms_curve(abs(signal), FRAME_SIZE, HOP_LENGTH)
    curve = min_max_signal(curve)

    #compute the moving average of the RMS curve
    sc = moving_average(curve, MOVING_SIZE)
    sc.insert(0, 0)
    sc.append(0)
    maxima= find_maxima(sc)

    maxima = remove_close_maxima(maxima, MAXIMA_SIZE)
    end_minima= find_end_minima(sc, maxima)    
    start_minima = find_start_minima(sc, maxima)
        
    if len(start_minima) > len(end_minima):
        for i in range(0, len(start_minima)-len(end_minima)):
            end_minima.append(len(sc))
    start_minima, end_minima = remove_same_start_end(start_minima, end_minima)
    start_time, end_time = convert_to_time(signal, start_minima, end_minima, HOP_LENGTH)
    start_time, end_time = delay_clip(start_time, end_time)

    if DEV_MODE:

        plt.figure(figsize=(15, 5))
        plt.plot(sc)
        plt.plot(maxima, [sc[i] for i in maxima], 'ro')
        plt.title('maxima above mean')
        plt.axhline(np.mean(sc), color='r', linestyle='--', label='Mean')
        plt.show()
                
        plt.figure(figsize=(15, 5))
        plt.title('end_minima')
        plt.plot(sc)
        plt.plot(end_minima, [sc[i] for i in end_minima], 'ro')
        plt.show()

        plt.figure(figsize=(15, 5))
        plt.title('start_minima')
        plt.plot(sc)
        plt.plot(start_minima, [sc[i] for i in start_minima], 'ro')
        plt.show()

    if SHOW_PLOT:

        plt.figure(figsize=(15, 5))
        plt.figure(figsize=(15, 5))
        plt.plot(curve, label='Min-Max Normalized RMS Curve')
        plt.plot(sc, label='Moving Average', color='r')
        plt.legend()
        plt.show()
        print("Length of RMS curve: ", len(curve))

        plt.figure(figsize=(15, 5))
        plt.plot(sc, label='Moving Average', color='r')
        plt.axhline(np.mean(sc), color='r', linestyle='--', label='Mean')
        plt.scatter(maxima, [sc[i] for i in maxima], label='Maxima')
        plt.scatter(start_minima, [sc[i-1] for i in start_minima], label='start', color='g', marker='o')
        plt.scatter(end_minima, [sc[i-1] for i in end_minima], label='end', color='r', marker='x')
        plt.legend()
        plt.show()

    if SHOW_PLOT:

        #plot the start and end time
        plt.figure(figsize=(15, 5))
        plt.plot(signal)
        for i in range(len(start_time)):
            plt.axvspan(start_time[i], end_time[i], alpha=0.5, color='red')
        plt.show()

    clips = save_clips(start_time, end_time, signal)

    if SHOW_PLOT:
        #print the start and end time
        print('Start Sample Point: ', start_time)
        print('End Sample Point: ', end_time)

    return start_time, end_time, clips

# -- Animal detection module. Step 2 -- #

def second_cliping(
        signal,
        FRAME_SIZE = 1,
        SIGMA = 1.04,
        SHOW_PLOT = False
):
    # Compute the mean and standard deviation of the signal
    mean = np.mean(signal)
    std = np.std(signal)

    # Compute the mean and standard deviation for each frame
    means = [np.mean(signal[i:i+FRAME_SIZE]) for i in range(0, len(signal), FRAME_SIZE)]
    stds = [np.std(signal[i:i+FRAME_SIZE]) for i in range(0, len(signal), FRAME_SIZE)]

    UCL = mean + SIGMA*std/np.sqrt(FRAME_SIZE)
    LCL = mean - SIGMA*std/np.sqrt(FRAME_SIZE)

    onset = np.where(np.abs(means) >= UCL)
    onset_time = [i*FRAME_SIZE for i in onset[0]]
    start = onset[0][0]*FRAME_SIZE
    end = onset[0][-1]*FRAME_SIZE

    if SHOW_PLOT:

        #plot the change points
        plt.figure(figsize=(15, 5))
        plt.plot(signal)
        plt.plot([UCL]*len(means), label='UCL', color='r', linestyle='--')
        plt.plot([LCL]*len(means), label='LCL', color='r', linestyle='--')
        plt.vlines(onset_time, np.max(signal), np.min(signal), color='y', alpha=0.05)
        plt.vlines(start, np.max(signal), np.min(signal), color='g')
        plt.vlines(end, np.max(signal), np.min(signal), color='r')
        plt.legend()
        plt.show()
        print("start: ", start)
        print("end: ", end)

    clip = signal[start:end]

    return clip

# Remove the clips with energy less than the mean
def remove_clips(clips):

    sum_energy = []
    for i in range(len(clips)):
        sum_energy.append(np.sum(clips[i]**2))
    mean_energy = np.mean(sum_energy)
    std_energy = np.std(sum_energy)

    new_clips = []
    for i in range(len(sum_energy)):
        if sum_energy[i] > mean_energy-0.3*std_energy:
            new_clips.append(clips[i])
    return new_clips

# -- Fecture extraction module. -- #

def band_energy_ratio(signal, sr, split_freq, f_ratio=0.5):
    
    spectrum = np.abs(np.fft.fft(signal))

    f = np.linspace(0, sr, len(spectrum))
    f_range = int(len(spectrum)*f_ratio)
    df = pd.DataFrame({'frequency': f, 'magnitude': spectrum})
    df = df[df['frequency'] < f_range]
    df = df.reset_index(drop=True)

    LFE = df.magnitude[(df['frequency'] < split_freq)]
    HFE = df.magnitude[(df['frequency'] >= split_freq)]
    if sum(HFE**2) != 0:
        ratio = sum(LFE**2)/sum(HFE**2)
    else:
        ratio = None
    return ratio

def one_ratio_band_energy(signal, sr, f_ratio=0.5):

    spectrum = np.abs(np.fft.fft(signal))

    f = np.linspace(0, sr, len(spectrum))
    f_range = int(len(spectrum)*f_ratio)
    df = pd.DataFrame({'frequency': f, 'magnitude': spectrum})
    df = df[df['frequency'] < f_range]
    df = df.reset_index(drop=True)

    ORBER = 0
    for i in range(0, 8192):
        LFE = df.magnitude[(df['frequency'] < i)]
        HFE = df.magnitude[(df['frequency'] >= i)]
        if sum(HFE**2) != 0:
            ratio = sum(LFE**2)/sum(HFE**2)  
            if ratio > 1:
                ORBER = i
                break
        else:
            ratio = None

    return ORBER

#Find frequency with highest power in the signal
def max_freq(signal, sr):
    freqs = np.fft.fftfreq(len(signal), 1/sr)
    fft = np.fft.fft(signal)
    freq = freqs[np.argmax(np.abs(fft))]
    return abs(freq)

def compute_mel_frequencies(signal, sr):
    mel_frequencies = librosa.feature.melspectrogram(y=signal,sr=sr,n_fft=len(signal),hop_length=len(signal)+1)
    return mel_frequencies

def compute_mfccs(mel_frequencies, sr, n_mfcc=5):
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_frequencies),sr=sr, n_mfcc=n_mfcc)
    return mfccs

#Computes the area under the curve of the signal computed with trapezoid rule.
def AUC(y, sr):
    AUC = tsfel.feature_extraction.features.auc(y, sr)
    return AUC

def spectral_centroid(signal, sr):
    spec_centroid = tsfel.feature_extraction.features.spectral_centroid(signal, sr)
    return spec_centroid

def spectral_bandwidth(signal, sr):
    spec_bandwidth = tsfel.feature_extraction.features.spectral_bandwidth(signal, sr)
    return spec_bandwidth

def power_bandwidth(signal, sr):
    psd_bandwidth = tsfel.feature_extraction.features.power_bandwidth(signal, sr)
    return psd_bandwidth

def spectral_entropy(signal, sr):
    spec_entropy = tsfel.feature_extraction.features.spectral_entropy(signal, sr)
    return spec_entropy

def spectral_flatness(signal, sr):
    spec_flatness = tsfel.feature_extraction.features.spectral_flatness(signal, sr)
    return spec_flatness

def spectral_skewness(signal, sr):
    spec_skewness = tsfel.feature_extraction.features.spectral_skewness(signal, sr)
    return spec_skewness

def spectral_spread(signal, sr):
    spec_spread = tsfel.feature_extraction.features.spectral_spread(signal, sr)
    return spec_spread

def zero_crossing_rate(signal):
    zero_crossing = tsfel.feature_extraction.features.zero_cross(signal)
    zcr = zero_crossing/len(signal)
    return zcr

def fundamental_frequency(signal, sr):
    fund_freq = tsfel.feature_extraction.features.fundamental_frequency(signal, sr)
    return fund_freq

def interquartile_range(signal):
    iqr = tsfel.feature_extraction.features.interq_range(signal)
    return iqr

def skewness(signal):
    skew = tsfel.feature_extraction.features.skewness(signal)
    return skew

def kurtosis(signal):
    kurtosis = tsfel.feature_extraction.features.kurtosis(signal)
    return kurtosis

def turning_rate(signal):
    neg_turning_points = tsfel.feature_extraction.features.negative_turning(signal)
    pos_turning_points = tsfel.feature_extraction.features.positive_turning(signal)
    turning_points = neg_turning_points + pos_turning_points
    turning_rate = turning_points/len(signal)
    return turning_rate

def neighbourhood_peaks_rate(signal, neighbourhood=10):
    peaks = tsfel.feature_extraction.features.neighbourhood_peaks(signal, neighbourhood)
    peaks_rate = peaks/len(signal)
    return peaks_rate

def peak_to_peak(signal):
    peak_to_peak = tsfel.feature_extraction.features.pk_pk_distance(signal)
    return peak_to_peak

def extract_features(signal, sr, file, label):
    
    features = []

    features.append(file)
    features.append(len(signal))
    features.append(label)


    features.append(AUC(signal, sr))
    features.append(max_freq(signal, sr))
    features.append(spectral_centroid(signal, sr))
    features.append(band_energy_ratio(signal, sr, 800))
    features.append(band_energy_ratio(signal, sr, 1600))
    features.append(one_ratio_band_energy(signal, sr))
    features.append(power_bandwidth(signal, sr))

    features.append(kurtosis(signal))
    features.append(skewness(signal))
    features.append(zero_crossing_rate(signal))

    features.append(spectral_entropy(signal, sr))
    features.append(spectral_skewness(signal, sr))
    features.append(spectral_spread(signal, sr))
    
    features.append(fundamental_frequency(signal, sr))
    features.append(interquartile_range(signal))
    features.append(turning_rate(signal))

    mfcc = compute_mfccs(compute_mel_frequencies(signal, sr), 5)
    for i in range(0, 5):
        features.append(mfcc[i][0])

    return features