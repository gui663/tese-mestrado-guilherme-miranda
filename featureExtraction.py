from scipy import signal
import pandas as pd
from scipy.signal import find_peaks
import statistics
from scipy.stats import skew
from scipy.stats import kurtosis
import numpy as np

def welch_mean_std(x):


    features = []
    for i in range(len(x)):

        signal = x.iloc[i,:]

        mean = statistics.mean(signal)
        stdev = statistics.stdev(signal)
        median = statistics.median(signal)
        variance = statistics.variance(signal)
        skewness = skew(signal, bias=False)
        kurtose = kurtosis(signal, fisher=False)

        feature = [mean, stdev, median, variance, skewness, kurtose]
        features.append(feature)

    features = pd.DataFrame(features)
    return features


def Welch(x):

    f, feature = signal.welch(x, 10000)

    return feature

def extractWelch(signal):

    features = []
    for i in range(len(signal)):

        feature = Welch(signal.iloc[i,:])
        features.append(feature)

    features = pd.DataFrame(features)
    return features

def p100amplitude(epoch):

    try:
        peaks = find_peaks(epoch, height=1*10**(6))
        peaks_pos = peaks[0]
        first_peak = epoch[peaks_pos[0]]

        epoch2 = epoch*-1
        minimum = find_peaks(epoch2, height=1*10**(6))
        minimum_pos = minimum[0]
        first_minimum = epoch2[minimum_pos[0]]

        p100 = abs(first_peak-first_minimum)

        return p100, peaks_pos[0], minimum_pos[0]
    except:
        
        print("Could not extract p100")
        return None

def extract_p100(signal):

    features = []
    for i in range(len(signal)):

        feature = p100amplitude(signal.iloc[i,:])
        if feature != None:
            p100 = feature[0]*10**(-6)
            features.append(p100)

    features = pd.DataFrame(features)
    return features