##We will see how to deal with noise in a few stages:
#Evaluate the result of passing this signal to our algorithm from part two;
#Careful: Sampling Frequency;
#Filter the signal to remove unwanted frequencies (noise);
#Improving detection accuracy with a dynamic threshold;
#Detecting incorrectly detected / missed peaks;
#Removing errors and reconstructing the R-R signal to be error-free.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.interpolate import interp1d
import sys

signal_measures = {}
time_measures = {}
frequency_measure ={}

def read_data(filename):
    """
    This function reads in the csv file as a pandas dataframe
    :param filename: String variable which contains the file name to be read
    :return: returns pandas dataframe containing the csv file
    """
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", filename)
    print "Reading file: %s"%filename
    try:
        data = pd.read_csv(file_path) # We can not make index_col= 0 because the function detect_peaks() makes use of
        #default indices
        print "File read successful"
    except:
        print "File does not exist. Exiting..."
        sys.exit()
    return data

def rolling_mean(data, window_size, frequency):
    """
    This function calculates a rolling mean signal which averages the given signal over window sizes defined by the
    multiple of window_size and frequency
    :param data: pandas dataframe which stores the dataset
    :param window_size: float value which defines the size of the window to be taken when calculating rolling mean
    :param frequency: frequency of sampling the heart signal
    :return: this function returns the pandas dataframe with an added column for rolling mean values
    """
    moving_average = pd.rolling_mean(data["hart"], window = window_size*frequency)
    heart_rate_average = np.mean(data["hart"])
    moving_average = [heart_rate_average if math.isnan(average) else average for average in moving_average]
    data["hart_rolling_mean"] = moving_average
    return data

def calc_freq_rate(data):
    sample_timer = [x for x in data["timer"]] # dataset.timer is a ms counter with start of recording at '0'
    frequency_measure["frequency"] = (len(sample_timer)/sample_timer[-1])*1000 #Divide total length of dataset by last timer
    # entry. This is in ms, so multiply by 1000 to get Hz value

def detect_peaks(data):
    """
    This function detects the peak values i.e the R values in the signal of the type QRS complex. It adds to the
    measures dictionary, all the R values and their x coordinates
    :param data: pandas dataframe which stores the dataset
    :return: This function does not return anything. It updates the dictionary, signal_measures
    """
    window = []
    peak_position_list = []
    for position, datapoint in enumerate(data["hart"]):
        rolling_mean_value = data["hart_rolling_mean"][position]
        if (datapoint>rolling_mean_value):
            window.append(datapoint) #put value of signal in window if the signal value is more than the mean
        elif (datapoint<=rolling_mean_value and len(window)<=1):
            pass
        else:
            R_value = max(window)
            R_position = position-len(window)+window.index(R_value)
            peak_position_list.append(R_position)
            window = []
    signal_measures["R_positions"] = peak_position_list
    signal_measures["R_values"] = [data["hart"][peaks] for peaks in peak_position_list]

def R_R_measures(frequency):
    """
    This function finds the x-coordinate distance between each consecutive R value, its square and the distance in terms
    of miliseconds
    :param data: pandas dataframe which stores the dataset
    :param frequency: integer value which gives the sampling frequency of heart signal
    :return: This function does not return anything. It updates the dictionary, signal_measures
    """
    R_positions = signal_measures["R_positions"]
    RR_msdiff = [] #Stores the mili second distance between consecutive R values
    for position in range(len(R_positions)-1):
        RR_interval = R_positions[position+1]-R_positions[position]
        distance_ms = (RR_interval/frequency)*1000.0
        RR_msdiff.append(distance_ms)

    RR_diff = []
    RR_sqdiff = []
    for position in range(len(RR_msdiff) - 1):
        RR_diff.append(abs(R_positions[position] - R_positions[position+1]))
        RR_sqdiff.append(math.pow(abs(R_positions[position] - R_positions[position+1]), 2))

    signal_measures["RR_msdiff"] = RR_msdiff
    signal_measures["RR_diff"] = RR_diff # difference in positions of consecutive R values in terms of x values
    signal_measures["RR_sqdiff"] = RR_sqdiff

def calc_ts_measures():
    """
    This function fills the time-measure values of the signals in the dictionary time_measures
    """
    time_measures["bpm"] = 6000/np.mean(signal_measures["RR_msdiff"]) #beats per minute
    time_measures["ibi"] = np.mean(signal_measures["RR_msdiff"]) #interbeat interval
    time_measures["sdnn"] = np.std(signal_measures["RR_msdiff"]) #standard deviation of milisecond difference between R values
    time_measures["sdsd"] = np.std(signal_measures["RR_diff"]) #standard deviation of standard differences between R values
    time_measures["rmsd"] = np.sqrt(np.mean(signal_measures["RR_sqdiff"]))
    nn_20 = [x for x in signal_measures["RR_diff"] if (x>20)]
    nn_50 = [x for x in signal_measures["RR_diff"] if (x>50)]
    time_measures["nn20"] = nn_20
    time_measures["nn50"] = nn_50
    time_measures["pnn20"] = float(len(nn_20))/len(signal_measures["RR_diff"])
    time_measures["pnn50"] = float(len(nn_50))/len(signal_measures["RR_diff"])

def calc_frequency_measures(data, frequency):
    R_positions = signal_measures["R_positions"][1:]
    RR_msdiff = signal_measures["RR_msdiff"]
    RR_msdiff_x = np.linspace(R_positions[0], R_positions[-1], R_positions[-1])
    func = interp1d(R_positions, RR_msdiff, kind = "cubic")
    n = len(data["hart"])
    frequency_f = np.fft.fftfreq(n = n, d = (1.0/frequency))
    frequency_f = frequency_f[range(n/2)]
    y = np.fft.fft(func(RR_msdiff_x))/n
    y = y[range(n/2)]
    frequency_measure["lf"] = np.trapz(abs(y[(frequency_f>=0.04) & (frequency_f<=0.15)]))
    frequency_measure["hf"] = np.trapz(abs(y[(frequency_f>=0.16) & (frequency_f<=0.5)]))

def plot_data(data, title):
    R_positions = signal_measures["R_positions"]
    ybeat = signal_measures["R_values"]
    plt.title(title)
    plt.plot(data["hart"], alpha=0.5, color='blue', label="raw signal")
    plt.plot(data["hart_rolling_mean"], color='green', label="moving average")
    plt.scatter(R_positions, ybeat, color='red', label="average: %.1f BPM" % time_measures['bpm'])
    plt.legend(loc=4, framealpha=0.6)
    plt.show()

if __name__ == "__main__":
    dataset = read_data("noisy_data.csv")
    window_size = 0.75
    calc_freq_rate(dataset)
    frequency = frequency_measure["frequency"]#100
    dataset_moving_average = rolling_mean(dataset, window_size, frequency)
    detect_peaks(dataset_moving_average)
    dataset = dataset_moving_average[["hart", "hart_rolling_mean"]]
    dataset.plot(title = "Heart Rate signal with moving average")

    # print dataset_moving_average
    R_R_measures(frequency)
    calc_ts_measures()
    # print time_measures
    # print signal_measures
    calc_frequency_measures(dataset_moving_average, frequency)
    # print frequency_measure
    plt.show()
    plot_data(dataset_moving_average, "Heart Beat Plot")
