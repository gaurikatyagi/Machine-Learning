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
from scipy import signal
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
    frequency_measure["frequency"] = (len(sample_timer)/sample_timer[-1])*100 #Divide total length of dataset by last timer
    # entry. This is in deci second, so multiply by 100 to get Hz value

def butter_lowpass(cutoff, frequency, order = 5): #5th order butterpassfilter
    nyquist_frequency = 0.5*frequency #Nyquist frequency is half the sampling frequency
    normal_cutoff = cutoff/nyquist_frequency
    b, a = signal.butter(order, normal_cutoff, btype = "low", analog = False)
    return b, a

def butter_lowpass_filter(data, cutoff, frequency, order):
    b, a = butter_lowpass(cutoff, frequency, order)
    y = signal.lfilter(b, a, data)
    return y

def detect_peaks(data, moving_average_percent, freq):
    """
    This function detects the peak values i.e the R values in the signal of the type QRS complex. It adds to the
    measures dictionary, all the R values and their x coordinates
    :param data: pandas dataframe which stores the dataset
    :param moving_average_percent: integer value which stores the trial value of moving average
    :param freq: calculated frequency of the signal
    :return: This function does not return anything. It updates the dictionary, signal_measures
    """
    # print moving_average_percent
    roll_mean = [x+((moving_average_percent/100)*x) for x in data["hart_rolling_mean"]]
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
    signal_measures["roll_mean"] = roll_mean
    R_R_measures(freq)
    signal_measures["RR_standard_deviation"] = np.std(signal_measures["RR_msdiff"])

def fit_peaks(data, fs):
    moving_average_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100] #list with
    #moving average raise perccentages
    rr_standard_deviation = []
    for ma in moving_average_list: #detect peaks with all moving average percentages
        detect_peaks(data, ma, fs)
        bpm = (len(signal_measures["R_positions"])/(len(data["hart"])/fs)*60)
        rr_standard_deviation.append((signal_measures["RR_standard_deviation"], bpm, ma))
    for sd, bpm_item, ma_item in rr_standard_deviation:
        if (sd>1 and (bpm_item>30 and bpm_item<130)):
            signal_measures["best"]= [sd, ma_item] #the items in rr_standard_deviation are sorted by moving average as they
            #are in the same sequence we fed in
            break
    # print signal_measures["best"]
    detect_peaks(data, signal_measures["best"][1], fs)

def check_peaks():
    RR_msdiff = signal_measures["RR_msdiff"]
    R_positions = signal_measures["R_positions"]
    R_values = signal_measures["R_values"]
    upper_threshold = np.mean(RR_msdiff) + 300 #all values which come after 300ms of the mean value
    lower_threshold = np.mean(RR_msdiff) - 300 # all values which come 300ms before the mean millisecond distance between values
    removed_beats_position = []
    removed_beats_y = []
    RR_ms_diff_corrected = []
    R_positions_corrected = []
    R_values_corrected =[]
    for index in range(len(RR_msdiff)):
        if (RR_msdiff[index] <upper_threshold and RR_ms_diff_corrected>lower_threshold):
            RR_ms_diff_corrected.append((RR_msdiff[index]))
            R_positions_corrected.append(R_positions[index])
            R_values_corrected.append(R_values[index])
        else:
            removed_beats_position.append(R_positions[index])
            removed_beats_y.append(R_values[index])
    signal_measures["RR_msdiff_corrected"] = RR_ms_diff_corrected
    signal_measures["R_positions_corrected"] = R_positions_corrected
    signal_measures["R_values_corrected"] = R_values_corrected

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
    time_measures["bpm"] = 60000/np.mean(signal_measures["RR_msdiff_corrected"]) #beats per minute
    time_measures["ibi"] = np.mean(signal_measures["RR_msdiff_corrected"]) #interbeat interval
    time_measures["sdnn"] = np.std(signal_measures["RR_msdiff_corrected"]) #standard deviation of milisecond difference between R values
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
    R_positions = signal_measures["R_positions_corrected"]
    ybeat = signal_measures["R_values_corrected"]
    plt.title(title)
    plt.plot(data["hart"], alpha=0.5, color='blue', label="raw signal")
    plt.plot(data["hart_rolling_mean"], color='green', label="moving average")
    plt.scatter(R_positions, ybeat, color='red', label="average: %.1f BPM" % time_measures['bpm'])
    plt.legend(loc=4, framealpha=0.6)
    plt.show()

if __name__ == "__main__":
    data = read_data("noisy_data.csv")
    window_size = 0.75
    calc_freq_rate(data)
    frequency = frequency_measure["frequency"]#100
    print "Frequency of the data is: ", frequency
    filtered = butter_lowpass_filter(data = data["hart"], cutoff = 2.5, frequency = frequency, order = 5)
    plt.subplot(211)
    plt.plot(data["hart"], color = "red", label = "original hart", alpha = 0.5)
    plt.legend(loc = "auto")
    plt.subplot(212)
    plt.plot(filtered, color="green", label="filtered hart", alpha=0.5)
    plt.legend(loc = "auto")
    plt.suptitle("original v/s filtered data")
    plt.show()
    data["hart"] = filtered
    dataset_moving_average = rolling_mean(data, window_size, frequency)
    # detect_peaks(dataset_moving_average)
    fit_peaks(dataset_moving_average, frequency)
    check_peaks()
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
