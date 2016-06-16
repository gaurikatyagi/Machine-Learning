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

def read_data(filename):
    """
    This function reads in the csv file as a pandas dataframe
    :param filename: String variable which contains the file name to be read
    :return: returns pandas dataframe containing the csv file
    """
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", filename)
    print "Reading file: %s"%filename
    try:
        data = pd.read_csv(file_path, index_col= 0)
        print "File read successful"
    except:
        print "File does not exist. Exiting..."
        sys.exit()
    return data

def rolling_mean(data, window_size, frequency):
    moving_average = pd.rolling_mean(data["hart"], window = window_size*frequency)
    heart_rate_average = np.mean(data["hart"])
    moving_average = [heart_rate_average if math.isnan(average) else average for average in moving_average]
    data["hart_rolling_mean"] = moving_average
    return data

def detect_peaks(data):
    window = []
    peak_position_list = []
    temp = None
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



def plot_data(data, title):
    data.plot(title = title)
    plt.show()

if __name__ == "__main__":
    dataset = read_data("noisy_data.csv")
    window_size = 0.75
    frequency = 100
    dataset_moving_average = rolling_mean(dataset, window_size, frequency)
    detect_peaks(dataset_moving_average)
    plot_data(dataset_moving_average, "Heart Rate signal with moving average")
    # print dataset_moving_average