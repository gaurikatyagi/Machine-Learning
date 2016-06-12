#Sampling frequency of the data is 100Hz

import pandas as pd
import matplotlib.pyplot as plt
import os

def read_csv(file_name):
    """
    This function reads the heart beats data as a pandas object
    :param file_name: string containing the name of the file which contains the data
    :return: returns a pandas object known as data
    """
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", file_name)
    data = pd.read_csv(file_path)
    return data

def plot_data(data, title):
    """
    This function plots the figure of the pandas object passed
    :param data: pandas object to be plotted
    :param title: string object which contains the title of the graph
    """
    plt.title(title)
    plt.plot(data)
    plt.show()

if __name__ == "__main__":
    data = read_csv("data.csv")
    plot_data(data, "Heart Rate Signal")
    # Now we need to find our region of interest (ROI). This can be found out by identifying the R-peaks in the QRS
    # portions of the signal. The first step is to find the position of all the R-peaks.
    # This can be done in 3 ways:
    # 1. Fit a curve on the ROI data-points and solve for the x-position of the maximum. This is the most exact and
    # most expensive computation. With huge datasets this will take a large amount of time.
    # 2. Determine the slope between each set of points in ROI. Then, find the set where the slope reverses.
    # 3. Mark datapoints within the ROI, find the position of the highest point.

    # For now, instead of considering the actual maximum of the curve (lies between the tw data points, rather than on
    # the data point. With a high sampling rate, this error decreases.