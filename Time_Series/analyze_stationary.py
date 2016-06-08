import pandas as pd
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller
import os
from read_csv import read

def stationary(time_series_data):
    #Rolling statistics
    rollmean = pd.rolling_mean(time_series_data, window = 10)
    rollstd = pd.rolling_std(time_series_data, window = 10)

    #Plotting rolling statistics:
    original = plt.plot(time_series_data, color = "green", label = "original")
    mean = plt.plot(rollmean, color = "blue", label = "mean")
    std = plt.plot(rollstd, color = "red", label = "standard deviation")
    plt.legend(loc = "best")
    plt.title("Rolling mean and standard deviation")
    plt.show()

if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__name__))
    file_name = "AirPassengers.csv"
    data = read(file_path, file_name)

    stationary(data)