# To find the x values for the R points, we will first separate the different peaks from one another.
#First, we find the moving average
#Mark the region of Interest (ROI) where the heart rate signal lies above the moving average.
#Finally marke the highest point for each ROI

import pandas as pd
import numpy as np
import math
from read_data import read_csv, plot_data
import matplotlib.pyplot as plt

#for moving average, the window size we take is determined as: If the window size is big, i.e 2 weeks rather than 2 days
# for example. Then, this means that the value of the moving average that you calculate changes more slowly since it is
# more influenced by past values.

def detect_peak(data, frequency, window_size):
    moving_average = pd.rolling_mean(data.hart, window = frequency*window_size)
    avg_heart_rate = np.mean(data.hart)
    #The beginning of the signal will have a moving average of NaN. Replacing that with the mean of the series.
    moving_average = [avg_heart_rate if math.isnan(element) else element for element in moving_average]
    moving_average = [x*1.2 for x in moving_average] #setting the threshold to the standard 0.2mV value for P and T
    data["hart_rolling_mean"] = moving_average
    # waves' peaks.

    #Mark regions of interest
    window = []
    peak_xlist = []
    #use a counter to move o different data values
    for position, data_point in enumerate(data.hart):
        rolling_mean = data.hart_rolling_mean[position]

        if (data_point < rolling_mean) and (len(window) < 1): #i.e an R peak has not been encountered - no activity.
           pass
            # position += 1

        elif (data_point > rolling_mean): #When the signal comes above local mean, i.e. the region of interest starts
            window.append(data_point)
            # position += 1

        else:
            maximum = max(window) # maximum data point within the window
            beat_R_x = position - len(window) + window.index(maximum) #Finding x-axis value of R point
            peak_xlist.append(beat_R_x) #Add detected R value's
            window = []
            # position += 1

    R_beat_value = [data.hart[ind] for ind in peak_xlist]
    return (peak_xlist, R_beat_value, moving_average)

if __name__ == "__main__":
    data = read_csv("data.csv")
    print "plotting the data"
    plot_data(data, "Heart Rate Signal")
    frequency = 100 #This dataset has a given frequency of 100Hz
    window_size = 0.75 # one sided window size as a proportion of the sampling frequency
    x_value, y_value, moving_average = detect_peak(data, frequency, window_size)
    plt.title ("Detected Peaks in Heart Rate Signal")
    plt.xlim(0, 2500)
    plt.plot(data.hart, alpha = 0.5, color = "blue")#aplha sets the transparency level
    plt.plot(moving_average, color = "green")
    plt.scatter(x = x_value, y = y_value, color = "magenta")
    plt.show()