# The IBI, SDNN, SDSD, RMSSD en pNNx (also frequency domain measures) are grouped under "Heart Rate Variability" (HRV)
# measures, because they give information about how the heart rate varies over time.


import matplotlib.pyplot as plt
import numpy as np
import math
from read_data import read_csv, plot_data
from detect_first_peaks import calc_heart_rate, detect_peak

def calc_rrdif_rrsqdiff (heart_measure):
    RR_list = heart_measure["peak_xlist"]
    RR_diff = []
    RR_sqdiff = []
    for count in range(len(RR_list)-1):
        RR_diff.append(abs(RR_list[count+1]-RR_list[count]))
        RR_sqdiff.append(math.pow((RR_list[count+1]-RR_list[count]), 2))
    heart_measure["RR_diff"] = RR_diff
    heart_measure["RR_sqdiff"] = RR_sqdiff
    return heart_measure

def calc_time_domain(heart_measure):
    """
    This function will calculate the "Heart Rate Variability"
    1. interbeat interval (the mean distance of interval between heartbeats
    2. SDNN (standard deviation of intervals between heartbeats)
    3. SDSD (standard deviation of successive differences between adjacent R-R intervals
    4. RMSSD (root mean square of successive differences between adjacent RR-intervals)
    5. pNN50 and pNN20 (the portion of differences greater than 50ms and 20ms respectively)

    :param heart_measure:
    :return:
    """
    #Calculating the mean interbeat interval- mean of x values of RR values
    heart_measure["ibi"] = np.mean(heart_measure["peak_xlist"])
    # print "Inter-beat interval is: ", ibi

    #Calculating standard deviations of all R-R intervals
    heart_measure["standard_deviation"] = np.std(heart_measure["peak_xlist"])

    # print "standard deviations of R values in the heart rate: ", standard_deviation

    #Calculating the square root of the mean of list of squared differences
    heart_measure["rmssd"] = np.sqrt(np.mean(heart_measure["RR_sqdiff"]))

    #Create a list of all values with RR_diff over 20 and 50
    nn20 = [x for x in heart_measure["RR_diff"] if (heart_measure["RR_diff"]>20)]
    nn50 = [x for x in heart_measure["RR_diff"] if (heart_measure["RR_diff"]>50)]
    heart_measure["pnn20"] = float(len(nn20)) / float(len(heart_measure["RR_diff"]))  # Calculate the proportion of NN20, NN50 intervals to all intervals
    heart_measure["pnn50"] = float(len(nn50)) / float(len(heart_measure["RR_diff"]))  # Note the use of float(), because we don't want Python to think we want an int() and round the proportion to 0 or 1
    # print "pNN20, pNN50:", pnn20, pnn50
    return heart_measure

if __name__ == "__main__":
    data = read_csv("data.csv")
    # plot_data(data, "Heart Rate Signal")
    frequency = 100  # This dataset has a given frequency of 100Hz
    window_size = 0.75  # one sided window size as a proportion of the sampling frequency
    data_new, heart_measures = detect_peak(data, frequency, window_size)
    heart_measures = calc_heart_rate(heart_measures, frequency)
    # print "bpm is: %0.01f" % bpm

    plt.title("Detected Peaks in Heart Rate Signal")
    plt.xlim(0, 2500)
    plt.plot(data.hart, alpha=0.5,
             color="blue",
             label="raw signal")  # aplha sets the transparency level
    plt.plot(heart_measures["moving_average"],
             color="black",
             ls="-.",
             label="moving average")
    plt.scatter(x=heart_measures["peak_xlist"],
                y=heart_measures["R_beat_value"],
                color="green",
                label="average: %.1f BPM" % heart_measures["bpm"])
    plt.legend(loc="best")
    plt.show()

    heart_measures = calc_time_domain(calc_rrdif_rrsqdiff(heart_measures))
    for k in heart_measures:
        print k