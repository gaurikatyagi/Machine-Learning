# The IBI, SDNN, SDSD, RMSSD en pNNx (also frequency domain measures) are grouped under "Heart Rate Variability" (HRV)
# measures, because they give information about how the heart rate varies over time.

#Frequency Domain Data
#On the frequency side of the heart rate signal the most often found measures are called the HF (High Frequency),
# MF (Mid Frequency) and LF (Low Frequency) bands. The MF and HF bands are taken together and labeled HF.
# LF and HF roughly correspond to 0.04-0.15Hz for the LF band and 0.16-0.5Hz for the HF band.
# The LF band seems related to short-term blood pressure variation, the HF band to breathing rate.

#The frequency spectrum is calculated performing a Fast Fourier Transform over the R-R interval dataseries.

## We will calculate the measure for HF and LF by first by re-sampling the signal so we can estimate the spectrum,
# then transforming the re-sampled signal to the frequency domain,
# and then integrating the area under the curve at the given intervals.

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
    #Calculating the mean interbeat interval- mean of x values of RR values
    ibi = np.mean(heart_measure["peak_xlist"])
    print "Inter-beat interval is: ", ibi

    #Calculating standard deviations of all R-R intervals
    standard_deviation = np.std(heart_measure["peak_xlist"])
    print "standard deviations of R values in the heart rate: ", standard_deviation

    #Calculating the square root of the mean of list of squared differences
    rmssd = np.sqrt(np.mean(heart_measure["RR_sqdiff"]))

    #Create a list of all values with RR_diff over 20 and 50
    nn20 = [x for x in heart_measure["RR_diff"] if (heart_measure["RR_diff"]>20)]
    nn50 = [x for x in heart_measure["RR_diff"] if (heart_measure["RR_diff"]>50)]
    pnn20 = float(len(nn20)) / float(len(heart_measure["RR_diff"]))  # Calculate the proportion of NN20, NN50 intervals to all intervals
    pnn50 = float(len(nn50)) / float(len(heart_measure["RR_diff"]))  # Note the use of float(), because we don't want Python to think we want an int() and round the proportion to 0 or 1
    print "pNN20, pNN50:", pnn20, pnn50

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

    heart_measures = calc_rrdif_rrsqdiff(heart_measures)
    calc_time_domain(heart_measures)