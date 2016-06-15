#Frequency Domain Data
#We do not want to transform the time domain data to frequency domain as that would just be a heartbeat expression in
# hertz. instead, we want the frequency measures based on the R-R intervals.

##heart rate varies over time as the heart speeds up and slows down. This variation is expressed in the changing
# distances between heart beats over time (the R-R intervals). The distances between R-R peaks vary over time with their
# own frequency.

#On the frequency side of the heart rate signal the most often found measures are called the HF (High Frequency),
# MF (Mid Frequency) and LF (Low Frequency) bands. The MF and HF bands are taken together and labeled HF.
# LF and HF roughly correspond to 0.04-0.15Hz for the LF band and 0.16-0.5Hz for the HF band.
# The LF band seems related to short-term blood pressure variation, the HF band to breathing rate.

#The frequency spectrum is calculated performing a Fast Fourier Transform over the R-R interval dataseries.

## We will calculate the measure for HF and LF by first by re-sampling the signal so we can estimate the spectrum,
# then transforming the re-sampled signal to the frequency domain,
# and then integrating the area under the curve at the given intervals.

import matplotlib.pyplot as plt
from read_data import read_csv
from time_variability import detect_peak, calc_heart_rate, calc_time_domain, calc_rrdif_rrsqdiff
from scipy.interpolate import interp1d
import numpy as np

def calc_frequency_measures(heart_measure):
    peak_list = heart_measure["peak_xlist"]
    # print peak_list
    RR_list = heart_measure["RR_list"]
    # print RR_list
    RR_x = peak_list[1:] # removing 1st entry because first interval starts from the second beat
    RR_y = RR_list # interval lengths
    RR_x_new = np.linspace(RR_x[0], RR_x[-1], RR_x[-1])
    function = interp1d(RR_x, RR_y, kind = "cubic")
    # print function(250)
    plt.title("Original and Interpolated Signal")
    plt.plot(RR_x, RR_y, label="Original", color='blue')
    plt.plot(RR_x_new, function(RR_x_new), label="Interpolated", color='red')
    plt.legend()
    plt.show()
    return function, RR_x_new

def find_frequency(data, function, frequency, x_new):
    n = len(data["hart"])
    frq = np.fft.fftfreq(n = n, d = (1.0/frequency))
    # print frq
    frq = frq[range(n / 2)]  # Get single side of the frequency range
    # print frq

    ##Performing fft
    Y = np.fft.fft(function(x_new))/n
    # print Y

    Y = Y[range(n/2)] #1side of FFT
    # Plot
    plt.title("Frequency Spectrum of Heart Rate Variability")
    plt.xlim(0, 0.6)  # Limit X axis to frequencies of interest (0-0.6Hz for visibility, we are interested in 0.04-0.5)
    # plt.ylim(0, 50)  # Limit Y axis for visibility
    plt.plot(frq, abs(Y))  # Plot it
    plt.xlabel("Frequencies in Hz")
    plt.show()
    return Y, frq

def lf_hf (Y, frq):
    #The last thing remaining is to integrate the area under curve at the LF (0.04 - 0.15Hz) and HF (0.16 - 0.5Hz)
    # frequency bands. We need to find the data points corresponding to the frequency range we're interested in.
    # During the FFT we calculated the one-sided frequency range frq, so we can search this for the required data point
    # positions.
    lf = np.trapz(abs(Y[(frq>=0.04) & (frq<=0.15)]))
    print "LF:", lf
    hf = np.trapz(abs(Y[(frq >= 0.16) & (frq <= 0.5)]))  # Do the same for 0.16-0.5Hz (HF)
    print "HF:", hf
    #HF is related to breathing and LF to short-term blood pressure regulation. The measures have also been implicated
    # in increased mental activity.
    return hf, lf

if __name__ == "__main__":
    data = read_csv("data.csv")
    # plot_data(data, "Heart Rate Signal")
    frequency = 100  # This dataset has a given frequency of 100Hz
    window_size = 0.75  # one sided window size as a proportion of the sampling frequency
    data_new, heart_measures = detect_peak(data, frequency, window_size)
    heart_measures = calc_heart_rate(heart_measures, frequency)
    # print "bpm is: %0.01f" % bpm

    # plt.title("Detected Peaks in Heart Rate Signal")
    # plt.xlim(0, 2500)
    # plt.plot(data.hart, alpha=0.5,
    #          color="blue",
    #          label="raw signal")  # aplha sets the transparency level
    # plt.plot(heart_measures["moving_average"],
    #          color="black",
    #          ls="-.",
    #          label="moving average")
    # plt.scatter(x=heart_measures["peak_xlist"],
    #             y=heart_measures["R_beat_value"],
    #             color="green",
    #             label="average: %.1f BPM" % heart_measures["bpm"])
    # plt.legend(loc="best")
    # plt.show()

    heart_measures = calc_time_domain(calc_rrdif_rrsqdiff(heart_measures))
    plt.plot(heart_measures["R_beat_value"])
    plt.title("R-R Intervals")
    plt.show()
    #Now we want to find the frequency that makes up this pattern: R-R interval. We can see that these are not evenly
    # spaced in time. This is because the position in time of the intervals is dependent dependent on their length
    # (different for each interval).

    ##we need to:
    #Create an evenly spaced timeline with the R-R intervals on it;
    #Interpolate the signal, which creates an evenly spaced time-series and increases the resolution;(also called
    # re-sampling)
    #Transform the signal to the frequency domain;
    #Integrate the area under the LF and HF portion of the spectrum.
    function, x_new = calc_frequency_measures(heart_measures)
    # Now find the frequencies that make up the interpolated signal, (use numpy's fast fourier transform np.fft.fft()
    # method)
    Y, frq = find_frequency(data, function, frequency, x_new)
    hf, lf = lf_hf(Y, frq)