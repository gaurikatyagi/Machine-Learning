import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
import os
from read_csv import read
from statsmodels.tsa.seasonal import seasonal_decompose


def stationary(time_series_data):
    #Rolling statistics
    rollmean = pd.rolling_mean(time_series_data, window = 12)
    rollstd = pd.rolling_std(time_series_data, window = 12)

    #Plotting rolling statistics:
    plt.plot(time_series_data, color = "green", label = "original")
    plt.plot(rollmean, color = "blue", label = "mean")
    plt.plot(rollstd, color = "red", label = "standard deviation")
    plt.legend(loc = "best")
    plt.title("Rolling mean and standard deviation")
    plt.show()

def dickey_fuller(time_series_data):
    #print time_series_data
    # print type(time_series_data)
    test = adfuller(time_series_data, maxlag = 12)
    # print test
    dfoutput = pd.Series(test[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in test[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput

def make_stationary(time_series_data):
    """
    One of the first tricks to reduce trend can be transformation. For example, in this case we can clearly see that
    there is a significant positive trend. So we can apply transformation which penalize higher values more than smaller
    values. These can be taking a log, square root, cube root, etc.
    :param time_series_data:
    :return:
    """
    ts_log = np.log(time_series_data)
    plt.plot(ts_log)
    plt.show()
    ##The visible forward trend needs to be now removed from the data (right now the trend and noise are present

    ##Estimating trend from the data
    ##Aggregation - taking average for a time period like monthly/weekly averages
    ##Smoothing - taking rolling averages
    ##Polynomial Fitting - fit a regression model

    ##Smoothing
    #we take average of k consecutive values depending on the frequency of time series. Here we can take the average
    # over the past 1 year, i.e. last 12 values.
    moving_avg = pd.rolling_mean(ts_log, 12)
    plt.plot(moving_avg, color = "green")
    plt.plot(ts_log, color = "red")
    plt.show()
    #since we are taking average of last 12 values, rolling mean is not defined for first 11 values. This can be
    # observed as
    ts_log_moving_avg_diff = ts_log-moving_avg
    print ts_log_moving_avg_diff.head(14)
    #Let us drop the first 11 NAN values
    avg_diff = ts_log_moving_avg_diff.dropna()

    stationary(avg_diff)
    result = dickey_fuller(avg_diff["#Passengers"])
    print result
    ##Now we can see that rolling values appear to be varying slightly but there is no specific trend. Also, the test
    # statistic is smaller than the 5% critical values so we can say with 95% confidence that this is a stationary series.

    # However, a drawback in this particular approach is that the time-period has to be strictly defined. In this case
    # we can take yearly averages but in complex situations like forecasting a stock price, its difficult to come up
    # with a number. So we take a 'weighted moving average' where more recent values are given a higher weight. There
    # can be many technique for assigning weights. A popular one is exponentially weighted moving average where weights
    # are assigned to all the previous values with a decay factor
    weight_ma = pd.ewma(ts_log, halflife = 12)
    ts_log_weight_diff = ts_log-weight_ma
    plt.plot(weight_ma, color="green")
    plt.plot(ts_log, color="red")
    plt.show()
    stationary(ts_log_weight_diff)
    #This TS has even lesser variations in mean and standard deviation in magnitude. Also, the test statistic is smaller
    # than the 1% critical value, which is better than the previous case. Note that in this case there will be no
    # missing values as all values from starting are given weights. So it'll work even with no previous values.

    ##Removing trend and seasonality from a highly seasonal data
    #Differencing - taking the differece with a particular time lag
    #Decomposition - modeling both trend and seasonality and removing them from the model
    #1. differencing
    ts_log_diff = ts_log - ts_log.shift()
    plt.plot(ts_log_diff)
    plt.show()
    #trend seems to have been reduced significantly
    # print ts_log_diff #first value is unkown because its is estimating by shifting
    ts_log_diff.dropna(inplace = True)
    stationary(ts_log_diff)
    #Dickey-Fuller test statistic is less than the 10% critical value, thus the TS is stationary with 90% confidence.
    # We can also take second or third order differences which might get even better results in certain applications.
    ts_log_diff2 = ts_log - ts_log.shift(periods = 2)
    plt.plot(ts_log_diff2)
    plt.show()
    # trend seems to have been reduced significantly
    # print ts_log_diff2  # first value is unkown because its is estimating by shifting
    ts_log_diff2.dropna(inplace=True)
    stationary(ts_log_diff2)

    ##2. Decomposing
    # both trend and seasonality are modeled separately and the remaining part of the series is returned.
    decomp = seasonal_decompose(ts_log)
    trend =  decomp.trend
    season = decomp.seasonal
    residual = decomp.resid
    plt.subplot(411)
    plt.plot(ts_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(season, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    #Lets check stationarity of residuals:
    ts_log_decompose = residual
    ts_log_decompose.dropna(inplace=True)
    stationary(ts_log_decompose)
    #The Dickey-Fuller test statistic is significantly lower than the 1% critical value. This TS is close to stationary.


if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__name__))
    file_name = "AirPassengers.csv"
    data = read(file_path, file_name)
    # print data
    stationary(data)
    stationarity_test = dickey_fuller(data["#Passengers"]) #passing series rather than dataframe
    print stationarity_test
    print "Though the variation in standard deviation is small, mean is clearly increasing with time and this is not a " \
          "stationary series. Also, the test statistic is way more than the critical values."
    make_stationary(data)
    