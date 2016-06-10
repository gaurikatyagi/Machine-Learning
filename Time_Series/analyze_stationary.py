import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
import os
from read_csv import read
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

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
    return ts_log

def forecast(ts, log_series):
    """
    make model on the TS after differencing. Having performed the trend and seasonality estimation techniques,
    there can be two situations:
    A strictly stationary series with no dependence among the values. This is the easy case wherein we can model the
    residuals as white noise. But this is very rare.
    A series with significant dependence among values. In this case we need to use some statistical models like ARIMA to
    forecast the data.

    The predictors depend on the parameters (p,d,q) of the ARIMA model:
    Number of AR (Auto-Regressive) terms (p): AR terms are just lags of dependent variable. For instance if p is 5,
    the predictors for x(t) will be x(t-1)...x(t-5).
    Number of MA (Moving Average) terms (q): MA terms are lagged forecast errors in prediction equation. For instance
    if q is 5, the predictors for x(t) will be e(t-1)...e(t-5) where e(i) is the difference between the moving average
    at ith instant and actual value.
    Number of Differences (d): These are the number of nonseasonal differences, i.e. in this case we took the first
    order difference. So either we can pass that variable and put d=0 or pass the original variable and put d=1.
    Both will generate same results.

    We use two plots to determine these numbers. Lets discuss them first.

    Autocorrelation Function (ACF): It is a measure of the correlation between the the TS with a lagged version of itself.
    For instance at lag 5, ACF would compare series at time instant 't1'...'t2' with series at instant 't1-5'...'t2-5'
    (t1-5 and t2 being end points).
    Partial Autocorrelation Function (PACF): This measures the correlation between the TS with a lagged version of itself
    but after eliminating the variations already explained by the intervening comparisons. Eg at lag 5, it will check
    the correlation but remove the effects already explained by lags 1 to 4.

    :param log_series:
    :return:
    """
    #ACF and PACF plots
    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff = ts_log_diff.dropna()
    lag_acf = acf(ts_log_diff, nlags = 20)
    lag_pacf = pacf(ts_log_diff, nlags = 20, method = "ols")
    #plot ACF
    plt.subplot(221)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle="--", color="gray")
    plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle="--", color="gray") #lower line of confidence interval
    plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff)), linestyle="--", color="gray") #upper line of confidence interval
    plt.title('Autocorrelation Function')

    # Plot PACF:
    plt.subplot(222)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle="--", color="gray")
    plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle="--", color="gray")
    plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff)), linestyle="--", color="gray")
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()

    #from these plots, we get p and q:
    #p - The lag value where the PACF chart crosses the upper confidence interval for the first time. If you notice
    # closely, in this case p=2.
    #q - The lag value where the ACF chart crosses the upper confidence interval for the first time. If you notice
    # closely, in this case q=2.

    #AR model
    res_arima = arima_models(ts_log, 2, 1, 0)
    # print pd.Series(res_arima.fittedvalues)
    plt.subplot(223)
    plt.plot(ts_log_diff)
    plt.plot(res_arima.fittedvalues, color='red')
    # plt.title('AR model--RSS: %.4f' % sum((pd.Series(res_arima.fittedvalues) - ts_log_diff) ** 2))

    #MA model
    res_ma = arima_models(ts_log, 0, 1, 2)
    plt.subplot(224)
    plt.plot(ts_log_diff)
    plt.plot(res_ma.fittedvalues, color='red')
    # plt.title('MA model--RSS: %.4f' % sum((res_ma.fittedvalues - ts_log_diff) ** 2))
    plt.plot()

    ##Combined model
    res = arima_models(ts_log, 2, 1, 2)
    plt.plot(ts_log_diff)
    plt.plot(res.fittedvalues, color='red')
    # plt.title('RSS: %.4f' % sum((res.fittedvalues - ts_log_diff) ** 2))
    plt.show()
    #Here we can see that the AR and MA models have almost the same RSS but combined is significantly better.

    #predicting
    predictions_diff = pd.Series(res.fittedvalues, copy=True)
    print predictions_diff.head()
    #Notice that these start from '1949-02-01' and not the first month; because we took a lag by 1 and first element
    # doesn't have anything before it to subtract from. The way to convert the differencing to log scale is to add these
    # differences consecutively to the base number. An easy way to do it is to first determine the cumulative sum at
    # index and then add it to the base number. The cumulative sum can be found as:
    predictions_diff_cumsum = predictions_diff.cumsum()
    #now add them to the base number

    predictions_arima_log = pd.Series(ts_log.ix[0], index = ts_log.index)
    predictions_arima_log = predictions_arima_log.add(predictions_diff_cumsum, fill_value = 0)
    #now let us take the exponential to regain original form of series
    predictions_ARIMA = np.exp(predictions_arima_log)
    plt.plot(ts)
    plt.plot(predictions_ARIMA)
    # plt.title('RMSE: %.4f' % np.sqrt(sum((predictions_ARIMA - ts) ** 2) / len(ts)))
    plt.show()

def arima_models(ts_log, p, d, q):
    model = ARIMA(ts_log, order = (p, d, q))
    results = model.fit(disp = -1)
    return results

if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__name__))
    file_name = "AirPassengers.csv"
    data = read(file_path, file_name)
    # print data
    # stationary(data)
    # stationarity_test = dickey_fuller(data["#Passengers"]) #passing series rather than dataframe
    # print stationarity_test
    print "Though the variation in standard deviation is small, mean is clearly increasing with time and this is not a " \
          "stationary series. Also, the test statistic is way more than the critical values."
    ts_log = make_stationary(data)
    forecast(data["#Passengers"], ts_log)