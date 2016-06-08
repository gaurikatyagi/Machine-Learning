import pandas as pd
import os
import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

def read(filepath, filename):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
    passengers_data = pd.read_csv(os.path.join(filepath, filename), parse_dates='Month', index_col='Month', date_parser=dateparse)
    return passengers_data

if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__name__))
    file_name = "AirPassengers.csv"
    data = read(file_path, file_name)
    # print data
    # print data.index
    # ts = data["#Passengers"]
    # print data.head()
    # print ts.head()
    print "All 1949 data"
    print data["1949"]
    print "Now, let us plot this data"
    plt.plot(data)
    plt.show()
    print "Overall we can see an increasing trend with some seasonality. But, we can not say thta the data is stationary right now."
    print "Checking if the data is stationary with the help of plots of rolling statistics and dickey fuller test"
