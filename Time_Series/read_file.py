import pandas as pd
import numpy as np
import os, sys
from pandas import set_option
from pandas.tools.plotting import scatter_matrix


set_option("display.max_rows", 16)

def read(filepath, filename):
    data = pd.read_csv(os.path.join(filepath, filename), parse_dates = True)
    return data

def convert_date(date_item):
    year, month = date_item.split("-")
    # print year
    # print month
    return pd.datetime(int(year), int(month), 1)

if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__name__))
    file_name = "AirPassengers.csv"
    data = read(file_path, file_name)
    print data.columns
    # print data
    print data.dtypes
    print "month is an object and number of passengers are int"
    print "let us see that the dates are all date-time objects and not junk."

    try:
        # print data["Months"]
        data["date"] = data["Month"].apply(convert_date)
        # print data["date"]
    except :
        print("Unexpected error:", sys.exc_info()[0])
        sys.exit()
    print "Conversion to months threw no error that means there is no junk value"
    print "Shows that this data has been recorded monthly. Now that we have the data structure and understand it better" \
          "We will read it as a time series data"

    _ = scatter_matrix(data[['date', 'Month', '#Passengers']], figsize=(14, 10))
    # format_date = pd.to_datetime(data["Month"], format='%Y-%M')
    # months = data["Month"].apply(format_date)
    # print months

    #OR we can directly use the code mentioned in read_csv
