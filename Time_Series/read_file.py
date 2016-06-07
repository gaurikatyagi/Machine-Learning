import pandas as pd
import numpy as np
import os

def read(filepath, filename):
    data = pd.read_csv(os.path.join(filepath, filename))
    return data

if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__name__))
    file_name = "AirPassengers.csv"
    data = read(file_path, file_name)
    print data