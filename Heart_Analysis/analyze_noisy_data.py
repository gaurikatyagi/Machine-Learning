##We will see how to deal with noise in a few stages:
#Evaluate the result of passing this signal to our algorithm from part two;
#Careful: Sampling Frequency;
#Filter the signal to remove unwanted frequencies (noise);
#Improving detection accuracy with a dynamic threshold;
#Detecting incorrectly detected / missed peaks;
#Removing errors and reconstructing the R-R signal to be error-free.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.interpolate import interp1d
import sys

time_measures = {}
def read_data(filename):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", filename)
    print "Reading file: %s"%filename
    try:
        data = pd.read_csv(file_path)
        print "File read successful"
    except:
        print "File does not exist. Exiting..."
        sys.exit()

if __name__ == "__main__":
    read_data("noisy_data")