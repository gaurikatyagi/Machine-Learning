import pandas as pd
import matplotlib.pyplot as plt
import os

def read_csv(file_name):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", file_name)
    data = pd.read_csv(file_path)
    return data

def plot_data(data, title):
    plt.title(title)
    plt.plot(data)
    plt.show()

if __name__ == "__main__":
    data = read_csv("data.csv")
    plot_data(data, "Heart Rate Signal")