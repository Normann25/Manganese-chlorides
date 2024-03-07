import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import os, sys
#%%
def read_data(path):
    files = os.listdir(path)
    data_dict = {}

    for file in files:
        if 'Leak_test' in file:
            name = file.split('.')[0]
            with open(os.path.join(path, file)) as f:
                df = pd.read_csv(f, sep = '\t')
                # df['Seconds'] = df['Seconds'] - df['Seconds'][0]
                # df = df.dropna()

            data_dict[name] = df
    return data_dict

def overview_plot_leak(ax, df):
    ax.plot(df['Seconds'], df['CH4 [ppm]'], lw = 1)
    
    ax.set(xlabel = 'Time / s', ylabel = 'CH4 concentration / ppm')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis = 'both', which = 'major', direction = 'out', bottom = True, left = True, labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', direction = 'out', width = 1, length = 2, bottom = True, left = True)
    ax.yaxis.offsetText.set_fontsize(9)

def dict_for_treatment(data_dict, idx_array):
    new_dict = {}
    for i, key in enumerate(data_dict.keys()):
        df = data_dict[key][idx_array[i][0]:idx_array[i][1]]
        df.reset_index(drop=True, inplace=True)
        df['Seconds'] = df['Seconds'] - df['Seconds'][0]
        new_dict[key] = df
    return new_dict

