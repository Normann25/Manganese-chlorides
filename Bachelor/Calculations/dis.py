#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.interpolate import PchipInterpolator as pchip
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import os, sys
#%%
def read_data(path):
    files = os.listdir(path)
    data_dict = {}

    for file in files:
        if '.CSV' in file:
            name = file.split('.')[0]
            with open(os.path.join(path, file)) as f:
                df = pd.read_csv(f, sep = ';', decimal = ',')
                df = df.dropna()

            data_dict[name] = df

        if '.csv' in file:
            name = file.split('.')[0]
            with open(os.path.join(path, file)) as f:
                df = pd.read_csv(f, sep = ';', decimal = ',')
                df = df.dropna()

            data_dict[name] = df
    return data_dict

def energies_in_eV(energy_dict):
    new_dict = {}
    for key in energy_dict.keys():
        if 'treated' in key:
            i = energy_dict[key].keys()[1:]
            energy_eV = energy_dict[key][i]*0.0367502
            new_dict[key] = energy_eV
        if 'v2' in key:
            i = energy_dict[key].keys()[1:]
            energy_eV = energy_dict[key][i]*0.0367502
            new_dict[key] = energy_eV
    return new_dict