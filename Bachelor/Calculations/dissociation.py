#%%
import matplotlib.pyplot as plt
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

def make_subplot(axes, dataname, scilimit, displaylabel, bb2a):
    ax = axes
    min_value = 0
    for key in dataname.keys():
        if min_value > np.min(dataname[key].values):
            min_value = np.min(dataname[key].values)
    for i, key in enumerate(dataname.keys()):
        if i == 0:
            ax.plot(dataname.index.values, dataname[key].values, label = displaylabel, lw = 1, color = 'tab:blue')
        else:
            ax.plot(dataname.index.values, dataname[key].values, lw = 1, color = 'tab:blue', linestyle = '--')
    min_value = np.round(min_value,3)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis = 'both', which = 'major', direction = 'out', bottom = True, left = True, labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', direction = 'out', width = 1, length = 2, bottom = True, left = True)
    ax.ticklabel_format(axis = "y", style = "plain", useOffset = min_value)
    ax.yaxis.offsetText.set_fontsize(9)
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles = h, frameon = False, loc = 4, bbox_to_anchor = bb2a, fontsize = 8, handlelength = 0, handletextpad = 0)
    # print(ax.get_yticklabels())

def make_subplot_individual(axes, dataname, bond_length_eq, gse_at_bleq, displaylabel, n, bb2a):
    ax = axes
    sns.lineplot(data = dataname, ax = ax, lw = 1, color = 'tab:blue')
    ax.scatter(bond_length_eq[n], gse_at_bleq[n], marker = '|', color = 'tab:red', zorder = 10, label = 'Eq bond length')
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis = 'both', which = 'major', direction = 'out', bottom = True, left = True, labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', direction = 'out', width = 1, length = 2, bottom = True, left = True)
    ax.yaxis.offsetText.set_fontsize(9)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles = handles, labels = displaylabel, frameon = False, ncol=2, loc = 4, bbox_to_anchor = bb2a, fontsize = 8)