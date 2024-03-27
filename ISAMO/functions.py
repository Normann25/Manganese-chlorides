import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns
from iminuit import Minuit
from scipy import stats
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import os, sys
sys.path.append('..')
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH, nice_string_output, add_text_to_ax
#%%
def read_data_leak(path):
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

def read_data_exp(path):
    files = os.listdir(path)
    data_dict = {}

    for file in files:
        if 'exp' in file:
            name = file.split('.')[0]
            with open(os.path.join(path, file)) as f:
                df = pd.read_csv(f, sep = '\t')
                # df['Seconds'] = df['Seconds'] - df['Seconds'][0]
                # df = df.dropna()

            data_dict[name] = df
    return data_dict

def overview_plot(ax, df):
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
        df['Minutes'] = df['Seconds'] / 60
        new_dict[key] = df
    return new_dict

def fit_exp(data_dict, a_guess, b_guess):
    array_a = np.zeros(len(data_dict.keys()))
    array_b = np.zeros(len(data_dict.keys()))
    array_ea = np.zeros(len(data_dict.keys()))
    array_eb = np.zeros(len(data_dict.keys()))
    array_Chi2 = np.zeros(len(data_dict.keys()))
    array_ndf = np.zeros(len(data_dict.keys()))
    array_Prob = np.zeros(len(data_dict.keys()))

    for i, key in enumerate(data_dict.keys()):
        x = data_dict[key]['Minutes']
        y = data_dict[key]['CH4 [ppm]']
        if max(y) > 100:
            ey = np.zeros(len(y)) + 250
        if max(y) < 100:
            ey = np.zeros(len(y)) + 0.8
        Npoints = len(y)

        def fit_func(x, a, b):
            return b * np.exp(a * x)

        def chi2_owncalc(a, b) :
            y_fit = fit_func(x, a, b)
            chi2 = np.sum(((y - y_fit) / ey)**2)
            return chi2
        chi2_owncalc.errordef = 1.0    # Chi2 definition (for Minuit)

        # Here we let Minuit know, what to minimise, how, and with what starting parameters:   
        minuit = Minuit(chi2_owncalc, a = a_guess[i], b = b_guess[i])

        # Perform the actual fit:
        minuit.migrad();

        # Extract the fitting parameters and their errors:
        a_fit = minuit.values['a']
        b_fit = minuit.values['b']
        sigma_a_fit = minuit.errors['a']
        sigma_b_fit = minuit.errors['b']

        Nvar = 2                     # Number of variables 
        Ndof_fit = Npoints - Nvar    # Number of degrees of freedom = Number of data points - Number of variables

        # Get the minimal value obtained for the quantity to be minimised (here the Chi2)
        Chi2_fit = minuit.fval                          # The chi2 value
        Prob_fit = stats.chi2.sf(Chi2_fit, Ndof_fit)    # The chi2 probability given N degrees of freedom 

        array_a[i] = a_fit
        array_b[i] = b_fit
        array_ea[i] = sigma_a_fit
        array_eb[i] = sigma_b_fit
        array_Chi2[i] = Chi2_fit
        array_ndf[i] = Ndof_fit
        array_Prob[i] = Prob_fit

        print(f"{key}  Fit: a={a_fit:6.6f}+-{sigma_a_fit:5.8f}  b={b_fit:5.3f}+-{sigma_b_fit:5.3f}  p={Prob_fit:6.6f}")
    
    return array_a, array_b, array_ea, array_eb, array_Chi2, array_ndf, array_Prob

def plot_fit(ax, df, a, b, ea, eb, chi2, ndf, prob):
    x1, y1 = df['Minutes'], df['CH4 [ppm]']
    y_fit = b * np.exp(a * x1)
    ax.plot(x1, y1, label = 'Measured CH4 conc')
    ax.plot(x1, y_fit, label = 'Fitted CH4 conc')
    ax.legend(frameon = False, fontsize = 9)
    ax.set(xlabel = 'Time / min', ylabel = 'CH4 concentration / ppm')
    d = {r'$\tau$':   [a, ea],
         'C$_{0}$':   [b, eb],
        # 'Chi2':     chi2,
        # 'Ndf':      ndf,
        # 'Prob':     prob,
        }
    text = nice_string_output(d, extra_spacing=2, decimals=5)
    add_text_to_ax(0.02, 0.2, text, ax, fontsize=8)