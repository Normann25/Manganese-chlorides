import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
import matplotlib as mpl
import pandas as pd
import numpy as np
from iminuit import Minuit
from scipy import stats
from sympy import *
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

def read_data_picarro(parent_path, dates):
    data_dict = {}

    for date in dates:
        path = os.path.join(parent_path, date)
        files = os.listdir(path)
        files_list = []

        for file in files:
            if '.dat' in file:
                with open(os.path.join(path, file)) as f:
                    df = pd.read_table(f, sep = '\s+')
                    df['Seconds'] = pd.to_timedelta(df['TIME']).astype('timedelta64[s]')

                    for key in df.keys()[2:]:
                        df[key] = pd.to_numeric(df[key].replace(',', '.'), errors='coerce')
                    # df = df.sort_values(by = 'TIME', ascending = True)
                    files_list.append(df)

        full_df = pd.concat(files_list)
        new_df = full_df[::4]
        new_df.reset_index(drop=True, inplace=True)
        new_df['Seconds'] = new_df['Seconds'] - new_df['Seconds'][0]
        data_dict[date] = new_df
        
    return data_dict

def get_treatment_dict(data_dict, time_stamps, new_keys):
    pd.options.mode.chained_assignment = None 
    
    idx_array = []
    for i, dict_key in enumerate(data_dict.keys()):
        idx_ts = np.zeros(len(time_stamps[i]))
        for j, ts in enumerate(time_stamps[i]):
            for k, time in enumerate(data_dict[dict_key]['TIME']):
                if ts in time:
                    idx_ts[j] += k
        idx_array.append(idx_ts)
    
        print(idx_ts)

    new_dict = {}
    for i, key in enumerate(data_dict.keys()):
        for j, idx in enumerate(idx_array[i][::2]):
            new_df = data_dict[key].iloc[int(idx):int(idx_array[i][j*2+1]), :]
            new_df.reset_index(drop=True, inplace=True)
            new_df['Minutes'] = new_df['Seconds'] / 60
            new_dict[new_keys[i][j]] = new_df

        for new_key in new_keys[i]:
            new_dict[new_key]['Seconds'] = new_dict[new_key]['Seconds'] - data_dict[key]['Seconds'][idx_array[i][0]]
    return new_dict

def dict_for_treatment(data_dict, idx_array, new_keys):
    new_dict = {}
    for i, key in enumerate(data_dict.keys()):
        for j, idx in enumerate(idx_array[i][::2]):
            new_df = data_dict[key].iloc[idx:idx_array[i][j*2+1], :]
            new_df.reset_index(drop=True, inplace=True)
            new_df['Minutes'] = new_df['Seconds'] / 60
            new_dict[new_keys[i][j]] = new_df
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
        if 'Leak' in key:
            x = data_dict[key]['Minutes']
            y = data_dict[key]['CH4 [ppm]']
            if max(y) > 100:
                ey = np.zeros(len(y)) + 250
            if max(y) < 100:
                ey = np.zeros(len(y)) + 0.8
        else:
            x = data_dict[key]['Seconds']
            y = data_dict[key]['HR_12CH4']
            ey = np.zeros(len(y)) + 50/1000
            for j, conc in enumerate(y):
                ey[j] += 0.05 * conc
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

        print(f"{key}  Fit: tau={a_fit:6.6f}+-{sigma_a_fit:5.8f}  c_0={b_fit:5.3f}+-{sigma_b_fit:5.3f}  p={Prob_fit:6.6f}")
    
    return array_a, array_b, array_ea, array_eb, array_Chi2, array_ndf, array_Prob

def overview_plot_axetris(ax, df):
    ax.plot(df['Seconds'], df['CH4 [ppm]'], lw = 1)
    
    ax.set(xlabel = 'Time / s', ylabel = 'CH4 concentration / ppm')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis = 'both', which = 'major', direction = 'out', bottom = True, left = True, labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', direction = 'out', width = 1, length = 2, bottom = True, left = True)
    ax.yaxis.offsetText.set_fontsize(9)

def plot_fit_axetris(ax, df, a, b, ea, eb, chi2, ndf, prob):
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

def plot_full_exp(ax, df, a, b, idx, lamp_interval):
    x = df['Seconds'][idx[0]:idx[1]] - df['Seconds'][idx[0]]
    y_leak1 = b[0] * np.exp(a[0] * x)
    y_lamp = b[1] * np.exp(a[1] * x)
    y_leak2 = b[2] * np.exp(a[2] * x)

    ax.plot(x, df['HR_12CH4'][idx[0]:idx[1]], lw = 0.9, color = 'k')
    ax.plot(x, y_leak1, lw = 1, color = 'tab:blue', ls = '--')
    ax.plot(x, y_lamp, lw = 1, color = 'red', ls = '--')
    ax.plot(x, y_leak2, lw = 1, color = 'forestgreen', ls = '--')

    ax.axvspan(df['Seconds'][lamp_interval[0]] - df['Seconds'][idx[0]], 
               df['Seconds'][lamp_interval[1]] - df['Seconds'][idx[0]], color='y', alpha=0.25, lw=0)
    
    # ax.set(xlabel = 'Time / s', ylabel = 'CH4 concentration / ppm')
    ax.set_xlabel('Time / s', fontsize = 8)
    ax.set_ylabel('CH4 concentration / ppm', fontsize = 8)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis = 'both', which = 'major', direction = 'out', bottom = True, left = True, labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', direction = 'out', width = 1, length = 2, bottom = True, left = True)

def plot_before_lamp(ax, df, a, b, exp_label):
    x = df[exp_label]['Seconds']
    y = b * np.exp(a * x)
    ax.plot(x, y, label = 'Fitted data', color = 'k')
    ax.scatter(x, df[exp_label]['HR_12CH4'], label = 'Experimental data', s = 5, zorder = 10)

    ax.legend(frameon = False, fontsize = 8)
    ax.set_xlabel('Time / s', fontsize = 8)
    ax.set_ylabel('CH4 concentration / ppm', fontsize = 8)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis = 'both', which = 'major', direction = 'out', bottom = True, left = True, labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', direction = 'out', width = 1, length = 2, bottom = True, left = True)

def get_mean_conc(x_mean, a, b, ea, eb):
    # Define variables:
    C,A,B,x = symbols("C, A, B, x")
    dC,dA,dB = symbols("sigma_C, sigma_A, sigma_B")

    # Define relation
    C = B*exp(x*A)

    # Calculate uncertainty
    dC = sqrt((C.diff(A) * dA)**2 + (C.diff(B) * dB)**2)

    # Turn expression into numerical functions 
    # lambdify transform SymPy expressions to lambda functions which can be used to calculate numerical values very fast
    fC = lambdify((A,B,x),C)
    fdC = lambdify((A,dA,B,dB,x),dC)

    # Define values and their errors
    vA, vdA, vB, vdB, vx = a, ea, b, eb, x_mean

    # Numerically evaluate expressions
    vC = fC(vA, vB, vx)
    vdC = fdC(vA, vdA, vB, vdB, vx)

    return vC, vdC

def plot_mean_conc(ax, df, a, b, ea, eb):
    x = df['Seconds']
    y = b[1] * np.exp(a[1] * x)
    x_mean = df['Seconds'].mean()
    y1, ey1 = get_mean_conc(x_mean, a[0], b[0], ea[0], eb[0])
    print('Before radiation: ', y1, '+-', ey1)
    y2, ey2 = get_mean_conc(x_mean, a[1], b[1], ea[1], eb[1])
    print('After radiation: ', y2, '+-', ey2)

    ax.plot(x, y, color = 'k')
    ax.scatter(x, df['HR_12CH4'], s = 10, zorder = 10)
    ax.scatter(x_mean, y1, s = 10, zorder = 10, color = 'g')
    ax.scatter(x_mean, y2, s = 10, zorder = 10, color = 'r')

    ax.set_xlabel('Time / s', fontsize = 8)
    ax.set_ylabel('CH4 concentration / ppm', fontsize = 8)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis = 'both', which = 'major', direction = 'out', bottom = True, left = True, labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', direction = 'out', width = 1, length = 2, bottom = True, left = True)

    return y1, ey1, y2, ey2