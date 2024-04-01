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
def read_data(path):
    files = os.listdir(path)
    files_list = []

    for file in files:
        if '.dat' in file:
            with open(os.path.join(path, file)) as f:
                df = pd.read_table(f, sep = '\s+')
                df['seconds'] = pd.to_timedelta(df['TIME']).astype('timedelta64[s]')

                for key in df.keys()[2:]:
                    df[key] = pd.to_numeric(df[key].replace(',', '.'), errors='coerce')

                files_list.append(df)

    full_df = pd.concat(files_list)
    new_df = full_df[::4]
    new_df.reset_index(drop=True, inplace=True)
    new_df['seconds'] = new_df['seconds'] - new_df['seconds'][0]
    
    return new_df

def dict_for_treatment(df, idx_array, keys):
    new_dict = {}
    for i, values in enumerate(idx_array):
        new_df = df[values[0]:values[1]]
        new_df.reset_index(drop=True, inplace=True)
        # new_df['seconds'] = new_df['seconds'] - new_df['seconds'][0]
        new_dict[keys[i]] = new_df
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
        x = data_dict[key]['seconds']
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

        print(f"{key}  Fit: a={a_fit:6.7f}+-{sigma_a_fit:5.8f}  b={b_fit:5.3f}+-{sigma_b_fit:5.3f}  p={Prob_fit:6.6f}")
    
    return array_a, array_b, array_ea, array_eb, array_Chi2, array_ndf, array_Prob

def plot_full_exp(ax, df, a, b, idx, lamp_interval):
    x = df['seconds'][idx[0]:idx[1]] - df['seconds'][idx[0]]
    y_leak1 = b[0] * np.exp(a[0] * x)
    y_lamp = b[1] * np.exp(a[1] * x)
    y_leak2 = b[2] * np.exp(a[2] * x)

    ax.plot(x, df['HR_12CH4'][idx[0]:idx[1]], label = 'Experimental data')
    ax.plot(x, y_leak1, label = 'Leak rate 1 (fit)', lw = 1)
    ax.plot(x, y_lamp, label = 'Radiation (fit)', lw = 1)
    ax.plot(x, y_leak2, label = 'Leak rate 2 (fit)', lw = 1)

    ax.scatter(df['seconds'][lamp_interval[0]] - df['seconds'][idx[0]], df['HR_12CH4'][lamp_interval[0]], marker = '|', color = 'k', s = 300, zorder = 10)
    ax.scatter(df['seconds'][lamp_interval[1]] - df['seconds'][idx[0]], df['HR_12CH4'][lamp_interval[1]], marker = '|', color = 'k', s = 300, zorder = 10)

    ax.legend(frameon = False, fontsize = 8)
    # ax.set(xlabel = 'Time / s', ylabel = 'CH4 concentration / ppm')
    ax.set_xlabel('Time / s', fontsize = 8)
    ax.set_ylabel('CH4 concentration / ppm', fontsize = 8)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis = 'both', which = 'major', direction = 'out', bottom = True, left = True, labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', direction = 'out', width = 1, length = 2, bottom = True, left = True)

def plot_before_lamp(ax, df, a, b, exp_label):
    x = df[exp_label]['seconds']
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

def plot_mean_conc(ax, df, bbox2anchor, a, b, ea, eb):
    x = df['seconds']
    y = b[1] * np.exp(a[1] * x)
    x_mean = df['seconds'].mean()
    y1, ey1 = get_mean_conc(x_mean, a[0], b[0], ea[0], eb[0])
    print('before radiation: ', y1, '+-', ey1)
    y2, ey2 = get_mean_conc(x_mean, a[1], b[1], ea[1], eb[1])
    print('after radiation: ', y2, '+-', ey2)

    ax.plot(x, y, label = 'Fitted', color = 'k')
    ax.scatter(x, df['HR_12CH4'], label = 'Experimental', s = 10, zorder = 10)
    ax.scatter(x_mean, y1, label = 'Mean before light', s = 10, zorder = 10, color = 'g')
    ax.scatter(x_mean, y2, label = 'Mean after light', s = 10, zorder = 10, color = 'r')

    ax.legend(frameon = False, fontsize = 8, bbox_to_anchor = bbox2anchor)
    ax.set_xlabel('Time / s', fontsize = 8)
    ax.set_ylabel('CH4 concentration / ppm', fontsize = 8)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis = 'both', which = 'major', direction = 'out', bottom = True, left = True, labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', direction = 'out', width = 1, length = 2, bottom = True, left = True)

    return y1, ey1, y2, ey2