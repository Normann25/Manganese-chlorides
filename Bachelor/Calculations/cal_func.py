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
                df = pd.read_csv(f)
                df = df.dropna()

            data_dict[name] = df
        
        else:
            pass
    return data_dict

def cal_spec_extention(df):
    x_values = np.linspace(0, 700, 10000)
    new_df = pd.DataFrame(index=x_values)

    for X_key, Y_key in zip(df.keys()[::2], df.keys()[1::2]):
        X = df[X_key].dropna().values
        ex_high_X = np.linspace(max(X)+0.1, 700)
        ex_low_X = np.linspace(0, min(X)-0.1)
        X_full = np.concatenate([ex_low_X, X, ex_high_X])

        Y = df[Y_key].dropna().values
        ex_high_Y = np.zeros_like(ex_high_X)
        ex_low_Y = np.zeros_like(ex_low_X)
        Y_full = np.concatenate([ex_low_Y, Y, ex_high_Y])

        spline = pchip(X_full, Y_full)
        new_df[Y_key] = spline(x_values)
    return new_df

def scaled_spectra(df, percentage_list):
    df_spline = cal_spec_extention(df)
    new_df = pd.DataFrame(index = df_spline.index)
    new_df['full spectrum'] = df_spline.sum(axis='columns')

    for i, percentage in enumerate(percentage_list):
        new_df[df_spline.keys()[i]] = df_spline[df_spline.keys()[i]].values*percentage
    
    new_df['scaled spectrum'] = new_df[new_df.keys()[1:]].sum(axis='columns')
    return new_df

def scaled_spec_water(df, percentage_low, percentage_high, functionals, aq_list):
    dict_low = {}
    dict_high = {}

    for func in functionals:
        for aq in aq_list:
            key = ' '.join([func, aq])
            mask = [key in i for i in df.keys()]
            uvvis_unscaled = df.loc[:, mask]
            if 'cam' in key:
                dict_low[key] = scaled_spectra(uvvis_unscaled, percentage_low)
                dict_high[key] = scaled_spectra(uvvis_unscaled, percentage_high)
    
    return dict_low, dict_high

def experimental_plot(ax, df, y_val, label, ncols):
    ax = ax
    ax.plot(df['wavelength'], df[y_val], label = label)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis = 'both', which = 'major', direction = 'out', bottom = True, left = True, labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', direction = 'out', width = 1, length = 2, bottom = True, left = True)
    # ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = scilimit)
    ax.yaxis.offsetText.set_fontsize(8)
    ax.legend(frameon = False, ncol=ncols, fontsize = 8)
    ax.set_xlabel('Wavelength / nm', fontsize = 8)
    ax.set_ylabel('Molar absorption \n coefficient / M$^{-1}$cm$^{-1}$', fontsize = 8)


def scaled_plot(ax, df_list, labels_list, scilimit, ncols):
    ax = ax
    for i, label in enumerate(labels_list):
        ax.plot(df_list[i].index, df_list[i]['scaled spectrum'], label = label)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis = 'both', which = 'major', direction = 'out', bottom = True, left = True, labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', direction = 'out', width = 1, length = 2, bottom = True, left = True)
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = scilimit)
    ax.yaxis.offsetText.set_fontsize(9)
    ax.legend(frameon = False, ncol=ncols, fontsize = 8)

def scaled_plot_dict(ax, df_dict, labels_list, scilimit, ncols):
    ax = ax
    for df, label in zip(df_dict.values(), labels_list):
        ax.plot(df.index, df['scaled spectrum'], label = label)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis = 'both', which = 'major', direction = 'out', bottom = True, left = True, labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', direction = 'out', width = 1, length = 2, bottom = True, left = True)
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = scilimit)
    ax.yaxis.offsetText.set_fontsize(9)
    ax.legend(frameon = False, ncol=ncols, fontsize = 8)

def scaled_plot_species(ax, df, y_vals, labels_list, scilimit, ncols):
    ax = ax
    df_species = df[y_vals]
    for i, label in enumerate(labels_list):
        ax.plot(df.index, df_species[df_species.keys()[i]], label = label)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis = 'both', which = 'major', direction = 'out', bottom = True, left = True, labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', direction = 'out', width = 1, length = 2, bottom = True, left = True)
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = scilimit)
    ax.yaxis.offsetText.set_fontsize(8)
    ax.legend(frameon = False, ncol=ncols, fontsize = 8)
    ax.set_xlabel('Wavelength / nm', fontsize = 8)
    ax.set_ylabel('Molar absorption \n coefficient / M$^{-1}$cm$^{-1}$', fontsize = 8)

def make_subplot(axes, data, xdataname,  ydataname, labels_aq, scilimit, ncols):
    ax = axes
    if '1aq' in xdataname:
        sns.lineplot(x = data[xdataname], y = data[ydataname], ax = ax, label = labels_aq[0], color = 'tab:blue')
    if '2aq' in xdataname:
        sns.lineplot(x = data[xdataname], y = data[ydataname], ax = ax, label = labels_aq[1], color = 'tab:orange')
    if '3aq' in xdataname:
        sns.lineplot(x = data[xdataname], y = data[ydataname], ax = ax, label = labels_aq[2], color = 'tab:green')
    if '4aq' in xdataname:
        sns.lineplot(x = data[xdataname], y = data[ydataname], ax = ax, label = labels_aq[3], color = 'tab:red')
    if '5aq' in xdataname:
        sns.lineplot(x = data[xdataname], y = data[ydataname], ax = ax, label = labels_aq[4], color = 'tab:purple')
    if '6aq' in xdataname:
        sns.lineplot(x = data[xdataname], y = data[ydataname], ax = ax, label = labels_aq[5], color = 'tab:brown')
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis = 'both', which = 'major', direction = 'out', bottom = True, left = True, labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', direction = 'out', width = 1, length = 2, bottom = True, left = True)
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = scilimit)
    ax.yaxis.offsetText.set_fontsize(9)
    ax.legend(frameon = False, ncol=ncols, fontsize = 8)
#%%
# df = pd.read_csv('./water_sim/uvvis_v2.CSV', sep = ';', decimal = ',')
#%%
# functionals = ['cam', 'wb']
# aq_list = ['1aq', '2aq']  #, '3aq', '4aq', '5aq', '6aq']
# percentage_test = [0.5, 0.3, 0.15, 0.05]

# df_dict_cam = {}
# df_dict_wb = {}

# for func in functionals:
#     for aq in aq_list:
#         key = ' '.join([func, aq])
#         mask = [key in i for i in df.keys()]
#         df_unscaled = df.loc[:, mask]
#         if 'cam' in key:
#             df_dict_cam[key] = scaled_spectra(df_unscaled, percentage_test)
#         if 'wb' in key:
#             df_dict_wb[key] = scaled_spectra(df_unscaled, percentage_test)

# labels_list = ['1 water', '2 water']

# fig, axes = plt.subplots(2,1)
# scaled_plot_dict(axes[0], df_dict_cam, labels_list, (3,3), 1)
# scaled_plot_dict(axes[1], df_dict_wb, labels_list, (3,3), 1)

#%%
# labels_Mn = [r'Mn$^{2+}$', r'MnCl$^{+}$', r'MnCl$_{2}$', r'MnCl$_{3}^{+}$']

# fig, ax = plt.subplots(1,1)
# scaled_plot_species(ax, df_dict_cam['cam 1aq'], df_dict_cam['cam 1aq'].keys()[1:5], labels_Mn, (3,3), 1)
#%%
# cam1 = 'cam 1aq'
# cam1_mask = [cam1 in i for i in df.keys()]
# df_cam1 = df.loc[:,cam1_mask]

# cam2 = 'cam 2aq'
# cam2_mask = [cam2 in i for i in df.keys()]
# df_cam2 = df.loc[:,cam2_mask]

# cam1_test = scaled_spectra(df_cam1, percentage_test)
# cam2_test = scaled_spectra(df_cam2, percentage_test)
#%%
# labels_list = ['1 water', '2 water']
# df_list = [cam1_test, cam2_test]

# fig, ax = plt.subplots(1,1)
# scaled_plot(ax, df_list, labels_list, (3,3), 1)
#%%
# df_cam1_ex = cal_spec_extention(df_cam1)

# plt.plot(df_cam1_ex)
#%%
# for X_key, Y_key in zip(df_cam1_ex.keys()[::2], df_cam1_ex.keys()[1::2]):
#     plt.plot(df_cam1_ex[X_key], df_cam1_ex[Y_key])
# plt.show()
# for X_key, Y_key in zip(df_cam1.keys()[::2], df_cam1.keys()[1::2]):
#     plt.plot(df_cam1[X_key], df_cam1[Y_key])
# plt.xlim(0, 700)
# plt.show()
