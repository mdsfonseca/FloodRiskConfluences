#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import xarray as xr
import numpy as np
import glob

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd

import lmom as lmom
from scipy.stats import spearmanr
from scipy.stats import norm



# In[2]:


def TSdict(path, ens, conf):
    """
    Time series dictionary
    path: path of only 1 of the met files
    ens: number of the ensemble to be processed
    conf: configuration of the confluences
    Returns: ds_
    """
    ds_ = xr.open_dataset(path, chunks={}, decode_times=False)
    
#Rainfall and temperatures to time series
    # store all time series PER file PER ensemble in a dictionary, all station!!     Total rainfall, not the rate!!! 
    # [Ensemble=ens, time=all(:), station=n]   --> only ensemble 'ens' out of 20
# the rain:
    stations = ds_.stations.values
    ds_rain = {f"S{n}": ds_.rainfall_rate[ens, :, n].to_series() * int(ds_.station_area[1, n].values) for n in range(len(stations))}
    #ds_rain #mm  km2 = rainfall_rate #mm   [Ensemble, time, station] * station_area  #km2  [Ensemble, station]

    ds_snow = {f"S{n}": ds_.rainfall_rate[1, :, n].to_series() * int(ds_.station_area[1, n].values) for n in range(len(stations))}
# the tremperature:
    ds_temp = {f"S{n}": ds_.air_temperature[1, :, n].to_series() for n in range(len(stations))}

#when T<0, P=0 (snow)
    for i in ds_rain:
        for j in range(len(ds_rain[i])):
            if ds_temp[i][j] > 0:
                ds_snow[i][j] = ds_rain[i][j]
            elif ds_temp[i][j] <= 0:
                ds_snow[i][j] = 0 

#Calculate (accumulate) the rain for the confluences..  
    for n in range(len(conf)):
        ds_snow[conf.Confluence[n]] = ds_snow[conf.Stream1[n]] + ds_snow[conf.Stream2[n]]
    
    return ds_rain, ds_snow, ds_temp

#This plot only draws 1 stream..   would be nice to draw the 2 streams at a confluence..  so change it for the report ;)
def dsplot(ds_rain, ds_snow, ds_temp, stream, start, end):
    plt.figure(figsize=(20, 5))
    plt.plot(ds_temp[stream][start:end], 'r', label='Temp')
    plt.axhline(y = 0, color = 'r', linestyle = '-')
    plt.title(f'{stream} Temperature') #Runoff
    plt.xlabel('Date')
    plt.ylabel('Temp (°C)') # Q m3/s
    plt.legend()
    plt.grid()

    plt.figure(figsize=(20, 5))
    plt.plot(ds_rain[stream][start:end], 'gray', label='Precipitation (w/o snow)')
    plt.plot(ds_snow[stream][start:end], 'b', label='Precipitation (w/ snow)')
    plt.title(f'{stream} Precipitation') #Runoff
    plt.xlabel('Date')
    plt.ylabel('Precipitation (mm-km2)') # Q m3/s
    plt.legend()
    plt.grid()
    return


# In[ ]:


#Accumulate rainfall
def TSdictCONF(path, conf):
    """
    Time series dictionary per confluence
    path: path of only 1 of the met files
    ens: number of the ensemble to be processed
    conf: configuration of the confluences
    Returns: ds_
    """
    
#Rainfall 
    ds_rain = np.load(path, allow_pickle=True).item()

#Calculate (accumulate) the rain for the confluences..  
    for n in range(len(conf)):
        ds_rain[conf.Confluence[n]] = ds_rain[conf.Stream1[n]] + ds_rain[conf.Stream2[n]]
    
    return ds_rain


# Marginals

# In[3]:


#Empirical distribution function
def ecdf(data):
    """ Compute Empirical CDF """
    x = np.sort(data)
    n = x.size
    rank = np.arange(1, n+1)
    y = rank / (n+1)
    return x, y

def marginalsPOT(b1, b2, set1):
    #Marginal distributions - Parameter estimation
    LMU1 = lmom.samlmu(set1[b1])
    LMU2 = lmom.samlmu(set1[b2])

    #General pareto fit
    gpa_fit1 = lmom.pelgpa(LMU1)
    gpa_fit2 = lmom.pelgpa(LMU2)
    
    rho, Sp = spearmanr(set1[b1], set1[b2])
    #rho = set1[b1].corr(set1[b2])
    return gpa_fit1, gpa_fit2, rho

def marginalsAM(b1, b2, set1):
    #Marginal distributions - Parameter estimation
    LMU1 = lmom.samlmu(set1[b1])
    LMU2 = lmom.samlmu(set1[b2])

    #General pareto fit
    gev_fit1 = lmom.pelgev(LMU1)
    gev_fit2 = lmom.pelgev(LMU2)
    
    rho, Sp = spearmanr(set1[b1], set1[b2])
    #rho = set1[b1].corr(set1[b2])
    return gev_fit1, gev_fit2, rho


# AM sets

# In[4]:


def AM(ds, conf, c, w):
    """
    Sets of Annual Maxima
    ds: dictionary with all the Time series per station 
    conf: configuration of the confluences
    c: confluence to be analyzed
    w: window of the multi-day events
    Returns: ds_
    """
    #create the Pandas DataFrame
    conf_1 = pd.DataFrame(columns=['P_S1', 'P_S2'])
    conf_1['P_S1'] = ds[conf.Stream1[c]]
    conf_1['P_S2'] = ds[conf.Stream2[c]]
    #1 Rolling 5 days
#     w = w #days
    conf_1['P_S1rol'] = conf_1['P_S1'].rolling(window=w).sum()
    conf_1['P_S2rol'] = conf_1['P_S2'].rolling(window=w).sum()
    conf_1['Conf'] = conf_1.P_S1rol + conf_1.P_S2rol
    
    #2 Extreme selection sets
    conf_1.index = conf_1.index.astype(int)
    conf_1['Time'] = pd.to_datetime(conf_1.index, unit='d')
    conf_1['year']=conf_1.Time.dt.year
    conf_1['month']=conf_1.Time.dt.month

    #Seasonal averages
    conditions = [
        (conf_1.month==1)|(conf_1.month==2)|(conf_1.month==12),
        (conf_1.month==3)|(conf_1.month==4)|(conf_1.month==5),
        (conf_1.month==6)|(conf_1.month==7)|(conf_1.month==8),
        (conf_1.month==9)|(conf_1.month==10)|(conf_1.month==11)]

    choices = ['DJF', 'MAM', 'JJA', 'SON']
    conf_1['season'] = np.select(conditions, choices, default='ERROR')
    
    savg_S1 = conf_1.groupby(['season'])['P_S1rol'].mean()
    savg_S2 = conf_1.groupby(['season'])['P_S2rol'].mean()
    savg_C = conf_1.groupby(['season'])['Conf'].mean()
    
#     conditionsALL = [conf_1.season=='DJF', conf_1.season=='JJA', conf_1.season=='MAM', conf_1.season=='SON']

#     choicesS1 = [savg_S1['DJF'], savg_S1['JJA'], savg_S1['MAM'], savg_S1['SON']]
#     choicesS2 = [savg_S2['DJF'], savg_S2['JJA'], savg_S2['MAM'], savg_S2['SON']]
#     choicesC = [savg_C['DJF'], savg_C['JJA'], savg_C['MAM'], savg_C['SON']]

#     conf_1['S1savg'] = np.select(conditionsALL, choicesS1, default='ERROR').astype(float)
#     conf_1['S2savg'] = np.select(conditionsALL, choicesS2, default='ERROR').astype(float)
#     conf_1['Csavg'] = np.select(conditionsALL, choicesC, default='ERROR').astype(float)
#     conf_1['S1/S1savg'] = conf_1.P_S1rol / conf_1.S1savg
#     conf_1['S2/S2savg'] = conf_1.P_S2rol / conf_1.S2savg
#     conf_1['C/Csavg'] = conf_1.Conf / conf_1.Csavg
    
    numyears = conf_1.year.iloc[-1] - conf_1.year.iloc[0] + 1
    firstyear = conf_1.year.iloc[0]


    am1 = pd.DataFrame() #columns=['Time', 'year', 'AccumP_rol0', 'AccumP_rol1']
    am2 = pd.DataFrame()
    am3 = pd.DataFrame()
    for i in range(numyears):
        am1 = am1.append(conf_1.loc[conf_1.P_S1rol[conf_1.year == firstyear + i].idxmax()])#, ignore_index=True)
        am2 = am2.append(conf_1.loc[conf_1.P_S2rol[conf_1.year == firstyear + i].idxmax()])
        am3 = am3.append(conf_1.loc[conf_1.Conf[conf_1.year == firstyear + i].idxmax()])        

    return am1, am2, am3, savg_S1, savg_S2, savg_C


# Metrics

# In[1]:


def met(am):
    gev_fit1, gev_fit2, rho = marginalsAM('P_S1rol', 'P_S2rol', am)
    SS1sav_mean = np.mean(am['S1/S1savg'])
    SS2sav_mean = np.mean(am['S2/S2savg'])
    
    return gev_fit1, gev_fit2, rho, SS1sav_mean, SS2sav_mean


# In[7]:


def met1(am):
    gev_fit1, gev_fit2, rho = marginalsAM('P_S1rol', 'P_S2rol', am)
    SS2sav_mean = np.mean(am['S2/S2savg'])
    SS2sav_std = np.std(am['S2/S2savg'])
    SS2sav_cv = SS2sav_std / SS2sav_mean
    
    return gev_fit1, gev_fit2, rho, SS2sav_mean, SS2sav_std, SS2sav_cv

def met2(am):
    gev_fit1, gev_fit2, rho = marginalsAM('P_S1rol', 'P_S2rol', am)
    SS1sav_mean = np.mean(am['S1/S1savg'])
    SS1sav_std = np.std(am['S1/S1savg'])
    SS1sav_cv = SS1sav_std / SS1sav_mean
    
    return gev_fit1, gev_fit2, rho, SS1sav_mean, SS1sav_std, SS1sav_cv

def met3(am):
    gev_fit1, gev_fit2, rho = marginalsAM('P_S1rol', 'P_S2rol', am)
    SS1sav_mean = np.mean(am['S1/S1savg'])
    SS1sav_std = np.std(am['S1/S1savg'])
    SS1sav_cv = SS1sav_std / SS1sav_mean
    SS2sav_mean = np.mean(am['S2/S2savg'])
    SS2sav_std = np.std(am['S2/S2savg'])
    SS2sav_cv = SS2sav_std / SS2sav_mean
    
    return gev_fit1, gev_fit2, rho, SS1sav_mean, SS1sav_std, SS1sav_cv, SS2sav_mean, SS2sav_std, SS2sav_cv


# Seasonal averages

# In[8]:


def Savg(ds, conf, c, w):
    """
    Seasonal average metric
    ds: dictionary with all the Time series per station 
    conf: configuration of the confluences
    c: confluence to be analyzed
    w: window of the multi-day events
    Returns: ds_
    """
    #create the Pandas DataFrame
    conf_1 = pd.DataFrame(columns=['P_S1', 'P_S2'])
    conf_1['P_S1'] = ds[conf.Stream1[c]]
    conf_1['P_S2'] = ds[conf.Stream2[c]]
    #1 Rolling 5 days
#     w = w #days
    conf_1['P_S1rol'] = conf_1['P_S1'].rolling(window=w).sum()
    conf_1['P_S2rol'] = conf_1['P_S2'].rolling(window=w).sum()
    conf_1['Conf'] = conf_1.P_S1rol + conf_1.P_S2rol
    
    #2 Extreme selection sets
    conf_1.index = conf_1.index.astype(int)
    conf_1['Time'] = pd.to_datetime(conf_1.index, unit='d')
    conf_1['year']=conf_1.Time.dt.year
    conf_1['month']=conf_1.Time.dt.month

    #Seasonal averages
    conditions = [
        (conf_1.month==1)|(conf_1.month==2)|(conf_1.month==12),
        (conf_1.month==3)|(conf_1.month==4)|(conf_1.month==5),
        (conf_1.month==6)|(conf_1.month==7)|(conf_1.month==8),
        (conf_1.month==9)|(conf_1.month==10)|(conf_1.month==11)]

    choices = ['DJF', 'MAM', 'JJA', 'SON']
    conf_1['season'] = np.select(conditions, choices, default='ERROR')
    
    savg_S1 = conf_1.groupby(['season'])['P_S1rol'].mean()
    savg_S2 = conf_1.groupby(['season'])['P_S2rol'].mean()
    savg_C = conf_1.groupby(['season'])['Conf'].mean()
    
    return savg_S1, savg_S2, savg_C


# In[ ]:




