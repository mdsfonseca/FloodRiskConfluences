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


# ## HID02

# AM sets
# 

# In[2]:


#See MetFunctions AM
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
    conf_1 = pd.DataFrame(columns=['year', 'month', 'day', 'P_S1', 'P_S2'])
    conf_1['P_S1'] = ds[conf.HVB_Q1[c]]
    conf_1['P_S2'] = ds[conf.HVB_Q2[c]]
    conf_1['year'] = ds['YEAR']
    conf_1['month'] = ds['MONTH']
    conf_1['day'] = ds['DAY']
    conf_1['P_S1'].loc[0] = 0
    conf_1['P_S2'].loc[0] = 0
    #1 Rolling 5 days
#     w = w #days
    conf_1['P_S1rol'] = conf_1['P_S1'].rolling(window=w).sum()
    conf_1['P_S2rol'] = conf_1['P_S2'].rolling(window=w).sum()
    conf_1['Conf'] = conf_1.P_S1rol + conf_1.P_S2rol
    
    #2 Extreme selection sets
#     conf_1.index = conf_1.index.astype(int)
#     conf_1['Time'] = pd.to_datetime(conf_1.index, unit='d')
#     conf_1['year']=conf_1.Time.dt.year
#     conf_1['month']=conf_1.Time.dt.month

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


# In[3]:


def OpenCVS(path):
    df = pd.read_csv(path,  delim_whitespace=True, skiprows=[0,1])
    df.columns = ['YEAR','MONTH','DAY','MIN','HOUR','I-RN-0001', 'I-RN-0002', 'I-RN-0003', 'I-RN-0004', 'I-RN-0005', 
                  'I-RN-0006', 'I-RN-0007', 'I-RN-0008', 'I-RN-0009', 'I-RN-0010', 'I-RN-0011', 'I-RN-0012', 'I-RN-0013', 
                  'I-RN-0014', 'I-RN-0015', 'I-RN-0016', 'I-RN-0017', 'I-RN-0018', 'I-RN-0019', 'I-RN-0020', 'I-RN-0021', 
                  'I-RN-0022', 'I-RN-0023', 'I-RN-0024', 'I-RN-0025', 'I-RN-0026', 'I-RN-0027', 'I-RN-0028', 'I-RN-0029', 
                  'I-RN-0030', 'I-RN-0031', 'I-RN-0032', 'I-RN-0033', 'I-RN-0034', 'I-RN-0035', 'I-RN-0036', 'I-RN-0037', 
                  'I-RN-0038', 'I-RN-0039', 'I-RN-0040', 'I-RN-0041', 'I-RN-0042', 'I-RN-0043', 'I-RN-0044', 'I-RN-0045', 
                  'I-RN-0046', 'I-RN-0047', 'I-RN-0048', 'I-RN-0049', 'I-RN-0050', 'I-RN-0051', 'I-RN-0052', 'I-RN-0053', 
                  'I-RN-0054', 'I-RN-0055', 'I-RN-0056', 'I-RN-0057', 'I-RN-0058', 'I-RN-0059', 'I-RN-0060', 'I-RN-0061', 
                  'I-RN-0062', 'I-RN-0063', 'I-RN-0064', 'I-RN-0065', 'I-RN-0066', 'I-RN-0067', 'I-RN-0068', 'I-RN-0069', 
                  'I-RN-0070', 'I-RN-0071', 'I-RN-0072', 'I-RN-0073', 'I-RN-0074', 'I-RN-0075', 'I-RN-0076', 'I-RN-0077', 
                  'I-RN-0078', 'I-RN-0079', 'I-RN-0080', 'I-RN-0081', 'I-RN-0082', 'I-RN-0083', 'I-RN-0084', 'I-RN-0085', 
                  'I-RN-0086', 'I-RN-0087', 'I-RN-0088', 'I-RN-0089', 'I-RN-0090', 'I-RN-0091', 'I-RN-0092', 'I-RN-0093', 
                  'I-RN-0094', 'I-RN-0095', 'I-RN-0096', 'I-RN-0097', 'I-RN-0098', 'I-RN-0099', 'I-RN-0100', 'I-RN-0101', 
                  'I-RN-0102', 'I-RN-0103', 'I-RN-0104', 'I-RN-0105', 'I-RN-0106', 'I-RN-0107', 'I-RN-0108', 'I-RN-0109', 
                  'I-RN-0110', 'I-RN-0111', 'I-RN-0112', 'I-RN-0113', 'I-RN-0114', 'I-RN-0115', 'I-RN-0116', 'I-RN-0117', 
                  'I-RN-0118', 'I-RN-0119', 'I-RN-0120a', 'I-RN-0120b', 'I-RN-0120c', 'I-RN-0120d', 'I-RN-0121', 'I-RN-0122', 
                  'I-RN-0123', 'I-RN-0124', 'I-RN-0125a', 'I-RN-0125b', 'I-RN-0125c', 'I-RN-0125d', 'I-RN-0125e', 'I-RN-0125f', 
                  'I-RN-0126', 'I-RN-0127a', 'I-RN-0127b', 'I-RN-0127c', 'I-RN-0127d', 'I-RN-0128', 'I-RN-0129a', 'I-RN-0129b', 
                  'I-RN-0129c', 'I-RN-0129d', 'I-RN-0130', 'I-RN-0131', 'I-RN-0132', 'I-RN-0133', 'I-RN-0134']
    return df


# ## HID03

# In[4]:


def AddSavg(am1, S1_savg, S2_savg, C_savg):
    conditionsALL = [am1.season=='DJF', am1.season=='JJA', am1.season=='MAM', am1.season=='SON']

    choicesS1 = [S1_savg['DJF'], S1_savg['JJA'], S1_savg['MAM'], S1_savg['SON']]
    choicesS2 = [S2_savg['DJF'], S2_savg['JJA'], S2_savg['MAM'], S2_savg['SON']]
    choicesC = [C_savg['DJF'], C_savg['JJA'], C_savg['MAM'], C_savg['SON']]

    am1['S1savg'] = np.select(conditionsALL, choicesS1, default='ERROR').astype(float)
    am1['S2savg'] = np.select(conditionsALL, choicesS2, default='ERROR').astype(float)
    am1['Csavg'] = np.select(conditionsALL, choicesC, default='ERROR').astype(float)
    am1['S1/S1savg'] = am1.P_S1rol / am1.S1savg
    am1['S2/S2savg'] = am1.P_S2rol / am1.S2savg
    am1['C/Csavg'] = am1.Conf / am1.Csavg
    return am1


# ## HID04

# In[5]:


def randSet(am1, n):  #CHANGED --> now POT also uses the window as an imput
    """
    Creates DataFrame set of random Annual Maxima (AM) of one point
    AM1: DataFrame with the AM 
    n: size of the random set e.g. (50)
    Returns:
        set1: random set of AM of Point1 and concurrent flow of Point2
    """
    am1['ind'] = np.arange(len(am1))
    randomindex1 = np.random.choice(am1.ind, size=n, replace=False)
    set1 = pd.DataFrame()
    for i in range(n):
        set1 = set1.append(am1.loc[am1.ind == randomindex1[i]])
    return set1

def randSetR(am1, n):  #CHANGED --> now POT also uses the window as an imput
    """
    Creates DataFrame set of random Annual Maxima (AM) of one point
    AM1: DataFrame with the AM 
    n: size of the random set e.g. (50)
    Returns:
        set1: random set of AM of Point1 and concurrent flow of Point2
    """
    am1['ind'] = np.arange(len(am1))
    randomindex1 = np.random.choice(am1.ind, size=n, replace=True)
    set1 = pd.DataFrame()
    for i in range(n):
        set1 = set1.append(am1.loc[am1.ind == randomindex1[i]])
    return set1


# In[6]:


#Create the dataframes to be filled per N-sample

def dfN():
    conf_AM = pd.read_csv('Confluences_wnames.csv', delimiter=';')
    conf_AM['rho_mean'] = ''
    conf_AM['rho_std'] = ''
#     conf_AM['rho_cv'] = ''
    conf_AM['S1avg_mean'] = ''
    conf_AM['S1avg_std'] = ''
    conf_AM['S1std_mean'] = ''
    conf_AM['S1std_std'] = ''
#     conf_AM['S1avg_cv'] = ''
    conf_AM['S2avg_mean'] = ''
    conf_AM['S2avg_std'] = ''
    conf_AM['S2std_mean'] = ''
    conf_AM['S2std_std'] = ''
#     conf_AM['S2avg_cv'] = ''
    return conf_AM


# Marginals

# In[7]:


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


# Metrics

# In[8]:


def met(am):
    gev_fit1, gev_fit2, rho = marginalsAM('P_S1rol', 'P_S2rol', am)
    SS1sav_mean = np.mean(am['S1/S1savg'])
    SS1sav_std = np.std(am['S1/S1savg'])
    SS2sav_mean = np.mean(am['S2/S2savg'])
    SS2sav_std = np.std(am['S2/S2savg'])
    
    return gev_fit1, gev_fit2, rho, SS1sav_mean, SS1sav_std, SS2sav_mean, SS2sav_std


# In[ ]:




