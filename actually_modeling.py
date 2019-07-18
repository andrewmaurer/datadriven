#################
## IMPORT MODULES
#################

import pandas as pd
import numpy as np

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

import os
import sys

import requests

imports = [pd, np, matplotlib, sns, requests]

for package in imports:
    print(f'{package.__name__} version: {package.__version__}')

############
## LOAD DATA
############

data = pd.read_csv('data/cleaned_data.csv')
data.date = pd.to_datetime(data.date)

#########################
## GET WEATHER CONDITIONS
#########################

weather_dict = {0:'clear', 1:'clear', 2:'rain', 3:'sleet', 4:'snow', 5:'fog', 6:'crosswind', 7:'sand', 8:np.nan, 10:'cloudy', 11:'snow', 12:'sleet', 98:np.nan, 99:np.nan}

path_base = '/home/amaurer/Documents/Insight/'
path_15 = f'{path_base}/data/NHTSA/FARS2015NationalCSV/accident.csv'
path_16 = f'{path_base}/data/NHTSA/FARS2016NationalCSV/accident.csv'
path_17 = f'{path_base}/data/NHTSA/FARS2017NationalCSV/accident.csv'

path_all = [path_15, path_16, path_17]
accident_all = [pd.read_csv(path) for path in path_all]

accident = pd.concat(accident_all, axis=0, join='outer', join_axes=None, ignore_index=False,
          keys=None, levels=None, names=None, verify_integrity=False,
          copy=True)

accident = accident[accident.STATE == 53]

weather_conditions = []

for idx, row in data.iterrows():
    lat, lon = row['lat'], row['lon']
    for _ , acc in accident.iterrows():
        if lat == acc['LATITUDE'] and lon == acc['LONGITUD']:
            weather_conditions.append(acc['WEATHER'])



