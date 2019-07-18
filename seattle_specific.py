##########################
##### IMPORT MODULES #####
##########################

import pandas as pd
import numpy as np

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

import os
import sys

from joblib import dump, load

import folium
from folium import plugins

import sklearn
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.neighbors.kde import KernelDensity
from sklearn.preprocessing import OneHotEncoder

imports = [pd, np, matplotlib, sns, sklearn, folium]

for package in imports:
    print(f'{package.__name__} version: {package.__version__}')

########################
## THINGS TO DO LATER ##
########################



###########################
###### IMPORT DATA ########
###########################

data = pd.read_csv('data/seattle/Collisions.csv')

no_time = data[np.logical_not(data.INCDTTM.str.contains('M'))]
data = data[data.INCDTTM.str.contains('M')]

data.INATTENTIONIND = data.INATTENTIONIND.fillna('N')
data.UNDERINFL = data.UNDERINFL.fillna('N')
data.WEATHER = data.WEATHER.fillna('Clear or Partly Cloudy')
data.INCDTTM = pd.to_datetime(data.INCDTTM)


cols_to_keep = ['X', 'Y', 'ADDRTYPE', 'LOCATION', 'SEVERITYCODE', 'INCDTTM', 'WEATHER', 'LIGHTCOND']
data = data[cols_to_keep]
no_time = no_time[cols_to_keep]

data['DATE'] = data.INCDTTM.dt.date
data['DAY'] = data.INCDTTM.dt.dayofweek
data['DAYNAME'] = data.INCDTTM.dt.weekday_name
data['HOUR'] = data.INCDTTM.dt.hour
data['MONTH'] = data.INCDTTM.dt.month + data.INCDTTM.dt.day / 30

data = data[np.logical_not(data.SEVERITYCODE.isna())]

weathers_to_keep = ['Raining', 'Clear or Partly Cloudy', 'Snowing', 'Fog/Smog/Smoke']

data = data[data.WEATHER.isin(weathers_to_keep)]

###################
# IMPORT NBHD DATA
###################

data = pd.read_csv('data/with_nbhd.csv')
data.INCDTTM = pd.to_datetime(data.INCDTTM)
data['DayName'] = data.INCDTTM.dt.weekday_name
data.head()

data.INCDTTM = pd.to_datetime(data.INCDTTM)

##########
# PLOTTING
##########

collisionarray = data[data.SEVERITYCODE > 1][:10000][['Y','X']].dropna().as_matrix()

m = folium.Map([47.618651088818055, -122.33034059447267], zoom_start=11)
m.add_child(plugins.HeatMap(collisionarray, radius=15))

m

##########
# SEVERITY NUMBERING
##########

severity_dict = {'0':0, '1':1, '2':2, '2b':2, '3':3}

data.SEVERITYCODE = data.SEVERITYCODE.map(severity_dict)

##########
# SEVERITY PLOTTING
##########

sns.plt.xlim(-122.425, -122.250)
sns.plt.ylim(47.5, 47.8)

sns.scatterplot(x='X', y='Y', hue='SEVERITYCODE', data=data[np.logical_not(data.SEVERITYCODE.isin(['0', '1']))])

severity1 = data[data.SEVERITYCODE.isin(['1','2','2b','3'])]
severity2 = data[data.SEVERITYCODE.isin(['2','2b','3'])]
severity3 = data[data.SEVERITYCODE.isin(['3'])]

sns.kdeplot(data=severity3.X, data2=severity3.Y, cmap='Reds', shade=True, bw=0.15)

import holoviews as hv
from holoviews import opts, Cycle
hv.extension('bokeh')

hv.Bivariate(severity3[['X','Y']])

normal = np.random.randn(100,2)

sns.jointplot("X", "Y", data=data, kind="kde")

##########
# GEOPLOTTING
#########

shapefile = geopandas.read_file(fp)

def add_basemap(ax, zoom, url='http://tile.stamen.com/terrain/tileZ/tileX/tileY.png'):
    xmin, xmax, ymin, ymax = ax.axis()
    basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, url=url)
    ax.imshow(basemap, extent=extent, interpolation='bilinear')
    # restore original x/y limits
    ax.axis((xmin, xmax, ymin, ymax))

ax = shapefile.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
add_basemap(ax, zoom=10)

crs = {'init': 'epsg:2926'}

geometry = [Point(xy) for xy in zip(severity3.X, severity3.Y)]

geo_df = gpd.GeoDataFrame(severity3, crs=crs, geometry=geometry)

fig, ax = plt.subplots(figsize=(10,10))
shapefile.plot(ax=ax, alpha=0.4, color='grey')
geo_df.plot(ax=ax, markersize=20)

#########
# Add neighborhood
#########

nbhds = pd.read_csv('data/seattle/seattle_districts.csv')

row0 = data.iloc[0]
dists = (nbhds.lat - row0.Y) **2 + (nbhds.lon - row0.X) ** 2

def closest_neighborhood(some_row):
    lat = some_row.Y
    lon = some_row.X
    dists = nbhds['dist^2'] = (nbhds.lat - lat)**2 + (nbhds.lon - lon)**2
    return nbhds.loc[dists.values.argmin()]['neighborhood']

calculated_closest = [closest_neighborhood(the_row) for idx, the_row in data.iterrows()]

data['NBHD'] = calculated_closest

sns.scatterplot(x='X', y='Y', hue='NBHD', data=data)

#############
## Count Weathered Days
##############

weather_by_day = data.groupby(['DATE', 'WEATHER']).size().unstack().idxmax(axis=1)
weather_by_day.value_counts()

############
# Visualization
############

day = 3
hour = 14
weather = 'Raining'

selected_data = data[ np.logical_and.reduce((data.DAY == day, data.HOUR == hour, data.WEATHER == weather))]

sns.scatterplot(x='X', y='Y', hue='SEVERITYCODE', data=selected_data)

##############
# Write to disk
##############

data.to_csv('with_nbhd.csv', index=False)

##############
# MODEL
##############

sample_columns = ['DAY', 'HOUR', 'WEATHER', 'NBHD', 'SEVERITYCODE']
sample_data = data[sample_columns]

# OneClassSVM

svm_data = sample_data[sample_data.SEVERITYCODE == 3].drop('SEVERITYCODE', axis=1)

model = svm.OneClassSVM()
%time model.fit(svm_data)

%time pd.value_counts(model.predict(svm_data))

%time pd.value_counts(model.predict(sample_data[sample_data.SEVERITYCODE < 3]))

preds = model.predict(sample_data[sample_data.SEVERITYCODE < 3])

predicted_dangerous = data[data.SEVERITYCODE < 3][preds == 1]
predicted_dangerous.groupby('DAY').size()
predicted_dangerous.groupby('NBHD').size()
pd_percent = predicted_dangerous.groupby('HOUR').size() / predicted_dangerous.groupby('HOUR').size().sum() * 100

sns.lineplot(x=list(range(1,23)), y=pd_percent)

predicted_safe = data[data.SEVERITYCODE < 3][preds == -1]

predicted_safe.groupby('DAY').size()
predicted_safe.groupby('NBHD').size()
ps_percent = predicted_safe.groupby('HOUR').size() / predicted_safe.groupby('HOUR').size().sum() * 100

sns.lineplot(x=list(range(24)), y=ps_percent)

tmp = pd.DataFrame(predicted_safe.groupby('HOUR').size(), columns=['Safe'])
tmp['Dangerous'] = predicted_dangerous.groupby('HOUR').size()
tmp.Dangerous = tmp.Dangerous.fillna(0)

tmp['Safety_Index'] = tmp.Safe / (tmp.Safe + tmp.Dangerous) * 100
tmp['Danger_Index'] = tmp.Dangerous / (tmp.Safe + tmp.Dangerous) * 100

tmp[['Dangerous', 'Safe']].plot()

sns.set()
col = 'DAY'
tmp = pd.DataFrame(predicted_safe.groupby(col).size(), columns=['Safe'])
tmp['Dangerous'] = predicted_dangerous.groupby(col).size()
tmp.Safe = tmp.Safe.fillna(0)
tmp.Dangerous = tmp.Dangerous.fillna(0)
tmp['Danger_Index'] = tmp.Dangerous / (tmp.Safe + tmp.Dangerous)
tmp[['Danger_Index']].plot()

data.groupby('DAY').size()

# Density Estimation

kde_data = sample_data[sample_data.SEVERITYCODE > 1].drop('SEVERITYCODE', axis=1)
n = round(len(kde_data) * 0.8)

train_data = kde_data[:n]
test_data = kde_data[n:]

kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(train_data)

test_ss = kde.score_samples(test_data)

proportions = []

for num in np.arange(test_ss.min(), test_ss.max(), 0.1):
    proportions.append(1 - np.sum(test_ss > num) / len(test_ss))

sns.set()
fig = sns.lineplot(np.arange(test_ss.min(), test_ss.max(), 0.1), proportions)
fig.get_figure().savefig('test.png')

# Some plotting

days = ['Sunday', 'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
grouped = data.groupby(['DAY','SEVERITYCODE']).size()
grouped.index = days

sns.barplot(x=grouped.index, y=grouped)
plt.xticks(rotation=75)

day = 6
hour = 11
weather = 'Clear or Partly Cloudy'
month = 6 + 14/30

#####################
# Productionalizing I guess
###################

sample_data = data

# One-hot encoding

cat_dat = sample_data[['WEATHER','NBHD']]
num_dat = sample_data[['SEVERITYCODE', 'DAY', 'HOUR']]

enc = OneHotEncoder()
enc.fit(cat_dat)


num_np = num_dat.values
cat_np = enc.transform(cat_dat).toarray()
all_data = pd.DataFrame(np.concatenate([num_np, cat_np], axis=1))

# SVM

svm_data = all_data[all_data[0] == 3].drop(0, axis=1)

svm_model = svm.OneClassSVM()
svm_model.fit(svm_data)

# KDE

kde_data = all_data[all_data[0] > 1].drop(0, axis=1)
n = round( 0.8 * len(kde_data))

kde_model = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(kde_data[:n])

# Dump the encode and two models

dump([enc, svm_model, kde_model], 'enc_svm_kde.joblib')

# Practice loading the models

enc1, svm1, kde1 = load('enc_svm_kde.joblib')

day = 1
hour = 23
weather = 'Clear or Partly Cloudy'
nbhd = 'Central'

processed = [[day, hour] + enc1.transform([[weather, nbhd]]).toarray()[0].tolist()]

svm1.predict(processed)
kde1.score_samples(processed)

#################
# Validate SVM
#################

svm_valid = all_data[np.logical_and(all_data[0] < 3, all_data[0] > 1)].drop(0, axis=1)

svm_preds = svm_model.predict(svm_valid)

pred_safe = svm_valid[svm_preds == -1]
pred_dang = svm_valid[svm_preds == 1]

# by day

col = 1
tmp = pd.DataFrame(pred_safe.groupby(col).size(), columns=['Safe'])
tmp['Dangerous'] = pred_dang.groupby(col).size()
tmp.Safe = tmp.Safe.fillna(0)
tmp.Dangerous = tmp.Dangerous.fillna(0)
tmp['Danger_Index'] = tmp.Dangerous / (tmp.Safe + tmp.Dangerous)/ 2.5

tmp['Day'] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

sns.set()
sns.lineplot(x='Day', y='Danger_Index', sort=False, data=tmp)
plt.xticks(rotation=45)

# by hour

plt.xticks(rotation=45)

######################
# VALIDATION
######################

day = list(range(1,8))
hour = list(range(24))
weather_len = 4
nbhd_len = 29

################
# Treat Hours and Days Categorically
################

cols = ['SEVERITYCODE', 'WEATHER', 'DAYNAME', 'HOUR', 'NBHD']

train_data = data[cols]

enc = OneHotEncoder()
enc.fit(train_data.drop('SEVERITYCODE', axis=1))

# test it

enc.transform([['Raining', 'Friday', 6, 'Central']]).toarray()

# SVM

svm_data = train_data[train_data.SEVERITYCODE == 3].drop('SEVERITYCODE', axis=1)

svm_model = svm.OneClassSVM()
svm_model.fit(enc.transform(svm_data))

svm_valid = train_data[train_data.SEVERITYCODE < 3].drop('SEVERITYCODE', axis=1)

preds = svm_model.predict(enc.transform(svm_valid))

pred_safe = train_data[train_data.SEVERITYCODE < 3][preds == -1]
pred_dang = train_data[train_data.SEVERITYCODE < 3][preds == 1]

pred_dang.groupby(by='HOUR').size() / (pred_dang.groupby(by='HOUR').size() + pred_safe.groupby(by='HOUR').size())

# KDE

kde_data = train_data[np.logical_and(train_data.SEVERITYCODE < 3, train_data.SEVERITYCODE > 1)].drop('SEVERITYCODE', axis=1)

n = round( 0.8 * len(kde_data))

kde_model = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(enc.transform(kde_data[:n]).toarray())

kde_preds = kde_model.score_samples(enc.transform(kde_data[n:]).toarray())

np.quantile(kde_preds, 0.5)
np.quantile(kde_preds, 0.75)

dump([enc, svm_model, kde_model], 'enc_svm_kde_2.joblib')

# A little more validation

svm_data.groupby(by='DAYNAME').size() / svm_data.groupby(by='DAYNAME').size().sum()



