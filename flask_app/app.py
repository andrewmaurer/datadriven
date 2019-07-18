from flask import Flask, render_template, flash, redirect, request, url_for
from forms import LoginForm
import flask

import io
import random

import pandas as pd
import numpy as np

# import matplotlib
# from matplotlib import pyplot as plt
# import seaborn as sns

import os
import sys

import joblib
from joblib import dump, load

import sklearn
from sklearn import svm
from sklearn.neighbors.kde import KernelDensity
from sklearn.preprocessing import OneHotEncoder

modules = [flask, pd, np, joblib, sklearn]

for mod in modules:
    print(f'{mod.__name__}: {mod.__version__} ')

#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#from matplotlib.figure import Figure
#####################
# Some modeling functions
#####################

day_to_num = {'Sunday':6, 'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5}

weather_dict = {'1':(1,0,0,0), '2': (0,1,0,0), '3':(0,0,1,0), '4':(0,0,0,1)}
# load('/home/amaurer/Documents/Insight/flask_app/static/enc_svm_kde.joblib')

#data = pd.read_csv('/home/amaurer/Documents/Insight/flask_app/static/with_nbhd.csv')

####################
# LOAD IN THE DATA
####################

enc, svm, kde = load('static/enc_svm_kde_2.joblib')

####################
# FLASK STUFF
####################

app = Flask(__name__)    

app.config['SECRET_KEY'] = 'any secret string'

def process(theday, thetime, theweather, thenbhd):
    days_to_name = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday', -1:-1}
    print(f'Weather: {theweather}')
    print(f'Time: {thetime}')
    print(f'NBHD: {thenbhd}')
    #print(f'Day: {days_to_name[theday]}')

    return [enc.transform([[theweather, days_to_name[int(theday)], int(thetime), thenbhd]]).toarray()[0].tolist()]

def find_better_day(thetime, theday, theweather, thenbhd, svm_pred, kde_pred):
    days = [0,1,2,3,4,5,6]
    days_to_name = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday', -1:-1}

    best_day = -1
    best_kde = -100

    for day in days:
        proc = process(day, thetime, theweather, thenbhd)
        this_svm = svm.predict(proc)[0]
        this_kde = kde.score_samples(proc)[0]

        if this_svm == -1 and this_kde > best_kde:
            best_day = day
            best_kde = this_kde
    
    print(f'best day: {best_day}')
    print(f'the day:  {theday}')

    if int(best_day) == int(theday):
        return -1
    else:
        return days_to_name[best_day]


def find_better_hour(thetime, theday, theweather, thenbhd, svm_pred, kde_pred):
    hours = list(range(8,18))

    best_hour = -1
    best_kde = -100

    for hour in hours:
        proc = process(theday, hour, theweather, thenbhd)
        this_svm = svm.predict(proc)[0]
        this_kde = kde.score_samples(proc)[0]

        if this_svm == -1 and this_kde > best_kde:
            best_hour = hour
            best_kde = this_kde

    if best_hour == -1 or best_hour == thetime:
        return best_hour
    else:
        return str(hour) + ':00'

def find_better_nbhd(thetime, theday, theweather, thenbhd, svm_pred, kde_pred):
    nbhds = ['Alki', 'Ballard', 'Beacon Hill', 'Bitter Lake', 'Capital Hill', 'Central', 'Columbia City', 'Downtown', 'Fauntleroy', 'First Hill', 'Fremont', 'Georgetown', 'Green Lake', 'Greenwood', 'Laurelhurst', 'Madison Park', 'Madrona Park', 'Magnolia', 'Magnuson', 'Montlake', 'Mount Baker', 'Northgate', 'Queen Ann', 'Rainier Park', 'Revenna', 'University District', 'Wallingford', 'West Seattle', 'White Center']

    best_nbhd = -1
    best_kde = -100

    for nbhd in nbhds:
        proc = process(theday, thetime, theweather, nbhd)
        this_svm = svm.predict(proc)[0]
        this_kde = kde.score_samples(proc)[0]

        if this_svm == -1 and this_kde > best_kde:
            best_nbhd = nbhd
            best_kde = this_kde

        print(f'{nbhd}: SVM predicts {this_svm} and KDE predicts {this_kde}.')

    if best_nbhd == thenbhd:
        return -1
    else:
        return best_nbhd

@app.route("/", methods=['GET', 'POST'])
def home():
    form = LoginForm()
    if form.validate_on_submit():
        flash('test')
        return redirect('/results')
    return render_template('index.html', form=form)
    
@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        result = request.form
        thetime = result['time']
        theday = result['weekday']
        theweather = result['weather']
        thenbhd = result['nbhd']

        processed = process(theday, thetime, theweather, thenbhd)
        
        #[[theday, thetime] + enc.transform([[theweather, thenbhd]]).toarray()[0].tolist()]

        weather_scale = {'Clear or Partly Cloudy':0, 'Fog/Smog/Smoke':5.25, 'Snowing':4.79, 'Raining':1.2}

        svm_pred = svm.predict(processed)[0]
        kde_pred = kde.score_samples(processed)[0] + weather_scale[theweather]

        better_day = find_better_day(thetime, theday, theweather, thenbhd, svm_pred, kde_pred)
        better_hour = find_better_hour(thetime, theday, theweather, thenbhd, svm_pred, kde_pred)
        better_nbhd = find_better_nbhd(thetime, theday, theweather, thenbhd, svm_pred, kde_pred)

        # plt.scatter(x=range(10), y=[num**2 for num in range(10)])
        # plt.savefig('/home/amaurer/Documents/Insight/flask_app/static/out.png')
        # plt.close()
        return render_template("results.html",result = result, svm_pred=svm_pred, kde_pred=kde_pred, processed=processed, better_day=better_day, better_hour=better_hour, better_nbhd=better_nbhd)
    return render_template('results.html')


if __name__ == "__main__":
    app.run(debug=True)
