# Data-Driven

Data Driven is a machine learning web-app which allows parents of new drivers to decide the safest times and neighborhoods to take their new driver on the road.

Web app is available [here](http://bit.ly/insight_dd). Short presentation available [here](http://bit.ly/insight_ddd)

----

# Data

The city of Seattle keeps records of 200K auto-vehicle collisions which have taken place since 2004. These are tagged with logitude / latitude, weather, date/time. Importantly, they also include a severity score.

# Machine Learning

Data-driven uses two models.
  - **One-Class SVM**: This is an anomaly detection algorithm. I trained the OCSVM to detect the conditions under which fatal collisions are likely. Any condition deemed by this model to be fatal-like is listed as unsafe for all drivers.
  - **Kernel Density Estimation**: This algorithm estimates the probability distribution from which the data was chosen. Using this, along with slight adjustments for weather conditions, I predicted a tiered danger scale wherein conditions more dangerous than the median were avalable only to license-ready drivers and those with a bit of experience. The least dangerous conditions were available to beginning drivers.
  
  # Deployment
  
  I deployed this app using Flask and AWS.
