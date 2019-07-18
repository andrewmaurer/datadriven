from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField
from wtforms.validators import DataRequired

class LoginForm(FlaskForm):
    weekdays = [(6,'Sunday'), (0, 'Monday'), (1, 'Tuesday'), (2, 'Wednesday'), (3, 'Thursday'), (4,'Friday'), (5,'Saturday')]
    times = [('0', '0:00'), ('1', '1:00'), ('2', '2:00'), ('3', '3:00'), ('4', '4:00'), ('5', '5:00'), ('6', '6:00'), ('7', '7:00'), ('8', '8:00'), ('9', '9:00'), ('10', '10:00'), ('11', '11:00'), ('12', '12:00'), ('13', '13:00'), ('14', '14:00'), ('15', '15:00'), ('16', '16:00'), ('17', '17:00'), ('18', '18:00'), ('19', '19:00'), ('20', '20:00'), ('21', '21:00'), ('22', '22:00'), ('23', '23:00')]
    weathers = [('Clear or Partly Cloudy', 'Clear'), ('Fog/Smog/Smoke','Foggy'), ('Snowing', 'Snowy'), ('Raining', 'Rainy')]
    nbhds = [('Alki', 'Alki'), ('Ballard', 'Ballard'), ('Beacon Hill', 'Beacon Hill'), ('Bitter Lake', 'Bitter Lake'), ('Capital Hill', 'Capital Hill'), ('Central', 'Central'), ('Columbia City', 'Columbia City'), ('Downtown', 'Downtown'), ('Fauntleroy', 'Fauntleroy'), ('First Hill', 'First Hill'), ('Fremont', 'Fremont'), ('Georgetown', 'Georgetown'), ('Green Lake', 'Green Lake'), ('Greenwood', 'Greenwood'), ('Laurelhurst', 'Laurelhurst'), ('Madison Park', 'Madison Park'), ('Madrona Park', 'Madrona Park'), ('Magnolia', 'Magnolia'), ('Magnuson', 'Magnuson'), ('Montlake', 'Montlake'), ('Mount Baker', 'Mount Baker'), ('Northgate', 'Northgate'), ('Queen Ann', 'Queen Ann'), ('Rainier Park', 'Rainier Park'), ('Revenna', 'Revenna'), ('University District', 'University District'), ('Wallingford', 'Wallingford'), ('West Seattle', 'West Seattle'), ('White Center', 'White Center')]

    weekday = SelectField('Select Day:', choices=weekdays, validators=[DataRequired()])
    time = SelectField('Select Time:', choices=times, validators=[DataRequired()])
    weather = SelectField('Select Weather:', choices=weathers, validators=[DataRequired()])
    nbhd = SelectField('Select Neighborhood:', choices=nbhds, validators=[DataRequired()])
    submit = SubmitField('Submit')