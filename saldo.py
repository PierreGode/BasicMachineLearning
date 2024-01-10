import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import calendar
from datetime import datetime

# Function to predict saldo for the last day of the current month
def predict_last_day_of_current_month(model):
    today = datetime.today()
    year = today.year
    month = today.month
    last_day = calendar.monthrange(year, month)[1]  # Get the last day of the current month
    date = pd.Timestamp(year, month, last_day)
    day_of_month = date.day
    day_of_week = date.dayofweek  # Monday=0, Sunday=6
    prediction = model.predict([[day_of_month, day_of_week]])
    return round(prediction[0], 2)

# Load your saldo data with a custom delimiter (,)
data = pd.read_csv('/home/PI/saldo_data.csv', delimiter=',')

# Remove duplicate entries
data = data.drop_duplicates(subset=['Date', 'Saldo'])

# Preprocess your data: Convert dates to features
data['Date'] = pd.to_datetime(data['Date'])
data['DayOfMonth'] = data['Date'].dt.day
data['DayOfWeek'] = data['Date'].dt