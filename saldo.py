import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import calendar
from datetime import datetime, timedelta

# Load your saldo data
data = pd.read_csv('/home/PI/saldo_data.csv', delimiter=',')

# Preprocess your data: Convert dates to features
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.sort_index(inplace=True)

# Calculate daily changes in saldo
data['DailyChange'] = data['Saldo'].diff()

# Handle the first day of each month (add 4000 to the change)
first_days = data.index.to_series().dt.is_month_start
data.loc[first_days, 'DailyChange'] += 4000

# Fill NaN values (first row) with initial saldo change
data['DailyChange'].fillna(data['Saldo'].iloc[0], inplace=True)

# Add additional features
data['DayOfMonth'] = data.index.day
data['DayOfWeek'] = data.index.dayofweek  # Monday=0, Sunday=6

# Define features and target variable
X = data[['DayOfMonth', 'DayOfWeek']]
y = data['DailyChange']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
predictions = model.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))

# Function to predict saldo for the last day of the current month
def predict_last_day_of_current_month(model, data):
    today = datetime.today()
    current_month_saldo = data['Saldo'].at[today - timedelta(days=today.day-1)]
    days_remaining = calendar.monthrange(today.year, today.month)[1] - today.day

    # Prepare future dates for prediction
    future_dates = pd.DataFrame({
        'DayOfMonth': [today.day + i for i in range(days_remaining)],
        'DayOfWeek': [(today + timedelta(days=i)).weekday() for i in range(days_remaining)]
    })

    # Predict future daily changes
    future_changes = model.predict(future_dates)
    end_of_month_saldo = current_month_saldo + np.sum(future_changes)
    return round(end_of_month_saldo, 2)

# Predict end-of-month saldo
prediction = predict_last_day_of_current_month(model, data)
print("End of current month prediction:", prediction)