import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import calendar
from datetime import datetime

# Function to predict saldo for the last day of the current month
def predict_last_day_of_current_month(model, start_saldo, X_future):
    # Predict daily changes
    daily_changes = model.predict(X_future)

    # Accumulate the changes to get the end of month saldo
    end_of_month_saldo = start_saldo + np.sum(daily_changes)
    return round(end_of_month_saldo, 2)

# Load your saldo data
data = pd.read_csv('/home/PI/saldo_data.csv', delimiter=',')

# Remove duplicate entries
data = data.drop_duplicates(subset=['Date', 'Saldo'])

# Preprocess your data: Convert dates to features and calculate daily changes
data['Date'] = pd.to_datetime(data['Date'])
data['DayOfMonth'] = data['Date'].dt.day
data['DayOfWeek'] = data['Date'].dt.dayofweek  # Monday=0, Sunday=6
data['DailyChange'] = data['Saldo'].diff().fillna(0)  # Calculate daily change in saldo

# Define your features and target variable
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

# Prepare data for end-of-month prediction
today = datetime.today()
start_of_month = today.replace(day=1)
start_saldo = data.loc[start_of_month]['Saldo'] + 4000  # Starting saldo for the month
days_in_month = calendar.monthrange(today.year, today.month)[1]
future_dates = pd.DataFrame({'DayOfMonth': range(today.day, days_in_month + 1),
                             'DayOfWeek': [(today + pd.Timedelta(days=i)).weekday() for i in range(days_in_month - today.day + 1)]})

# Predict end-of-month saldo
end_of_month_prediction = predict_last_day_of_current_month(model, start_saldo, future_dates)
print("End of current month prediction:", end_of_month_prediction)