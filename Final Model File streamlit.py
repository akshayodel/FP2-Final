import json
import csv
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from dateutil.relativedelta import relativedelta
from datetime import datetime

st.set_option('deprecation.showPyplotGlobalUse', False)

# API URL
api_url = "https://api.eia.gov/v2/international/data/?api_key=PYTvNry19FP8DgMdBAikT14IOtP21vBOZY2ZuDTJ&frequency=monthly&data[0]=value&facets[activityId][]=1&facets[productId][]=53&facets[countryRegionId][]=IND&facets[unit][]=TBPD&start=1990-01&end=2023-03&sort[0][column]=period&sort[0][direction]=asc&offset=0&length=5000"
# Send GET request to the API
response = requests.get(api_url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Get the response data
    data = response.json()
    df = pd.DataFrame(data['response']['data'])

df['ds'] = pd.to_datetime(df['period'])
df['y'] = df['value']

data_series = df[['ds', 'y']]

API_TOKEN = 'a122af78b08f7eccfe99605fe17fb2050d3116e1'

df_cpi = pd.read_csv(
    'https://www.econdb.com/api/series/CPIIN/?token=%s&format=csv' % API_TOKEN,
    parse_dates=['Date']
)

merged_df = df_cpi.merge(data_series, left_on='Date', right_on='ds', how='inner')

# ARIMAX

from statsmodels.tsa.arima.model import ARIMA

# Defining period

period = len(merged_df) - 6

train_data = merged_df[['ds', 'CPIIN', 'y']][:period]
test_data = merged_df[['ds', 'CPIIN', 'y']][period:]

order = (1, 0, 0)  # ARIMA order
model_arimax = ARIMA(train_data['y'], exog=train_data['CPIIN'], order=order)
model_arimax_fit = model_arimax.fit()

# Predicting oil production
exog_reshaped = test_data['CPIIN'].values.reshape(-1, 1)
# Make predictions using the trained ARIMAX model
predicted_values_user = model_arimax_fit.predict(start=len(merged_df) - 6, end=len(merged_df) - 6 + 6 - 1, exog=exog_reshaped)

# Calculate MAPE
actual_values = test_data['y']
mape = round(np.mean(np.abs((actual_values - predicted_values_user) / actual_values)), 4)

# Calculate MAPE row-wise
mape_list = []
for i in range(357, 363):
    actual_value_1 = actual_values[i]
    predicted_value_1 = predicted_values_user[i]
    mape = round(np.abs((actual_value_1 - predicted_value_1) / actual_value_1), 4)
    mape_list.append(mape)

# Add the MAPE values as a new column to the DataFrame
test_data['MAPE'] = mape_list

Monitoring_data = test_data[['ds', 'MAPE']]
Monitoring_data.reset_index(drop=True, inplace=True)
Monitoring_data.to_csv('monitoring_data.csv', index=False)

# ARIMA to predict CPIIN

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Create a DataFrame with the date and value columns
data = merged_df[:period]
# Set the date column as the index
data.set_index('ds', inplace=True)
# Create the ARIMA model
model = ARIMA(data['CPIIN'], order=(1, 0, 0))  # Set the order (p, d, q) for ARIMA
# Fit the model
model_fit = model.fit()

# Voila

def prediction(months_diff, title):
    # Predicting CPIIN
    predicted_value_cpiin_user = pd.DataFrame(model_fit.predict(start=len(data), end=len(data) + months_diff - 1))
    predicted_value_cpiin_user['CPIIN'] = predicted_value_cpiin_user['predicted_mean']
    # Create a new index
    new_index = range(len(data), len(data) + months_diff)
    # Set the new index using set_index()
    predicted_value_cpiin_user.index = new_index

    # Predicting oil production
    exog_reshaped = predicted_value_cpiin_user['CPIIN'].values.reshape(-1, 1)
    # Make predictions using the trained ARIMAX model
    predicted_values_user = pd.DataFrame(
        model_arimax_fit.predict(start=len(merged_df) - 6, end=len(merged_df) - 6 + months_diff - 1, exog=exog_reshaped))
    predicted_values_user['Oil_Production'] = predicted_values_user['predicted_mean'].round(2)
    # Reset the index and start from 0
    predicted_values_user.reset_index(drop=True, inplace=True)

    # Generate the monthly dates within the specified range
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    # Create a DataFrame with the month column
    predicted_values_user['Month'] = pd.DataFrame({'Month': dates})

    predicted_values_user = predicted_values_user.drop(columns=["predicted_mean"])

    st.dataframe(predicted_values_user)

    # Plot the line graph
    plt.plot(predicted_values_user['Month'], predicted_values_user['Oil_Production'])
    # Add labels and title
    plt.xlabel('Month')
    plt.ylabel('Oil_Production')
    plt.title('Oil Production Forecast ' + title)
    # Show the plot
    st.pyplot()

st.title('Oil Production Forecast')



# Create a date widget
#start_date = st.date_input('Select Test Start Date', value=merged_df['ds'].dt.date.max() - relativedelta(months=5))
start_date = merged_df['ds'].dt.date.max() - relativedelta(months=5)
st.text('Start Date: ' + str(start_date))
# Create a date widget
end_date = st.date_input('Select End Date', value=datetime.today().date() + relativedelta(months=4))

# Create a button
button_send_2 = st.button('Predict')

if button_send_2:
    # Calculate the difference
    date_diff = end_date - start_date
    # Calculate the difference in months
    months_diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
    Short_Term_Duration = 6
    Long_Term_Duration = 60
    st.text("User Generated Prediction")
    prediction(months_diff, "User Generated Prediction")
    end_date = start_date + relativedelta(months=Short_Term_Duration)
    st.text("Short-Term Prediction")
    prediction(Short_Term_Duration, "Short-Term Prediction")
    end_date = start_date + relativedelta(months=Long_Term_Duration)
    st.text("Long-Term prediction")
    prediction(Long_Term_Duration, "Long-Term prediction")
    end_date = start_date + relativedelta(days=date_diff.days)
