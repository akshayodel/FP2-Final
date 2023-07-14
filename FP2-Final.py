import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# API URL
api_url = "https://api.eia.gov/v2/international/data/?api_key=PYTvNry19FP8DgMdBAikT14IOtP21vBOZY2ZuDTJ&frequency=monthly&data[0]=value&facets[activityId][]=1&facets[productId][]=53&facets[countryRegionId][]=IND&facets[unit][]=TBPD&start=1990-01&end=2023-03&sort[0][column]=period&sort[0][direction]=asc&offset=0&length=5000"

# Send GET request to the API
response = requests.get(api_url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Get the response data
    data = response.json()
    df = pd.DataFrame(data['response']['data'])
else:
    st.error("Failed to retrieve data. Status code: " + str(response.status_code))

# Convert 'period' column to datetime format and assign it to 'ds' column
df['ds'] = pd.to_datetime(df['period'])

# Assign 'value' column to 'y' column
df['y'] = df['value']

# Create a new DataFrame with columns 'ds' and 'y'
data_series = df[['ds', 'y']]

API_TOKEN = 'a122af78b08f7eccfe99605fe17fb2050d3116e1'

# Read CPI data from the specified URL and parse the 'Date' column as dates
df_cpi = pd.read_csv(
    'https://www.econdb.com/api/series/CPIIN/?token=%s&format=csv' % API_TOKEN,
    parse_dates=['Date'])

# Merge the CPI dataframe (df_cpi) and the data series dataframe (data_series)
# based on the common columns 'Date' and 'ds', using an inner join
merged_df = df_cpi.merge(data_series, left_on='Date', right_on='ds', how='inner')

# Streamlit App
st.title("Oil Production Forecast")

# Select the test start date
start_date = st.date_input("Select Test Start Date", value=(datetime.now() - relativedelta(months=5)).date())

# Select the test end date
end_date = st.date_input("Select Test End Date", value=(datetime.now() + relativedelta(months=4)).date())

# Calculate the difference in months between the end date and start date
months_diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1

# Calculate the difference in months between the end date and start date for short-term and long-term durations
short_term_duration = 6
long_term_duration = 60

# Function to predict oil production
def predict_oil_production(months_diff, title):
    # Calculate the number of rows to slice for training data
    period = len(merged_df) - months_diff

    # Select the columns 'ds', 'CPIIN', and 'y' from the merged DataFrame
    # and slice the rows from the beginning up to 'period' (exclusive) to create the training data
    train_data = merged_df[['ds', 'CPIIN', 'y']][:period]

    # Define the ARIMA order
    order = (1, 0, 0)  # ARIMA order

    # Create an ARIMAX model with the specified order, using 'y' as the endogenous variable
    # and 'CPIIN' as the exogenous variable in the training data
    model_arimax = ARIMA(train_data['y'], exog=train_data['CPIIN'], order=order)

    # Fit the ARIMAX model to the training data
    model_arimax_fit = model_arimax.fit()

    # Predicting CPIIN
    predicted_value_cpiin = pd.DataFrame(model_arimax_fit.predict(start=len(merged_df), end=len(merged_df) + months_diff - 1,
                                                                  exog=train_data['CPIIN'][-months_diff:].values.reshape(-1, 1)))
    predicted_value_cpiin['CPIIN'] = predicted_value_cpiin['predicted_mean']
    # Create a new index
    new_index = range(len(merged_df), len(merged_df) + months_diff)
    # Set the new index using set_index()
    predicted_value_cpiin.index = new_index

    # Predicting oil production
    exog_reshaped = predicted_value_cpiin['CPIIN'].values.reshape(-1, 1)
    # Make predictions using the trained ARIMAX model
    predicted_values = pd.DataFrame(model_arimax_fit.predict(start=len(merged_df) - 6, end=len(merged_df) - 6 + months_diff - 1,
                                                             exog=exog_reshaped))
    predicted_values['Oil_Production'] = predicted_values['predicted_mean'].round(2)
    # Reset the index and start from 0
    predicted_values.reset_index(drop=True, inplace=True)

    # Generate the monthly dates within the specified range
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    # Create a DataFrame with the month column
    predicted_values['Month'] = pd.DataFrame({'Month': dates})

    predicted_values = predicted_values.drop(columns=["predicted_mean"])
    predicted_values = predicted_values[['Month', 'Oil_Production']]

    st.write(predicted_values.to_string(index=False))

    # Plot the line graph
    plt.figure(figsize=(8, 4))
    plt.plot(predicted_values['Month'], predicted_values['Oil_Production'])
    plt.xlabel('Month')
    plt.ylabel('Oil Production')
    plt.title(title)
    plt.xticks(rotation=45)
    st.pyplot()

# Predict for custom date range
st.header("Oil Production Forecast: Custom Date")
predict_oil_production(months_diff, "Oil Production Forecast: Custom Date")

# Predict for short-term duration
end_date_short_term = start_date + relativedelta(months=short_term_duration)
st.header("Oil Production Forecast: Short-Term Duration")
predict_oil_production(short_term_duration, "Oil Production Forecast: Short-Term Duration")

# Predict for long-term duration
end_date_long_term = start_date + relativedelta(months=long_term_duration)
st.header("Oil Production Forecast: Long-Term Duration")
predict_oil_production(long_term_duration, "Oil Production Forecast: Long-Term Duration")
