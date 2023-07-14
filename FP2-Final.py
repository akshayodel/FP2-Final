#!/usr/bin/env python
# coding: utf-8

# In[3]:


#pip install streamlit


# In[8]:


#pip install requests


# In[10]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from dateutil.relativedelta import relativedelta
from datetime import datetime

# API URL
api_url = "https://api.eia.gov/v2/international/data/?api_key=PYTvNry19FP8DgMdBAikT14IOtP21vBOZY2ZuDTJ&frequency=monthly&data[0]=value&facets[activityId][]=1&facets[productId][]=53&facets[countryRegionId][]=IND&facets[unit][]=TBPD&start=1990-01&end=2023-03&sort[0][column]=period&sort[0][direction]=asc&offset=0&length=5000"


# In[14]:


import requests


# In[15]:


#Read API data into a DataFrame
response = requests.get(api_url)
if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data['response']['data'])
    df['ds'] = pd.to_datetime(df['period'])
    df['y'] = df['value']
    data_series = df[['ds', 'y']]


# In[16]:



API_TOKEN = 'a122af78b08f7eccfe99605fe17fb2050d3116e1'

# Read CPI data from the specified URL
df_cpi = pd.read_csv(
    'https://www.econdb.com/api/series/CPIIN/?token=%s&format=csv' % API_TOKEN,
    parse_dates=['Date'])

# Merge the CPI dataframe and the data series dataframe
merged_df = df_cpi.merge(data_series, left_on='Date', right_on='ds', how='inner')

# Set the cutoff period for training
period = len(merged_df) - 6

# Select the training data
train_data = merged_df[['ds', 'CPIIN', 'y']][:period]

# Select the test data
test_data = merged_df[['ds', 'CPIIN', 'y']][period:]

# Define the ARIMA order
order = (1, 0, 0)

# Create and fit the ARIMAX model
model_arimax = ARIMA(train_data['y'], exog=train_data['CPIIN'], order=order)
model_arimax_fit = model_arimax.fit()


# In[17]:




# Define the prediction function
def predict_oil_production(months_diff):
    # Predict CPIIN
    predicted_cpiin = model_arimax_fit.predict(start=len(data), end=len(data) + months_diff - 1)
    predicted_cpiin = pd.DataFrame(predicted_cpiin, columns=['CPIIN'])
    
    # Predict oil production using the trained ARIMAX model
    exog_reshaped = predicted_cpiin['CPIIN'].values.reshape(-1, 1)
    predicted_values = model_arimax_fit.predict(start=len(merged_df)-6, end=len(merged_df)-6 + months_diff - 1, exog=exog_reshaped)
    predicted_values = pd.DataFrame(predicted_values, columns=['Oil_Production'])
    
    # Generate monthly dates
    start_date = merged_df['ds'].dt.date.max() - relativedelta(months=5)
    end_date = datetime.today().date() + relativedelta(months=4)
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # Create a DataFrame with the predicted values and dates
    predicted_df = pd.DataFrame({'Month': dates})
    predicted_df = pd.concat([predicted_df, predicted_values], axis=1)
    
    return predicted_df

# def predict_oil_production(months_diff):
#     # Predict CPIIN
#     predicted_cpiin = model_arimax_fit.predict(start=len(data), end=len(data) + months_diff - 1)
#     predicted_cpiin = pd.DataFrame(predicted_cpiin, columns=['CPIIN'])
    
#     # Predict oil production using the trained ARIMAX model
#     exog_reshaped = np.array(predicted_cpiin['CPIIN']).reshape(-1, 1)
#     #exog_reshaped = predicted_cpiin['CPIIN'].values.reshape(-1, 1)
#     predicted_values = model_arimax_fit.predict(start=len(merged_df)-6, end=len(merged_df)-6 + months_diff - 1, exog=exog_reshaped)
#     predicted_values = pd.DataFrame(predicted_values, columns=['Oil_Production'])
    
#     # Generate monthly dates
#     start_date = merged_df['ds'].dt.date.max() - relativedelta(months=5)
#     end_date = datetime.today().date() + relativedelta(months=4)
#     dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    
#     # Create a DataFrame with the predicted values and dates
#     predicted_df = pd.DataFrame({'Month': dates})
#     predicted_df = pd.concat([predicted_df, predicted_values], axis=1)
    
#     return predicted_df

# Create Streamlit widgets
st.header('Oil Production Forecasting')

start_date = st.date_input('Select Test Start Date', value=merged_df['ds'].dt.date.max() - relativedelta(months=5))
end_date = st.date_input('Select Test End Date', value=datetime.today().date() + relativedelta(months=4))
button_predict = st.button('Predict')

# Define the main Streamlit app logic
if button_predict:
    date_diff = end_date - start_date
    months_diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
    
    # Predict for the specified duration
    st.subheader('Oil Production Forecast: Custom Date')
    prediction_custom = predict_oil_production(months_diff)
    st.dataframe(prediction_custom)
    
    # Predict for the short-term duration (6 months)
    st.subheader('Oil Production Forecast: Short-Term Duration')
    prediction_short_term = predict_oil_production(6)
    st.dataframe(prediction_short_term)
    
    # Predict for the long-term duration (60 months)
    st.subheader('Oil Production Forecast: Long-Term Duration')
    prediction_long_term = predict_oil_production(60)
    st.dataframe(prediction_long_term)
    
    # Reset the end date
    end_date = start_date + relativedelta(days=date_diff.days)
    
    # Plot the oil production forecast
    st.subheader('Oil Production Forecast: Graph')
    plt.figure(figsize=(8, 4))
    plt.plot(prediction_custom['Month'], prediction_custom['Oil_Production'], label='Custom Date')
    plt.plot(prediction_short_term['Month'], prediction_short_term['Oil_Production'], label='Short-Term Duration')
    plt.plot(prediction_long_term['Month'], prediction_long_term['Oil_Production'], label='Long-Term Duration')
    plt.xlabel('Month')
    plt.ylabel('Oil Production')
    plt.title('Oil Production Forecast')
    plt.legend()
    st.pyplot(plt)


# In[ ]:




