#!/usr/bin/env python
# coding: utf-8

# # Monitoring Performance of Forecasted Data (Oil Production)

# In[22]:


# importing libraries

import pandas as pd
import matplotlib.pyplot as plt


# In[30]:


#reading dataset
monitoring_data = pd.read_csv("monitoring_data.csv")


# In[31]:


#Calculating average MAPE

MAPE = round(monitoring_data['MAPE'].mean(),4)
print("Average MAPE: ",MAPE)


# In[32]:


print(monitoring_data[['ds','MAPE']].to_string(index=False))


# In[33]:


# Plot the line graph
monitoring_data.plot(x='ds', y='MAPE', figsize=(8, 4))

# Add labels and title
plt.xlabel('Month')
plt.ylabel('MAPE')
plt.title('MAPE with threshold 5%')
# Define threshold value
threshold = 0.05

# Plot the threshold line
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')

# Show the plot
plt.show()


# In[34]:


# Calculate the average of all previous values
previous_avg = monitoring_data['y'].iloc[:-1].mean()

# Get the last value of the column
last_value = monitoring_data['y'].iloc[-1]


# In[35]:


# Compare the last value with the average
if abs(last_value/previous_avg-1) > 0.1:
    print("There is drift in actual data. Model might require retraining")
else:
    print("There is no drift in actual data. Model does not require retraining")


# In[36]:


# Check if any value in the column is greater than 0.05
if (monitoring_data['MAPE'] > 0.05).any():
    print("MAPE is exceeding threshold, need to retrain model.")
else:
    print("MAPE is below threshold, no need to retrain model.")


# # Monitoring Performance of Forecasted Data (CPI)

# In[38]:


#reading dataset
cpi_monitoring_data = pd.read_csv("cpi_monitoring_data.csv")


# In[39]:


#Calculating average MAPE

MAPE = round(cpi_monitoring_data['MAPE'].mean(),4)
print("Average MAPE: ",MAPE)


# In[41]:


print(cpi_monitoring_data[['ds','MAPE']].to_string(index=False))


# In[48]:


# Plot the line graph
cpi_monitoring_data.plot(x='ds', y='MAPE', figsize=(8, 4))

# Add labels and title
plt.xlabel('Month')
plt.ylabel('MAPE')
plt.title('MAPE with threshold 5%')
# Define threshold value
threshold = 0.05

# Plot the threshold line
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')

# Show the plot
plt.show()


# In[50]:


# Calculate the average of all previous values
previous_avg = cpi_monitoring_data['CPIIN'].iloc[:-1].mean()

# Get the last value of the column
last_value = cpi_monitoring_data['CPIIN'].iloc[-1]


# In[51]:


# Compare the last value with the average
if abs(last_value/previous_avg-1) > 0.1:
    print("There is drift in actual data. Model might require retraining")
else:
    print("There is no drift in actual data. Model does not require retraining")


# In[52]:


# Check if any value in the column is greater than 0.05
if (cpi_monitoring_data['MAPE'] > 0.05).any():
    print("MAPE is exceeding threshold, need to retrain model.")
else:
    print("MAPE is below threshold, no need to retrain model.")


# In[ ]:




