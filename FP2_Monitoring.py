#!/usr/bin/env python
# coding: utf-8

# # Monitoring MAPE of Forecasted Data

# In[1]:


# importing libraries

import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#reading dataset

monitoring_data = pd.read_csv("monitoring_data.csv")


# In[3]:


#Calculating average MAPE

MAPE = round(monitoring_data['MAPE'].mean(),4)
print("Average MAPE: ",MAPE)


# In[4]:


print(monitoring_data.to_string(index=False))


# In[5]:


# Plot the line graph
monitoring_data.plot(x='ds', y='MAPE', figsize=(8, 4))

# Add labels and title
plt.xlabel('Month')
plt.ylabel('MAPE')
plt.title('MAPE')

# Show the plot
plt.show()


# In[ ]:




