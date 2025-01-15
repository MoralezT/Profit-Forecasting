# to interact with the operating system and variable access 
import os

#importing libraries  

#Python Data Analysis Library for data manipulation and analysis
import pandas as pd

# Numerical Python for numerical computations 
import numpy as np

# Matrix Laboratory-like Plotting Library for plots 
import matplotlib.pyplot as plt

# for Gradient Boosting 
from xgboost import XGBRegressor

# for linear regression 
from sklearn.linear_model import LinearRegression

# for feature scaling 
from sklearn.preprocessing import MinMaxScaler

# functions for accuracy score and metrics for evaluating model performance 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# for creating a sequential neural network model 
from tensorflow.keras.models import Sequential

# for building neural network architecture 
from tensorflow.keras.layers import Dense, LSTM

# for training control 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# for system specific parameters and functions 
import sys

file_path = "C:\Documents\DataAnalyticsPortfolio\ProfitForecastingModel\Superstoredata.csv"

# Load data
profit = pd.read_csv(file_path)
profit.head(10)

#dropping columns
profit = profit.drop(['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 
                   'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 
                   'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Sales'], axis=1)
#sales.info()

#Converting 'sale_date' column datatype from 'object' to a pandas DateTime-like format using pd.to_datetime(). 
profit['Order Date'] = pd.to_datetime(profit['Order Date'])
profit.info()

#Converting the 'sale_date' column to monthly periods
profit['Order Date'] = profit['Order Date'].dt.to_period("M")

#Grouping the sales data by the 'sale_date'(converted to monthly periods)
#Allows calculating the sum of sales for each month/aggregating sales data by month to get total sales for each month
monthly_profit = profit.groupby('Order Date').sum().reset_index()

#Converts each element in the 'sale_date' column to a pandas Timestamp
#Works only when already converted to a pandas DateTime-like format (datetime64[ns])
monthly_profit['Order Date'] = monthly_profit['Order Date'].dt.to_timestamp()

# monthly_profit.head(10) 
# monthly_profit.info()

# plot an empty graph of the sales data and set dimensions 15 width and 5 height
plt.figure(figsize=(15,5))
# insert the sales date and quantities to the plotted graph 
plt.plot(monthly_profit['Order Date'], monthly_profit['Profit'])
# assign date to the horizontal axis 
plt.xlabel("Date")
# assign quantities to the vertical axis 
plt.ylabel("Profit")
# provide a title to represent what is being plotted 
plt.title("Superstore Sales Data")
# display graph 
plt.show()

