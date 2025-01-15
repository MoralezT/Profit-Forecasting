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
#profit.info()

#Converting 'sale_date' column datatype from 'object' to a pandas DateTime-like format using pd.to_datetime(). 
profit['Order Date'] = pd.to_datetime(profit['Order Date'])
#profit.info()

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
# plt.figure(figsize=(15,5))
# # insert the sales date and quantities to the plotted graph 
# plt.plot(monthly_profit['Order Date'], monthly_profit['Profit'])
# # assign date to the horizontal axis 
# plt.xlabel("Date")
# # assign quantities to the vertical axis 
# plt.ylabel("Profit")
# # provide a title to represent what is being plotted 
# plt.title("Superstore Sales Data")
# # display graph 
# plt.show()

# apply diff() function to difference the QTY values 
# create a new column to store the differenced quantities 
monthly_profit['profit_diff'] = monthly_profit['Profit'].diff()
# drop the rows with missing values caused by differencing  
monthly_profit = monthly_profit.dropna()
# show new column 
# monthly_profit.head(5)
# #monthly_profit.info()

# plot figure and set dimensions 
# plt.figure(figsize = (15,5))
# # insert data into graph 
# plt.plot(monthly_profit['Order Date'], monthly_profit['Profit'])
# # x axis label
# plt.xlabel('Date')
# # y axis label 
# plt.ylabel('Profit')
# # title the graph 
# plt.title('Profit difference')
# # display graph 
# plt.show()

# drop the existing columns and create new dataframe 
# fill this with differenced values for supervised training 
supervised_data = monthly_profit.drop(['Order Date', 'Profit'], axis=1)
# loop over a range from 1-12 starting at 1 and stopping at 13
for i in range (1,13):
    # create columns to represent the months
    # use an iterator to concatenate months
    col_name = 'month_' + str(i)
    # creating lagged values for predictions 
    supervised_data[col_name] = supervised_data['profit_diff'].shift(i)
# drop NaN values and reset index of dataframe 
supervised_data =supervised_data.dropna().reset_index(drop=True)
#supervised_data.head(5)

# train the model with all data excluding last 50 rows
train_data = supervised_data[:-12]
# save the last 50 rows for testing 
test_data = supervised_data[-12:]
# print('Train data shape', train_data.shape)
# print('Test data shape', test_data.shape)


# scale features to a specified range for better performance 
scaler = MinMaxScaler(feature_range=(-1,1))
# training the scaler to compute the min and max values of training features
scaler.fit(train_data)
# apply scaling to training data 
train_data = scaler.transform(train_data)
# apply scaling to testing data 
test_data = scaler.transform(test_data)

# for training - assign all lagged predictor values to X and response values to y
X_train, y_train = train_data[:,1:], train_data[:,0:1]
# for testing - assign all lagged predictor values to X and response values to y
X_test, y_test = test_data[:,1:], test_data[:,0:1]
# converting the response variable from 2d to 1d array for compatability with scikit-learn model
y_train = y_train.ravel()
y_test = y_test.ravel()
# print("X_train_shape", X_train.shape)
# print("y_train_shape", y_train.shape)
# print("X_test_shape", X_test.shape)
# print("y_test_shape", y_test.shape)

# extract last 12 rows of dates and reset index to start from 0
profit_dates = monthly_profit['Order Date'][-12:].reset_index(drop=True)
# create new dataframe with one column to store sales dates only 
# later predicted target values for y will also be added to this dataframe
predictions_df = pd.DataFrame(profit_dates)

# extract last 12 rows of actual sales quantities (target variable)
# store in a new variable and create a list 
act_profit = monthly_profit['Profit'][-12:].to_list()
#print(act_profit)

# Build Linear regression model and create an instance of the LR model
lr_model = LinearRegression()
# train the model 
lr_model.fit(X_train, y_train)
# generate predictions for the response variable 
# based on the input features of the test aata
predictions = lr_model.predict(X_test)
#print(predictions)

# reshape predictions into column vector 
predictions = predictions.reshape(-1,1)
# create matrix with input features of X test set and predicted output
predicted_test_set = np.concatenate([predictions, X_test], axis =1)
# reverse back into original scale 
predicted_test_set = scaler.inverse_transform(predicted_test_set)
#print(predicted_test_set)

# create a new list to store the predictions vs actual profit
result =[]
# loop through and sum all predicted and actual profit values
for index in range(0,len(predicted_test_set)):
    result.append(predicted_test_set[index][0] + act_profit[index])
# create a series (one dimensional labelled array) from the list of predictions
predicted_series = pd.Series(result, name="Linear Prediction")
# merge the series of predictions with the data frame containing the profit dates
predictions_df = predictions_df.merge(predicted_series, left_index = True, right_index = True)
#print(predictions_df)

#calculate average squared difference between predicted and true profit
MSE = np.sqrt(mean_squared_error(predictions_df["Linear Prediction"], monthly_profit['Profit'][-12:]))
# calculate the average absolute difference between predicted and true profit
MAE = mean_absolute_error(predictions_df["Linear Prediction"], monthly_profit['Profit'][-12:])
# calculate how well the model fit the data
R2 = r2_score(predictions_df["Linear Prediction"], monthly_profit['Profit'][-12:])

# print("Linear Regression Mean Squared Error: ", MSE)
# print("Linear Regression Mean Absolute Error: ", MAE)
# print("Linear Regression R2: ", R2)

# set the dimensions for the plot figure 
# plt.figure(figsize=(15,5))
# #Actual profit
# plt.plot(monthly_profit['Order Date'], monthly_profit['Profit'])
# #Predicted profit
# plt.plot(predictions_df['Order Date'], predictions_df['Linear Prediction'])
# # give the plot a title 
# plt.title("Superstore Profit Forecast")
# # label for the Date axis (x axis) horizontal
# plt.xlabel("Date")
# # label for y axis (profit) vertical
# plt.ylabel("Profit")
# # create a marker to show which color line represents predicted or actual values 
# plt.legend(["Actual profit", "Predicted profit"])
# # display the graph 
# plt.show()

# Extract actual and predicted values of y
true_labels = monthly_profit['Profit'][-12:]
predicted_labels = predictions_df['Linear Prediction']

# Create scatter plot
# plt.figure(figsize=(8, 6))
# plt.scatter(true_labels, predicted_labels, color='blue')

# # Add diagonal line for reference (y = x)
# plt.plot(true_labels, true_labels, color='red', linestyle='--')

# # Label the axes and add a title
# plt.xlabel('Actual Profit')
# plt.ylabel('Predicted Profit')
# plt.title('Actual vs Predicted Profit')

# # Add a legend
# plt.legend(['Reference Line', 'Predicted vs Actual'])

# # Display plot
# plt.show()

# Extract actual and predicted values of y
predicted_labels = predictions_df['Linear Prediction']
true_labels = monthly_profit['Profit'][-12:].reset_index(drop=True)

# Calculate residuals
residuals = true_labels - predicted_labels

# Create a residuals plot
# plt.figure(figsize=(8, 6))
# plt.scatter(true_labels, residuals, color='green')

# # Add a horizontal line at y=0 for reference
# plt.axhline(y=0, color='red', linestyle='--')

# # Label the axes and add a title
# plt.xlabel('True Labels')
# plt.ylabel('Residuals')
# plt.title('Residuals Plot')

# # Add legend
# plt.legend(['Reference Line', 'Residuals'])

# # Display the plot
# plt.show()