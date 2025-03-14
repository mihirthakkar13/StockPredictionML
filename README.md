# StockPredictionML
This application is a Stock Market Predictor built using Python, Streamlit, and Keras. It enables users to visualize historical stock data, compare prices against moving averages, and predict future stock prices using a machine learning model.

How to Use the Application

Step 1: Enter a Stock Ticker

In the text box labeled "Enter Stock Ticker", type the stock ticker symbol (e.g., GOOG for Google, AAPL for Apple) and press Enter.

The app will automatically fetch data from Yahoo Finance for the selected stock from 2012-01-01 to 2022-12-31.

Step 2: View Stock Data

The application displays the fetched stock data, including:

Date

Closing Price

Opening Price

High Price

Low Price

Volume

Step 3: Visualizing Moving Averages

The app provides three visualizations to track stock trends:

Price vs MA50 (50-day moving average)

Price vs MA50 vs MA100 (50-day and 100-day moving averages)

Price vs MA100 vs MA200 (100-day and 200-day moving averages)

These moving averages help identify market trends and potential buy/sell signals.

Step 4: Predicted vs Actual Stock Prices

The model predicts future stock prices using a trained LSTM (Long Short-Term Memory) model.

The graph compares:

Red Line: Predicted Price

Green Line: Actual (Original) Price

This comparison helps users assess the model's accuracy and identify potential investment opportunities.


**#Technical Details:**

**Technologies Used**

Python for data processing and model implementation

Streamlit for the web interface

yFinance for retrieving stock data

Keras for the LSTM model used in prediction

MinMaxScaler for scaling data between 0 and 1, improving model performance

Matplotlib for visualizing stock trends and predictions

Data Preparation

The dataset is divided into Training (80%) and Testing (20%) sets.

The last 100 days of training data are added to the test set for smoother predictions.

Model Overview

Uses an LSTM (Long Short-Term Memory) neural network trained to predict stock prices based on past data.

The model learns patterns over a 100-day window to predict future prices.



