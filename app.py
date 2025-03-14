import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

from tensorflow.keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the trained Keras model
model = load_model("C:/Users/mihir/OneDrive/Desktop/Projects'UIC/StockPricePrediction/StockPrediction.keras")

# Streamlit App Title
st.header('Stock Market Predictor')

# Input for Stock Ticker with default value 'GOOG'
stock = st.text_input('Enter Stock Ticker', 'GOOG')
start = "2012-01-01"
end = "2022-12-31"

# Download stock data from Yahoo Finance
data = yf.download(stock, start, end)

# Display the downloaded data
st.subheader('Stock Data')
st.write(data)

# Split data into training (80%) and test (20%) sets
data_train = data.Close[0:int(len(data)*0.8)]
data_test = data.Close[int(len(data)*0.8):]

# Scale data between 0 and 1 for better model performance
scaler = MinMaxScaler(feature_range=(0, 1))

# Include the last 100 days from training data for continuity
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)

# Scale the combined test data
data_test_scale = scaler.fit_transform(data_test.values.reshape(-1, 1))

# Moving Average Plot (50-day MA)
st.subheader('Price vs MA50')
ma_50_days = data['Close'].rolling(50).mean()

# Visualization for Moving Average and Closing Price
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Closing Price')
plt.xlabel('Years')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(fig1)

# for moving average price vs 50 vs 100

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data['Close'].rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label = 'MA50')
plt.plot(ma_100_days, 'b', label = 'MA100')
plt.plot(data.Close, 'g', label = 'Closing Price')
xlabel = 'Years'
ylabel = 'Stock Price'
plt.legend()
plt.show()
st.pyplot(fig2) 

# for moving average price vs 100 vs 200

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data['Close'].rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r', label = 'MA100')
plt.plot(ma_200_days, 'b', label = 'MA200')
plt.plot(data.Close, 'g', label = 'Closing Price')
xlabel = 'Years'
ylabel = 'Stock Price'
plt.legend()
plt.show()
st.pyplot(fig3) 

# Prepare data for prediction (X: last 100 days data, Y: next day's price)
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])  # Collect the previous 100 days of data
    y.append(data_test_scale[i, 0])      # Target: Next day's closing price

# Convert lists to numpy arrays for model compatibility
x, y = np.array(x), np.array(y)

# Predict future prices using the loaded model
predict = model.predict(x)

# Rescale predicted and actual values back to the original scale
scale = 1 / scaler.scale_[0]  # Reverse scaling factor
predict = predict * scale
y = y * scale

# Visualization: Predicted Prices vs Original Prices
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)
