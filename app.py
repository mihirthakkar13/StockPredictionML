import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


model = load_model("C:/Users/mihir/OneDrive/Desktop/Projects'UIC/StockPricePrediction/StockPrediction.keras")

st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Ticker', 'GOOG')
start = "2012-01-01"
end = "2022-12-31"

data = yf.download(stock,start,end)

st.subheader('Stock Data')
st.write(data)


# model training

data_train = data.Close[0:int(len(data)*0.8)]

data_test = data.Close[int(len(data)*0.8):]


scaler = MinMaxScaler(feature_range=(0,1))


pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days,data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)




# for moving average price vs 50

st.subheader('Price vs MA50')
ma_50_days = data['Close'].rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data.Close, 'g', label = 'Closing Price')
xlabel = 'Years'
ylabel = 'Stock Price'
plt.legend()
plt.show()
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


# dnfldnfdnd
x= []
y= []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])  # last 100 days data
    y.append(data_test_scale[i,0])
    
x, y = np.array(x), np.array(y)


predict  = model.predict(x)

scale = 1/scaler.scale_[0]

predict = predict * scale

y = y * scale 

# predicted value

st.subheader('Original Price Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label = 'Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4) 

