import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from datetime import date, timedelta
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import cred
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

st.title("Stock Price Prediction")
user_input = st.text_input('Enter Stock Ticker', 'TATAMOTORS.BSE')

d = timedelta(days=5000)
end = date.today()
start = end - d
print(end, start)

df = data.DataReader(user_input, 'av-daily', str(start), str(end), api_key=cred.api)
# Describing data
st.subheader('Data from {} - {}'.format(start, end))
st.write(df.describe())

df = df.reset_index()['close']
print(len(df))

# Visualisation
st.subheader("Closing Price v/s Time Chart")
fig = plt.figure(figsize=(12, 6))
plt.plot(df)
st.pyplot(fig)

st.subheader("Closing Price v/s Time Chart with 100 moving average")
ma100 = df.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df)
plt.plot(ma100, 'r')
st.pyplot(fig)

st.subheader("Closing Price v/s Time Chart with 100 & 200 moving average")
ma200 = df.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df)
st.pyplot(fig)

# Splitting data into training and testing
train_data_n = pd.DataFrame(df[0: int(len(df) * 0.75)])
test_data_n = pd.DataFrame(df[int(len(df) * 0.75): int(len(df))])


scaler = MinMaxScaler(feature_range=(0, 1))

train_data = scaler.fit_transform(np.array(train_data_n).reshape(-1, 1))
test_data = scaler.fit_transform(np.array(train_data_n).reshape(-1, 1))

time_step = cred.time_step

x_train = []
y_train = []
for i in range(time_step, train_data.shape[0]):
    x_train.append(train_data[i - time_step:i])
    y_train.append(train_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# LSTM Model
# ML Model
# '''
# model = Sequential()
# model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
# model.add(Dropout(0.2))

# model.add(LSTM(units=60, activation='relu', return_sequences=True))
# model.add(Dropout(0.3))

# model.add(LSTM(units=80, activation='relu', return_sequences=True))
# model.add(Dropout(0.4))

# model.add(LSTM(units=120, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(units=1))

# print(model.summary())

# print(x_train.shape, y_train.shape)

# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(x_train, y_train, epochs=50)
# '''
# model.save('keras_model.keras')


# Load my mode
model = load_model('keras_model.keras')

# Testing part
past_100_days = train_data_n.tail(time_step)

final_df = pd.concat([past_100_days, test_data_n], ignore_index=True)
print(past_100_days, final_df)

input_data = scaler.fit_transform(final_df)
print(input_data.shape)

x_test = []
y_test = []
for i in range(time_step, input_data.shape[0]):
    x_test.append(input_data[i - time_step:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test.shape, y_test.shape)
y_predicted = model.predict(x_test)

scaler_factor = 1 / scaler.scale_
y_predicted = scaler.inverse_transform(y_predicted)
y_test = y_test * scaler_factor

st.subheader("Prediction v/s Original")
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)

# Predict for next n days
days = int(st.text_input('For how many days you want to predict', 30))
n = days
p_100_days = test_data_n.tail(time_step)
print(p_100_days.shape)
i_data = scaler.fit_transform(p_100_days)
print(type(i_data))
i_data = i_data.reshape(1, time_step, 1)
print(i_data.shape)

final_predicted = []
for i in range(time_step, time_step + n):
    i_data = i_data.reshape(1, i, 1)
    a = model.predict(i_data[time_step - i:])
    i_data = np.append(i_data, a)
    final_predicted.append(a)

i_data = i_data.reshape(time_step+n, 1)
i_data = scaler.inverse_transform(i_data)


# Plotting next n days prediction
st.subheader('Next {} days prediction after {} in Time axis'.format(n, time_step))
fig3 = plt.figure(figsize=(12, 6))
plt.plot(i_data)
plt.xlabel("Time")
plt.ylabel("Price")
st.pyplot(fig3)
