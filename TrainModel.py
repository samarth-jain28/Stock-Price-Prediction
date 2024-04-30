import numpy as np
import pandas as pd
import pandas_datareader as data
from datetime import date, timedelta
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import cred
from sklearn.preprocessing import MinMaxScaler


d = timedelta(days=5000)
end = date.today()
start = end - d
print(end, start)

df = data.DataReader("TATAMOTORS.BSE", 'av-daily', str(start), str(end), api_key=cred.api)
df = df.reset_index()['close']

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

# ML Model

model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

print(model.summary())

print(x_train.shape, y_train.shape)

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=100)

model.save('keras_model.keras')
