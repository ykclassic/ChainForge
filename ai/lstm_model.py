import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


class LSTMForecaster:

    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()

    def train(self, series):

        data = series.values.reshape(-1,1)

        scaled = self.scaler.fit_transform(data)

        X, y = [], []

        for i in range(10, len(scaled)):
            X.append(scaled[i-10:i])
            y.append(scaled[i])

        X = np.array(X)
        y = np.array(y)

        self.model = Sequential([
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(1)
        ])

        self.model.compile(
            optimizer="adam",
            loss="mse"
        )

        self.model.fit(X, y, epochs=5, verbose=0)

    def predict(self, series):

        seq = self.scaler.transform(
            series.values[-10:].reshape(-1,1)
        )

        seq = seq.reshape(1,10,1)

        pred = self.model.predict(seq)

        return float(self.scaler.inverse_transform(pred)[0][0])
