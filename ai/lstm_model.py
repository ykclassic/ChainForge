import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def lstm_predict(series):

    X=[]
    y=[]

    for i in range(10,len(series)):
        X.append(series[i-10:i])
        y.append(series[i])

    X=np.array(X)
    y=np.array(y)

    X=X.reshape(X.shape[0],X.shape[1],1)

    model=Sequential([
        LSTM(32),
        Dense(1)
    ])

    model.compile("adam","mse")
    model.fit(X,y,epochs=5,verbose=0)

    pred=model.predict(X[-1].reshape(1,10,1))[0][0]

    return pred
