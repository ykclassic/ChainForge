from sklearn.linear_model import LinearRegression

def regression_predict(X,y,last):

    model=LinearRegression()

    model.fit(X,y)

    return model.predict(last)[0]
