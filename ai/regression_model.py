from sklearn.linear_model import LinearRegression

class PriceRegression:

    def train(self, df):

        X = df[["close","rsi"]]
        y = df["close"].shift(-1).dropna()

        X = X.iloc[:-1]

        self.model = LinearRegression()

        self.model.fit(X,y)

    def predict(self, df):

        x = df[["close","rsi"]].iloc[-1:]

        return float(self.model.predict(x)[0])
