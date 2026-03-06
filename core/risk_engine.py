def generate_levels(price, direction, volatility):

    risk = price * volatility * 1.5

    if direction == "BUY":

        entry = price
        stop = price - risk
        tp = price + risk * 2

    else:

        entry = price
        stop = price + risk
        tp = price - risk * 2

    return entry, stop, tp
