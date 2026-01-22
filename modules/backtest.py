import pandas as pd
import numpy as np

def run_backtest(df, initial_capital=10000):
    """
    Vectorized backtesting engine for ChainForge Pro.
    """
    data = df.copy()
    
    # 1. Define Signals
    data['signal'] = 0
    # Buy when price < lower band and RSI < 30
    data.loc[(data['close'] <= data['lower_band']) & (data['rsi'] < 30), 'signal'] = 1
    # Sell when price > upper band or RSI > 70
    data.loc[(data['close'] >= data['upper_band']) | (data['rsi'] > 70), 'signal'] = -1
    
    # 2. Simulate Positions
    # Fill signal forward: if we buy (1), stay in until a sell (-1)
    data['position'] = data['signal'].replace(0, np.nan).ffill().fillna(0)
    data['position'] = data['position'].replace(-1, 0) # Convert "Sell" signal to "Flat"
    
    # 3. Calculate Returns
    data['pct_change'] = data['close'].pct_change()
    data['strategy_returns'] = data['position'].shift(1) * data['pct_change']
    
    # 4. Performance Metrics
    data['equity_curve'] = initial_capital * (1 + data['strategy_returns']).cumprod()
    
    total_return = (data['equity_curve'].iloc[-1] / initial_capital - 1) * 100
    win_rate = len(data[data['strategy_returns'] > 0]) / len(data[data['strategy_returns'] != 0]) if len(data[data['strategy_returns'] != 0]) > 0 else 0
    
    return {
        "df": data,
        "total_return": round(total_return, 2),
        "win_rate": round(win_rate * 100, 2),
        "final_value": round(data['equity_curve'].iloc[-1], 2)
    }
