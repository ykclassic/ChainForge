import pandas as pd
import numpy as np

def run_backtest(df, initial_capital=10000, risk_free_rate=0.02):
    """
    Advanced vectorized backtesting engine with Sharpe Ratio and Max Drawdown.
    """
    data = df.copy()
    
    # 1. Signal Logic (Maintained from original)
    data['signal'] = 0
    data.loc[(data['close'] <= data['lower_band']) & (data['rsi'] < 30), 'signal'] = 1
    data.loc[(data['close'] >= data['upper_band']) | (data['rsi'] > 70), 'signal'] = -1
    
    # 2. Position Management
    data['position'] = data['signal'].replace(0, np.nan).ffill().fillna(0)
    data['position'] = data['position'].replace(-1, 0) 
    
    # 3. Returns Calculation
    data['pct_change'] = data['close'].pct_change()
    data['strategy_returns'] = data['position'].shift(1) * data['pct_change']
    data['equity_curve'] = initial_capital * (1 + data['strategy_returns']).fillna(0).cumprod()
    
    # 4. ADVANCED RISK METRICS
    # Total Return
    total_return = (data['equity_curve'].iloc[-1] / initial_capital - 1)
    
    # Annualized Sharpe Ratio
    # We assume 365 days for crypto; adjust sqrt factor if using 1h or 4h
    daily_returns = data['strategy_returns'].dropna()
    if len(daily_returns) > 1 and daily_returns.std() != 0:
        excess_returns = daily_returns - (risk_free_rate / 365)
        sharpe_ratio = np.sqrt(365) * (excess_returns.mean() / daily_returns.std())
    else:
        sharpe_ratio = 0.0

    # Maximum Drawdown (Peak to Trough)
    running_max = data['equity_curve'].cummax()
    drawdown = (data['equity_curve'] - running_max) / running_max
    max_drawdown = drawdown.min() # This is a negative number
    
    # Win Rate
    trades = data[data['strategy_returns'] != 0]
    win_rate = len(trades[trades['strategy_returns'] > 0]) / len(trades) if len(trades) > 0 else 0
    
    return {
        "df": data,
        "total_return_pct": round(total_return * 100, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "win_rate": round(win_rate * 100, 2),
        "final_value": round(data['equity_curve'].iloc[-1], 2)
    }
