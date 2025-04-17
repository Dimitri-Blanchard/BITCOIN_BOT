import pandas as pd
import numpy as np

def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    if period <= 0 or len(data) < period:
        return pd.Series(index=data.index, dtype=float)
    return data.rolling(window=period, min_periods=period).mean()

def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    if period <= 0 or len(data) < period:
        return pd.Series(index=data.index, dtype=float)
    return data.ewm(span=period, adjust=False, min_periods=period).mean()

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    if period <= 0 or len(data) < period + 1:
        return pd.Series(index=data.index, dtype=float)

    delta = data.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)

    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
    if len(data) < slow_period or len(data) < fast_period or fast_period <= 0 or slow_period <= 0 or signal_period <=0 or fast_period >= slow_period:
         return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)

    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    macd_histogram = macd_line - signal_line

    return macd_line, signal_line, macd_histogram

def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2):
    if period <= 0 or std_dev <= 0 or len(data) < period:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    middle_band = calculate_sma(data, period)
    rolling_std = data.rolling(window=period, min_periods=period).std()

    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)

    return upper_band, middle_band, lower_band

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
    if k_period <= 0 or d_period <= 0 or len(close) < k_period:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()

    range_diff = highest_high - lowest_low
    range_diff = range_diff.replace(0, 1e-10)

    k = 100 * ((close - lowest_low) / range_diff)

    d = calculate_sma(k, d_period)

    return k, d

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    if period <= 0 or len(close) < period + 1:
        return pd.Series(index=close.index, dtype=float)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = calculate_sma(tr, period)

    return atr

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    if len(close) == 0 or len(volume) == 0 or len(close) != len(volume):
        return pd.Series(dtype=float)

    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0] if len(obv) > 0 else np.nan

    close_change = close.diff()

    for i in range(1, len(close)):
        if close_change.iloc[i] > 0:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close_change.iloc[i] < 0:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]

    return obv

def calculate_momentum(data: pd.Series, period: int = 10) -> pd.Series:
    if period <= 0 or len(data) < period + 1:
        return pd.Series(index=data.index, dtype=float)

    shifted_data = data.shift(period)
    momentum = data / shifted_data.replace(0, 1e-10) * 100
    return momentum