import pandas as pd
import numpy as np
import indicators

def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame | None:
    required_periods = 200
    if df is None or len(df) < required_periods + 1:
        print(f"Données insuffisantes pour calculer les indicateurs (besoin d'au moins {required_periods+1} périodes).")
        return None

    df = df.copy()

    close_prices = df['close']
    high_prices = df['high']
    low_prices = df['low']
    volume = df['volume']

    df['rsi'] = indicators.calculate_rsi(close_prices, period=14)

    macd, macd_signal, macd_hist = indicators.calculate_macd(close_prices)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist

    upper, middle, lower = indicators.calculate_bollinger_bands(close_prices)
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'].replace(0, 1e-10)

    df['atr'] = indicators.calculate_atr(high_prices, low_prices, close_prices)

    stoch_k, stoch_d = indicators.calculate_stochastic(high_prices, low_prices, close_prices)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d

    df['mom'] = indicators.calculate_momentum(close_prices)

    df['sma_20'] = indicators.calculate_sma(close_prices, period=20)
    df['sma_50'] = indicators.calculate_sma(close_prices, period=50)
    df['sma_200'] = indicators.calculate_sma(close_prices, period=200)
    df['ema_9'] = indicators.calculate_ema(close_prices, period=9)

    df['obv'] = indicators.calculate_obv(close_prices, volume)

    df['swing_high'] = identify_swing_points(high_prices, is_high=True)
    df['swing_low'] = identify_swing_points(low_prices, is_high=False)

    return df

def identify_swing_points(data: pd.Series, window: int = 5, is_high: bool = True) -> pd.Series:
    swing_points = pd.Series(index=data.index, dtype=float)
    if len(data) < 2 * window + 1:
        return swing_points

    for i in range(window, len(data) - window):
        current_val = data.iloc[i]
        left_window = data.iloc[i-window : i]
        right_window = data.iloc[i+1 : i+window+1]

        is_swing = False
        if is_high:
            if not left_window.empty and not right_window.empty:
                 if current_val > left_window.max() and current_val > right_window.max():
                     is_swing = True
        else:
            if not left_window.empty and not right_window.empty:
                if current_val < left_window.min() and current_val < right_window.min():
                    is_swing = True

        if is_swing:
            swing_points.iloc[i] = current_val

    return swing_points

def identify_key_levels(df: pd.DataFrame) -> dict:
    min_length = 30
    if df is None or len(df) < min_length or 'close' not in df.columns or 'atr' not in df.columns:
        print(f"Données insuffisantes ou colonnes manquantes pour identifier les niveaux clés (besoin d'au moins {min_length} périodes et colonnes 'close', 'atr').")
        return {"resistance_levels": [], "support_levels": [], "current_price": np.nan, "latest_atr": np.nan}

    current_price = df['close'].iloc[-1]
    latest_atr = df['atr'].iloc[-1] if 'atr' in df.columns and not df['atr'].isnull().all() else 0.0

    if pd.isna(latest_atr) or latest_atr <= 0:
        latest_atr = current_price * 0.01

    swing_highs = df['swing_high'].dropna()
    swing_lows = df['swing_low'].dropna()

    def cluster_levels(levels, threshold_percent=0.5, current_price_ref=current_price):
        if levels.empty:
            return []

        threshold = current_price_ref * threshold_percent / 100
        clustered = []
        sorted_levels = sorted(levels.unique())

        if not sorted_levels:
            return []

        current_cluster = [sorted_levels[0]]
        for level in sorted_levels[1:]:
            if level - current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]

        if current_cluster:
            clustered.append(np.mean(current_cluster))

        return clustered

    threshold_percent = min(1.0, max(0.2, (latest_atr / current_price if current_price else 0.01) * 100 * 0.5))

    clustered_highs = cluster_levels(swing_highs, threshold_percent, current_price)
    clustered_lows = cluster_levels(swing_lows, threshold_percent, current_price)

    resistance_levels = sorted([level for level in clustered_highs if level > current_price])

    support_levels = sorted([level for level in clustered_lows if level < current_price], reverse=True)

    rolling_window = 20
    if len(resistance_levels) < 2 and len(df) > rolling_window:
        recent_highs = df['high'].rolling(rolling_window).max().dropna().unique()
        additional_resistances = sorted([h for h in recent_highs if h > current_price])
        combined_res = sorted(list(set(resistance_levels + additional_resistances)))
        resistance_levels = cluster_levels(pd.Series(combined_res), threshold_percent * 0.5, current_price)
        resistance_levels = [r for r in resistance_levels if r > current_price]

    if len(support_levels) < 2 and len(df) > rolling_window:
        recent_lows = df['low'].rolling(rolling_window).min().dropna().unique()
        additional_supports = sorted([l for l in recent_lows if l < current_price], reverse=True)
        combined_sup = sorted(list(set(support_levels + additional_supports)), reverse=True)
        support_levels = cluster_levels(pd.Series(combined_sup), threshold_percent * 0.5, current_price)
        support_levels = sorted([s for s in support_levels if s < current_price], reverse=True)

    max_levels = 5
    resistance_levels = resistance_levels[:max_levels]
    support_levels = support_levels[:max_levels]

    return {
        "resistance_levels": [round(r, 4) for r in resistance_levels],
        "support_levels": [round(s, 4) for s in support_levels],
        "current_price": float(current_price),
        "latest_atr": float(latest_atr)
    }