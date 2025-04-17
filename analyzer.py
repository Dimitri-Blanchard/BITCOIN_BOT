import pandas as pd
import numpy as np
from datetime import datetime

from api_client import BinanceAPIClient
import analysis_utils
import indicators

class MarketAnalyzer:
    def __init__(self, api_client: BinanceAPIClient):
        self.api_client = api_client

    def _get_market_data(self, symbol="BTCUSDT", interval="1h", klines_limit=250, order_book_limit=100):
        klines_df = self.api_client.get_klines(symbol, interval, klines_limit)
        order_book = self.api_client.get_order_book(symbol, order_book_limit)
        ticker_24h = self.api_client.get_ticker_24h(symbol)

        return klines_df, order_book, ticker_24h

    def _calculate_order_book_pressure(self, order_book):
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            return 0.0, 0.0, 1.0

        bid_volume = sum(float(bid[1]) for bid in order_book['bids'][:10])
        ask_volume = sum(float(ask[1]) for ask in order_book['asks'][:10])

        ratio = bid_volume / ask_volume if ask_volume > 0 else float('inf')

        return bid_volume, ask_volume, ratio

    def _determine_trend(self, latest_indicators: pd.Series):
        trend = "neutre"
        sma20 = latest_indicators.get('sma_20', np.nan)
        sma50 = latest_indicators.get('sma_50', np.nan)
        sma200 = latest_indicators.get('sma_200', np.nan)

        if pd.isna(sma20) or pd.isna(sma50) or pd.isna(sma200):
            return "indéterminée (données manquantes)"

        if sma20 > sma50 > sma200:
            trend = "fortement haussière"
        elif sma20 > sma50:
            trend = "haussière"
        elif sma20 < sma50 < sma200:
            trend = "fortement baissière"
        elif sma20 < sma50:
            trend = "baissière"
        elif sma50 > sma200:
            trend = "plutôt haussière (fond)"
        elif sma50 < sma200:
             trend = "plutôt baissière (fond)"

        return trend

    def _generate_signals(self, indicators_df: pd.DataFrame, order_book_ratio: float):
        if indicators_df is None or indicators_df.empty:
            return []

        latest = indicators_df.iloc[-1]
        previous = indicators_df.iloc[-2] if len(indicators_df) > 1 else latest

        signals = []
        current_price = latest.get('close', np.nan)
        if pd.isna(current_price): return []

        rsi = latest.get('rsi', 50)
        if rsi < 30: signals.append({"indicator": "RSI", "signal": "survendu", "strength": "fort"})
        elif rsi < 40: signals.append({"indicator": "RSI", "signal": "potentiellement survendu", "strength": "modéré"})
        elif rsi > 70: signals.append({"indicator": "RSI", "signal": "suracheté", "strength": "fort"})
        elif rsi > 60: signals.append({"indicator": "RSI", "signal": "potentiellement suracheté", "strength": "modéré"})

        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        macd_hist = latest.get('macd_hist', 0)
        prev_macd_hist = previous.get('macd_hist', 0)
        if macd > macd_signal and macd_hist > 0 and macd_hist > prev_macd_hist: signals.append({"indicator": "MACD", "signal": "croisement bullish dynamique", "strength": "fort"})
        elif macd > macd_signal and macd_hist > 0: signals.append({"indicator": "MACD", "signal": "bullish", "strength": "modéré"})
        elif macd < macd_signal and macd_hist < 0 and macd_hist < prev_macd_hist: signals.append({"indicator": "MACD", "signal": "croisement bearish dynamique", "strength": "fort"})
        elif macd < macd_signal and macd_hist < 0: signals.append({"indicator": "MACD", "signal": "bearish", "strength": "modéré"})

        bb_upper = latest.get('bb_upper', np.inf)
        bb_lower = latest.get('bb_lower', -np.inf)
        bb_width = latest.get('bb_width', np.nan)
        avg_bb_width = indicators_df['bb_width'].rolling(20).mean().iloc[-1] if 'bb_width' in indicators_df.columns else np.nan
        if current_price > bb_upper: signals.append({"indicator": "Bollinger", "signal": "cassure haute (surachat)", "strength": "fort"})
        elif current_price < bb_lower: signals.append({"indicator": "Bollinger", "signal": "cassure basse (survente)", "strength": "fort"})
        if not pd.isna(bb_width) and not pd.isna(avg_bb_width) and bb_width < avg_bb_width * 0.7:
            signals.append({"indicator": "Bollinger", "signal": "compression (squeeze)", "strength": "modéré"})

        stoch_k = latest.get('stoch_k', 50)
        stoch_d = latest.get('stoch_d', 50)
        prev_stoch_k = previous.get('stoch_k', 50)
        prev_stoch_d = previous.get('stoch_d', 50)
        if stoch_k < 20 and stoch_d < 20: signals.append({"indicator": "Stochastique", "signal": "survendu", "strength": "fort"})
        elif stoch_k > 80 and stoch_d > 80: signals.append({"indicator": "Stochastique", "signal": "suracheté", "strength": "fort"})
        elif prev_stoch_k < prev_stoch_d and stoch_k > stoch_d and stoch_k < 50: signals.append({"indicator": "Stochastique", "signal": "croisement bullish", "strength": "modéré"})
        elif prev_stoch_k > prev_stoch_d and stoch_k < stoch_d and stoch_k > 50: signals.append({"indicator": "Stochastique", "signal": "croisement bearish", "strength": "modéré"})

        if order_book_ratio > 2.0: signals.append({"indicator": "Carnet d'ordres", "signal": "forte pression acheteuse", "strength": "fort"})
        elif order_book_ratio > 1.5: signals.append({"indicator": "Carnet d'ordres", "signal": "pression acheteuse", "strength": "modéré"})
        elif order_book_ratio < 0.5: signals.append({"indicator": "Carnet d'ordres", "signal": "forte pression vendeuse", "strength": "fort"})
        elif order_book_ratio < 0.7: signals.append({"indicator": "Carnet d'ordres", "signal": "pression vendeuse", "strength": "modéré"})

        return signals

    def _generate_recommendation(self, signals: list, trend: str, key_levels: dict):
        position = "NEUTRE"
        entry_price = key_levels.get("current_price", np.nan)
        stop_loss = None
        take_profit = None
        risk_reward = None
        confidence = "faible"

        if pd.isna(entry_price):
            return {"position": "ERREUR", "message": "Prix actuel non disponible"}

        bullish_score = 0
        bearish_score = 0
        strong_bullish = 0
        strong_bearish = 0

        signal_map = {
            "survendu": (2, True), "potentiellement survendu": (1, True),
            "bullish": (1, True), "croisement bullish dynamique": (2, True), "croisement bullish": (1, True),
            "pression acheteuse": (1, True), "forte pression acheteuse": (2, True),
            "cassure basse (survente)": (2, True),

            "suracheté": (2, False), "potentiellement suracheté": (1, False),
            "bearish": (1, False), "croisement bearish dynamique": (2, False), "croisement bearish": (1, False),
            "pression vendeuse": (1, False), "forte pression vendeuse": (2, False),
            "cassure haute (surachat)": (2, False),
        }

        for s in signals:
            score, is_bullish = signal_map.get(s['signal'], (0, None))
            strength_multiplier = 1.5 if s['strength'] == 'fort' else 1.0
            final_score = score * strength_multiplier

            if is_bullish is True:
                bullish_score += final_score
                if s['strength'] == 'fort': strong_bullish += 1
            elif is_bullish is False:
                bearish_score += final_score
                if s['strength'] == 'fort': strong_bearish += 1

        long_condition = (
            bullish_score > bearish_score * 1.2 and
            strong_bullish >= 1 and
            trend in ["haussière", "fortement haussière", "plutôt haussière (fond)"]
        ) or (strong_bullish >= 2 and trend != "fortement baissière")

        short_condition = (
            bearish_score > bullish_score * 1.2 and
            strong_bearish >= 1 and
            trend in ["baissière", "fortement baissière", "plutôt baissière (fond)"]
        ) or (strong_bearish >= 2 and trend != "fortement haussière")

        atr = key_levels.get("latest_atr", entry_price * 0.01)
        sl_multiplier = 2.5
        tp_multiplier = 4.0

        support_levels = key_levels.get("support_levels", [])
        resistance_levels = key_levels.get("resistance_levels", [])

        if long_condition:
            position = "LONG"
            confidence = "moyenne" if strong_bullish < 2 else "élevée"

            sl_atr = entry_price - atr * sl_multiplier
            closest_support = support_levels[0] if support_levels else -np.inf
            stop_loss = min(sl_atr, closest_support * 0.998) if closest_support > -np.inf else sl_atr
            stop_loss = max(stop_loss, entry_price * 0.95)

            tp_atr = entry_price + atr * tp_multiplier
            target_resistance = resistance_levels[1] if len(resistance_levels) > 1 else (resistance_levels[0] if resistance_levels else np.inf)
            take_profit = max(tp_atr, target_resistance * 0.998) if target_resistance < np.inf else tp_atr
            take_profit = min(take_profit, entry_price * 1.15)


        elif short_condition:
            position = "SHORT"
            confidence = "moyenne" if strong_bearish < 2 else "élevée"

            sl_atr = entry_price + atr * sl_multiplier
            closest_resistance = resistance_levels[0] if resistance_levels else np.inf
            stop_loss = max(sl_atr, closest_resistance * 1.002) if closest_resistance < np.inf else sl_atr
            stop_loss = min(stop_loss, entry_price * 1.05)

            tp_atr = entry_price - atr * tp_multiplier
            target_support = support_levels[1] if len(support_levels) > 1 else (support_levels[0] if support_levels else -np.inf)
            take_profit = min(tp_atr, target_support * 1.002) if target_support > -np.inf else tp_atr
            take_profit = max(take_profit, entry_price * 0.85)


        if position != "NEUTRE" and stop_loss is not None and take_profit is not None:
             try:
                 if position == "LONG":
                     risk = entry_price - stop_loss
                     reward = take_profit - entry_price
                 else:
                     risk = stop_loss - entry_price
                     reward = entry_price - take_profit

                 if risk > 1e-6:
                     risk_reward = round(reward / risk, 2)
                 else:
                     risk_reward = None

                 if risk_reward is not None and risk_reward < 1.0:
                     position = "NEUTRE"
                     stop_loss = None
                     take_profit = None
                     risk_reward = None
                     confidence = "faible (R:R < 1)"

             except Exception:
                 risk_reward = None


        return {
            "position": position,
            "entry_price": round(entry_price, 4),
            "stop_loss": round(stop_loss, 4) if stop_loss else None,
            "take_profit": round(take_profit, 4) if take_profit else None,
            "risk_reward_ratio": risk_reward,
            "confidence": confidence
        }


    def analyze_market(self, symbol="BTCUSDT", interval="1h"):
        klines_df, order_book, ticker_24h = self._get_market_data(symbol, interval)

        if klines_df is None or order_book is None or ticker_24h is None:
            return {"status": "error", "message": "Échec de la récupération des données du marché."}

        indicators_df = analysis_utils.calculate_all_indicators(klines_df)
        if indicators_df is None:
             return {"status": "error", "message": "Échec du calcul des indicateurs techniques."}

        latest_indicators = indicators_df.iloc[-1]

        key_levels = analysis_utils.identify_key_levels(indicators_df)

        bid_pressure, ask_pressure, order_book_ratio = self._calculate_order_book_pressure(order_book)

        trend = self._determine_trend(latest_indicators)

        signals = self._generate_signals(indicators_df, order_book_ratio)

        recommendation = self._generate_recommendation(signals, trend, key_levels)

        analysis_data = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "timeframe": interval,
            "status": "success",
            "market_data": {
                "current_price": key_levels.get("current_price"),
                "24h_change_percent": float(ticker_24h.get("priceChangePercent", 0)),
                "24h_volume": float(ticker_24h.get("volume", 0)),
                "key_levels": {
                    "support": key_levels.get("support_levels"),
                    "resistance": key_levels.get("resistance_levels")
                },
                "latest_atr": key_levels.get("latest_atr")
            },
            "technical_indicators": {
                k: round(float(v), 4) if isinstance(v, (int, float, np.number)) and not pd.isna(v) else (str(v) if not pd.isna(v) else None)
                for k, v in latest_indicators.items() if k not in ['open_time', 'close_time']
            },
            "order_book_analysis": {
                "bid_volume_top10": round(bid_pressure, 4),
                "ask_volume_top10": round(ask_pressure, 4),
                "buy_sell_ratio": round(order_book_ratio, 4) if order_book_ratio != float('inf') else 'inf'
            },
            "trend": trend,
            "signals": signals,
            "recommendation": recommendation
        }

        if recommendation['position'] != "NEUTRE" and recommendation['entry_price'] and recommendation['stop_loss'] and recommendation['take_profit']:
            entry = recommendation['entry_price']
            sl = recommendation['stop_loss']
            tp = recommendation['take_profit']
            if 'percentage_move' not in recommendation:
                recommendation['percentage_move'] = {}

            recommendation['percentage_move']['to_stop_loss'] = round((sl / entry - 1) * 100, 2) if entry else None
            recommendation['percentage_move']['to_take_profit'] = round((tp / entry - 1) * 100, 2) if entry else None
        elif 'recommendation' in analysis_data:
             analysis_data['recommendation']['percentage_move'] = None


        return analysis_data