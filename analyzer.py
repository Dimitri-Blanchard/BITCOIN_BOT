import pandas as pd
import numpy as np
from datetime import datetime
import time
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

from api_client import BinanceAPIClient
import analysis_utils
import indicators

                         
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("analyzer")

class MarketAnalyzer:
    def __init__(self, api_client: BinanceAPIClient, cache_ttl: int = 60):
                   
        self.api_client = api_client
        self.cache_ttl = cache_ttl
        self._cache = {}
        self._last_fetch_time = {}
        
    def _generate_signals(self, indicators_df: pd.DataFrame, order_book_ratio: float) -> List[Dict]:
                   
        if indicators_df is None or indicators_df.empty:
            logger.warning("Aucun indicateur disponible pour générer des signaux")
            return []

                                                          
        if len(indicators_df) < 2:
            logger.warning("Historique insuffisant pour générer des signaux")
            return []

        latest = indicators_df.iloc[-1]
        previous = indicators_df.iloc[-2]
        
        signals = []
        current_price = latest.get('close', np.nan)
        if pd.isna(current_price):
            logger.warning("Prix actuel non disponible")
            return []

                                                      
        rsi = latest.get('rsi', 50)
        prev_rsi = previous.get('rsi', 50)
        
                                  
        if rsi < 25: 
            signals.append({"indicator": "RSI", "signal": "survendu extrême", "strength": "très fort"})
        elif rsi < 30: 
            signals.append({"indicator": "RSI", "signal": "survendu", "strength": "fort"})
        elif rsi < 40: 
            signals.append({"indicator": "RSI", "signal": "potentiellement survendu", "strength": "modéré"})
        elif rsi > 75: 
            signals.append({"indicator": "RSI", "signal": "suracheté extrême", "strength": "très fort"})
        elif rsi > 70: 
            signals.append({"indicator": "RSI", "signal": "suracheté", "strength": "fort"})
        elif rsi > 60: 
            signals.append({"indicator": "RSI", "signal": "potentiellement suracheté", "strength": "modéré"})
        
                         
        if len(indicators_df) > 10:
            price_slope = (current_price - indicators_df['close'].iloc[-10]) > 0
            rsi_slope = (rsi - indicators_df['rsi'].iloc[-10]) > 0
            
            if price_slope and not rsi_slope and rsi > 60:
                signals.append({"indicator": "RSI", "signal": "divergence baissière", "strength": "fort"})
            elif not price_slope and rsi_slope and rsi < 40:
                signals.append({"indicator": "RSI", "signal": "divergence haussière", "strength": "fort"})

                                       
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        macd_hist = latest.get('macd_hist', 0)
        prev_macd_hist = previous.get('macd_hist', 0)
        
                                                         
        if macd > macd_signal and macd_hist > 0:
            strength = "modéré"
            signal_type = "bullish"
            
                                     
            if macd_hist > prev_macd_hist and prev_macd_hist > 0:
                signal_type = "croisement bullish dynamique"
                strength = "fort"
            elif macd_hist > 0 and prev_macd_hist <= 0:
                signal_type = "croisement bullish recent"
                strength = "fort"
                
            signals.append({"indicator": "MACD", "signal": signal_type, "strength": strength})
            
        elif macd < macd_signal and macd_hist < 0:
            strength = "modéré"
            signal_type = "bearish"
            
                                     
            if macd_hist < prev_macd_hist and prev_macd_hist < 0:
                signal_type = "croisement bearish dynamique"
                strength = "fort"
            elif macd_hist < 0 and prev_macd_hist >= 0:
                signal_type = "croisement bearish recent"
                strength = "fort"
                
            signals.append({"indicator": "MACD", "signal": signal_type, "strength": strength})

                                               
        bb_upper = latest.get('bb_upper', np.inf)
        bb_lower = latest.get('bb_lower', -np.inf)
        bb_width = latest.get('bb_width', np.nan)
        
                                                          
        avg_bb_width = indicators_df['bb_width'].rolling(20).mean().iloc[-1] if 'bb_width' in indicators_df.columns else np.nan
        
                             
        if current_price > bb_upper * 1.005:                         
            signals.append({"indicator": "Bollinger", "signal": "cassure haute forte (surachat)", "strength": "très fort"})
        elif current_price > bb_upper:
            signals.append({"indicator": "Bollinger", "signal": "cassure haute (surachat)", "strength": "fort"})
        elif current_price < bb_lower * 0.995:                         
            signals.append({"indicator": "Bollinger", "signal": "cassure basse forte (survente)", "strength": "très fort"})
        elif current_price < bb_lower:
            signals.append({"indicator": "Bollinger", "signal": "cassure basse (survente)", "strength": "fort"})
        
                                             
        if not pd.isna(bb_width) and not pd.isna(avg_bb_width):
            if bb_width < avg_bb_width * 0.7:
                signals.append({"indicator": "Bollinger", "signal": "compression forte (squeeze)", "strength": "fort"})
            elif bb_width < avg_bb_width * 0.85:
                signals.append({"indicator": "Bollinger", "signal": "compression (squeeze)", "strength": "modéré"})
            elif bb_width > avg_bb_width * 1.5:
                signals.append({"indicator": "Bollinger", "signal": "expansion forte (volatilité élevée)", "strength": "fort"})
            elif bb_width > avg_bb_width * 1.2:
                signals.append({"indicator": "Bollinger", "signal": "expansion (volatilité)", "strength": "modéré"})

                                        
        stoch_k = latest.get('stoch_k', 50)
        stoch_d = latest.get('stoch_d', 50)
        prev_stoch_k = previous.get('stoch_k', 50)
        prev_stoch_d = previous.get('stoch_d', 50)
        
                                      
        if stoch_k < 20 and stoch_d < 20:
            strength = "fort" if stoch_k < 10 and stoch_d < 10 else "modéré"
            signals.append({"indicator": "Stochastique", "signal": "survendu", "strength": strength})
        elif stoch_k > 80 and stoch_d > 80:
            strength = "fort" if stoch_k > 90 and stoch_d > 90 else "modéré"
            signals.append({"indicator": "Stochastique", "signal": "suracheté", "strength": strength})
        
                                   
        if prev_stoch_k < prev_stoch_d and stoch_k > stoch_d:
            strength = "fort" if stoch_k < 30 else "modéré"
            signals.append({"indicator": "Stochastique", "signal": "croisement bullish", "strength": strength})
        elif prev_stoch_k > prev_stoch_d and stoch_k < stoch_d:
            strength = "fort" if stoch_k > 70 else "modéré"
            signals.append({"indicator": "Stochastique", "signal": "croisement bearish", "strength": strength})

                       
        if 'obv' in indicators_df.columns:
            current_obv = latest.get('obv', np.nan)
            prev_obv = previous.get('obv', np.nan)
            
            if not pd.isna(current_obv) and not pd.isna(prev_obv):
                                                                     
                if len(indicators_df) >= 10:
                    obv_10_periods_ago = indicators_df['obv'].iloc[-10]
                    obv_trend = current_obv - obv_10_periods_ago
                    price_trend = current_price - indicators_df['close'].iloc[-10]
                    
                                         
                    if obv_trend > 0 and price_trend < 0:
                        signals.append({"indicator": "Volume", "signal": "divergence OBV haussière", "strength": "fort"})
                    elif obv_trend < 0 and price_trend > 0:
                        signals.append({"indicator": "Volume", "signal": "divergence OBV baissière", "strength": "fort"})
                
                                          
                obv_change_pct = (current_obv - prev_obv) / abs(prev_obv) if prev_obv != 0 else 0
                if obv_change_pct > 0.05:                 
                    signals.append({"indicator": "Volume", "signal": "accumulation forte", "strength": "fort"})
                elif obv_change_pct < -0.05:
                    signals.append({"indicator": "Volume", "signal": "distribution forte", "strength": "fort"})

                                            
        if order_book_ratio > 2.5:
            signals.append({"indicator": "Carnet d'ordres", "signal": "déséquilibre acheteur extrême", "strength": "très fort"})
        elif order_book_ratio > 2.0:
            signals.append({"indicator": "Carnet d'ordres", "signal": "forte pression acheteuse", "strength": "fort"})
        elif order_book_ratio > 1.5:
            signals.append({"indicator": "Carnet d'ordres", "signal": "pression acheteuse", "strength": "modéré"})
        elif order_book_ratio < 0.4:
            signals.append({"indicator": "Carnet d'ordres", "signal": "déséquilibre vendeur extrême", "strength": "très fort"})
        elif order_book_ratio < 0.5:
            signals.append({"indicator": "Carnet d'ordres", "signal": "forte pression vendeuse", "strength": "fort"})
        elif order_book_ratio < 0.7:
            signals.append({"indicator": "Carnet d'ordres", "signal": "pression vendeuse", "strength": "modéré"})

                                                
        sma20 = latest.get('sma_20', np.nan)
        sma50 = latest.get('sma_50', np.nan)
        sma200 = latest.get('sma_200', np.nan)
        
        if not pd.isna(sma20) and not pd.isna(sma50):
                                             
            prev_sma20 = previous.get('sma_20', np.nan) 
            prev_sma50 = previous.get('sma_50', np.nan)
            
            if prev_sma20 < prev_sma50 and sma20 > sma50:
                signals.append({"indicator": "MoyennesMobiles", "signal": "golden cross 20/50", "strength": "fort"})
            elif prev_sma20 > prev_sma50 and sma20 < sma50:
                signals.append({"indicator": "MoyennesMobiles", "signal": "death cross 20/50", "strength": "fort"})
                
                              
            if current_price < sma20 * 0.95:
                signals.append({"indicator": "MoyennesMobiles", "signal": "prix très inférieur à SMA20", "strength": "modéré"})
            elif current_price > sma20 * 1.05:
                signals.append({"indicator": "MoyennesMobiles", "signal": "prix très supérieur à SMA20", "strength": "modéré"})

                                           
        atr = latest.get('atr', np.nan)
        if not pd.isna(atr) and not pd.isna(current_price):
            atr_percent = (atr / current_price) * 100
            
            if atr_percent > 5:
                signals.append({"indicator": "Volatilité", "signal": "volatilité extrême", "strength": "très fort"})
            elif atr_percent > 3:
                signals.append({"indicator": "Volatilité", "signal": "volatilité élevée", "strength": "fort"})
            elif atr_percent < 1:
                signals.append({"indicator": "Volatilité", "signal": "volatilité basse", "strength": "modéré"})

                                   
        strength_values = {"très fort": 4, "fort": 3, "modéré": 2, "faible": 1}
        signals.sort(key=lambda x: strength_values.get(x["strength"], 0), reverse=True)
        
        return signals

    def _get_cached_or_fetch(self, key: str, fetch_func, *args, **kwargs) -> Any:
                   
        current_time = time.time()
        if key in self._cache and (current_time - self._last_fetch_time.get(key, 0)) < self.cache_ttl:
            return self._cache[key]
        
        try:
            data = fetch_func(*args, **kwargs)
            if data is not None:
                self._cache[key] = data
                self._last_fetch_time[key] = current_time
            return data
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données pour {key}: {str(e)}")
            return self._cache.get(key)                                                 
    
    def _get_market_data(self, symbol: str = "BTCUSDT", interval: str = "1h", 
                         klines_limit: int = 250, order_book_limit: int = 100) -> Tuple[pd.DataFrame, dict, dict]:
                   
                                                          
        klines_key = f"klines_{symbol}_{interval}"
        klines_df = self._get_cached_or_fetch(
            klines_key, 
            self.api_client.get_klines, 
            symbol, interval, klines_limit
        )
        
        order_book_key = f"order_book_{symbol}"
        order_book = self._get_cached_or_fetch(
            order_book_key,
            self.api_client.get_order_book,
            symbol, order_book_limit
        )
        
        ticker_key = f"ticker_24h_{symbol}"
        ticker_24h = self._get_cached_or_fetch(
            ticker_key,
            self.api_client.get_ticker_24h,
            symbol
        )
        
        return klines_df, order_book, ticker_24h

    def _calculate_order_book_pressure(self, order_book: Dict) -> Tuple[float, float, float]:
                   
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            logger.warning("Carnet d'ordres vide ou incomplet")
            return 0.0, 0.0, 1.0
        
                                                          
                                                                  
        depth_levels = [5, 10, 20]
        bid_volumes = []
        ask_volumes = []
        
        for depth in depth_levels:
            actual_depth = min(depth, len(order_book['bids']), len(order_book['asks']))
            
                                                                                      
            bid_vol = sum(float(bid[1]) * (1 - 0.01 * i) for i, bid in enumerate(order_book['bids'][:actual_depth]))
            ask_vol = sum(float(ask[1]) * (1 - 0.01 * i) for i, ask in enumerate(order_book['asks'][:actual_depth]))
            
            bid_volumes.append(bid_vol)
            ask_volumes.append(ask_vol)
        
                                                             
        bid_volume = sum(bid_volumes) / len(bid_volumes)
        ask_volume = sum(ask_volumes) / len(ask_volumes)
        
                                  
        ratio = bid_volume / max(ask_volume, 1e-10)
        
        return bid_volume, ask_volume, ratio

    def _determine_trend(self, indicators_df: pd.DataFrame) -> Dict[str, str]:
                   
        if indicators_df is None or indicators_df.empty:
            return {"primary": "indéterminée", "detail": "données insuffisantes"}
        
        latest = indicators_df.iloc[-1]
        
                                             
        sma20 = latest.get('sma_20', np.nan)
        sma50 = latest.get('sma_50', np.nan)
        sma200 = latest.get('sma_200', np.nan)
        
                                                
        macd = latest.get('macd', np.nan)
        macd_signal = latest.get('macd_signal', np.nan)
        rsi = latest.get('rsi', np.nan)
        
                                          
        stoch_k = latest.get('stoch_k', np.nan)
        stoch_d = latest.get('stoch_d', np.nan)
        obv_slope = 0
        
        if 'obv' in indicators_df.columns and len(indicators_df) > 20:
            obv_current = indicators_df['obv'].iloc[-1]
            obv_past = indicators_df['obv'].iloc[-20]
            obv_slope = (obv_current - obv_past) / 20 if obv_past != 0 else 0
        
                                         
        trends = {"primary": "neutre", "timeframes": {}}
        
                                       
        if pd.notna(sma20) and pd.notna(sma50) and pd.notna(sma200):
            if sma20 > sma50 > sma200:
                trends["primary"] = "fortement haussière"
                trends["timeframes"]["long_term"] = "haussière"
            elif sma20 < sma50 < sma200:
                trends["primary"] = "fortement baissière"
                trends["timeframes"]["long_term"] = "baissière"
            elif sma50 > sma200:
                trends["timeframes"]["long_term"] = "plutôt haussière"
            elif sma50 < sma200:
                trends["timeframes"]["long_term"] = "plutôt baissière"
        
                                
        if pd.notna(sma20) and pd.notna(sma50):
            if sma20 > sma50:
                trends["timeframes"]["medium_term"] = "haussière"
                if trends["primary"] == "neutre":
                    trends["primary"] = "haussière"
            elif sma20 < sma50:
                trends["timeframes"]["medium_term"] = "baissière"
                if trends["primary"] == "neutre":
                    trends["primary"] = "baissière"
        
                                               
        short_term_signals = []
        
        if pd.notna(macd) and pd.notna(macd_signal):
            if macd > macd_signal:
                short_term_signals.append("haussière")
            else:
                short_term_signals.append("baissière")
        
        if pd.notna(rsi):
            if rsi > 60:
                short_term_signals.append("haussière")
            elif rsi < 40:
                short_term_signals.append("baissière")
        
        if pd.notna(stoch_k) and pd.notna(stoch_d):
            if stoch_k > stoch_d and stoch_k < 80:
                short_term_signals.append("haussière")
            elif stoch_k < stoch_d and stoch_k > 20:
                short_term_signals.append("baissière")
        
                                                                    
        if short_term_signals:
            bullish_count = short_term_signals.count("haussière")
            bearish_count = short_term_signals.count("baissière")
            
            if bullish_count > bearish_count:
                trends["timeframes"]["short_term"] = "haussière"
            elif bearish_count > bullish_count:
                trends["timeframes"]["short_term"] = "baissière"
            else:
                trends["timeframes"]["short_term"] = "neutre"
        
                  
        if obv_slope > 0:
            trends["momentum"] = "positif"
        elif obv_slope < 0:
            trends["momentum"] = "négatif"
        else:
            trends["momentum"] = "neutre"
        
        return trends

    def _generate_recommendation(self, signals: List[Dict], trend_analysis: Dict, key_levels: Dict) -> Dict:
                   
        position = "NEUTRE"
        entry_price = key_levels.get("current_price", np.nan)
        stop_loss = None
        take_profit = None
        risk_reward = None
        confidence = "faible"
        trade_timeframe = None
        
        if pd.isna(entry_price):
            logger.error("Prix actuel non disponible pour générer une recommandation")
            return {"position": "ERREUR", "message": "Prix actuel non disponible"}

                                               
        bullish_score = 0
        bearish_score = 0
        strong_bullish = 0
        strong_bearish = 0
        
                                                   
        signal_map = {
                               
            "survendu extrême": (3, True),
            "survendu": (2, True), 
            "potentiellement survendu": (1, True),
            "bullish": (1, True), 
            "croisement bullish dynamique": (3, True), 
            "croisement bullish recent": (2.5, True),
            "croisement bullish": (1.5, True),
            "pression acheteuse": (1, True), 
            "forte pression acheteuse": (2, True),
            "déséquilibre acheteur extrême": (3, True),
            "cassure basse forte (survente)": (2.5, True),
            "cassure basse (survente)": (2, True),
            "divergence haussière": (2, True),
            "divergence OBV haussière": (2, True),
            "golden cross 20/50": (2.5, True),
            "accumulation forte": (1.5, True),
            
                               
            "suracheté extrême": (3, False),
            "suracheté": (2, False), 
            "potentiellement suracheté": (1, False),
            "bearish": (1, False), 
            "croisement bearish dynamique": (3, False),
            "croisement bearish recent": (2.5, False),
            "croisement bearish": (1.5, False),
            "pression vendeuse": (1, False), 
            "forte pression vendeuse": (2, False),
            "déséquilibre vendeur extrême": (3, False),
            "cassure haute forte (surachat)": (2.5, False),
            "cassure haute (surachat)": (2, False),
            "divergence baissière": (2, False),
            "divergence OBV baissière": (2, False),
            "death cross 20/50": (2.5, False),
            "distribution forte": (1.5, False),
        }
        
        for s in signals:
            score, is_bullish = signal_map.get(s['signal'], (0, None))
            
                                                    
            strength_multiplier = {
                "très fort": 2.0,
                "fort": 1.5,
                "modéré": 1.0,
                "faible": 0.5
            }.get(s['strength'], 1.0)
            
            final_score = score * strength_multiplier
            
            if is_bullish is True:
                bullish_score += final_score
                if s['strength'] in ["fort", "très fort"]: 
                    strong_bullish += 1
            elif is_bullish is False:
                bearish_score += final_score
                if s['strength'] in ["fort", "très fort"]:
                    strong_bearish += 1
        
                                                     
        trend_factor = 1.0
        primary_trend = trend_analysis.get("primary", "neutre")
        
        if primary_trend in ["fortement haussière", "haussière"]:
            trend_factor = 1.2                               
        elif primary_trend in ["fortement baissière", "baissière"]:
            trend_factor = 0.8                                                                      
        
        adjusted_bullish = bullish_score * trend_factor
        adjusted_bearish = bearish_score * (2 - trend_factor)                  
        
                                            
        long_strong_condition = (
            adjusted_bullish > adjusted_bearish * 1.5 and
            strong_bullish >= 2 and
            primary_trend in ["fortement haussière", "haussière"]
        )
        
        long_moderate_condition = (
            adjusted_bullish > adjusted_bearish * 1.2 and
            strong_bullish >= 1 and
            primary_trend not in ["fortement baissière"]
        )
        
                                             
        short_strong_condition = (
            adjusted_bearish > adjusted_bullish * 1.5 and
            strong_bearish >= 2 and
            primary_trend in ["fortement baissière", "baissière"]
        )
        
        short_moderate_condition = (
            adjusted_bearish > adjusted_bullish * 1.2 and
            strong_bearish >= 1 and
            primary_trend not in ["fortement haussière"]
        )
        
                         
        recommendation = {
            "position": "NEUTRE",
            "entry_price": entry_price,
            "stop_loss": None,
            "take_profit": None,
            "risk_reward_ratio": None,
            "confidence": "faible",
            "trade_timeframe": None,
            "percentage_move": {}
        }
        
                                                          
        atr = key_levels.get("atr", entry_price * 0.02)                               
        
        if long_strong_condition or long_moderate_condition:
            position = "LONG"
            
                                                                                     
            potential_sl_levels = [
                key_levels.get("support_1", entry_price * 0.98),
                key_levels.get("support_2", entry_price * 0.95),
                entry_price - (atr * 2)                   
            ]
            
                                                                                                       
            viable_sl_levels = [level for level in potential_sl_levels if level < entry_price - (atr * 0.75)]
            if viable_sl_levels:
                stop_loss = max(viable_sl_levels)                                                             
            else:
                stop_loss = entry_price - (atr * 2)                       
            
                                                             
            risk = entry_price - stop_loss
            if long_strong_condition:
                take_profit = entry_price + (risk * 3)                                          
                confidence = "élevée"
                trade_timeframe = "moyen terme"
            else:
                take_profit = entry_price + (risk * 2)                                            
                confidence = "modérée"
                trade_timeframe = "court terme"
                
        elif short_strong_condition or short_moderate_condition:
            position = "SHORT"
            
                                                 
            potential_sl_levels = [
                key_levels.get("resistance_1", entry_price * 1.02),
                key_levels.get("resistance_2", entry_price * 1.05),
                entry_price + (atr * 2)
            ]
            
                                                                                                         
            viable_sl_levels = [level for level in potential_sl_levels if level > entry_price + (atr * 0.75)]
            if viable_sl_levels:
                stop_loss = min(viable_sl_levels)
            else:
                stop_loss = entry_price + (atr * 2)
            
                                                             
            risk = stop_loss - entry_price
            if short_strong_condition:
                take_profit = entry_price - (risk * 3)
                confidence = "élevée"
                trade_timeframe = "moyen terme"
            else:
                take_profit = entry_price - (risk * 2)
                confidence = "modérée"
                trade_timeframe = "court terme"
        
                                                         
        if position != "NEUTRE" and stop_loss is not None and take_profit is not None:
            risk = abs(entry_price - stop_loss)
            reward = abs(entry_price - take_profit)
            risk_reward = round(reward / risk, 2) if risk > 0 else None
            
                                                    
            if entry_price > 0:
                percentage_to_sl = round(((stop_loss / entry_price) - 1) * 100, 2)
                percentage_to_tp = round(((take_profit / entry_price) - 1) * 100, 2)
                
                recommendation["percentage_move"] = {
                    "to_stop_loss": percentage_to_sl,
                    "to_take_profit": percentage_to_tp
                }
        
                                         
        recommendation["position"] = position
        recommendation["entry_price"] = entry_price
        recommendation["stop_loss"] = stop_loss
        recommendation["take_profit"] = take_profit
        recommendation["risk_reward_ratio"] = risk_reward
        recommendation["confidence"] = confidence
        recommendation["trade_timeframe"] = trade_timeframe
        
        return recommendation

    def _identify_key_levels(self, klines_df: pd.DataFrame, indicators_df: pd.DataFrame) -> Dict:
                   
        if klines_df is None or klines_df.empty:
            logger.warning("Aucune donnée disponible pour identifier les niveaux clés")
            return {}
        
        current_price = klines_df['close'].iloc[-1]
        
                             
        atr = indicators_df['atr'].iloc[-1] if 'atr' in indicators_df.columns else current_price * 0.02
        
                                   
        high_prices = klines_df['high'].values
        low_prices = klines_df['low'].values
        close_prices = klines_df['close'].values
        
        key_levels = {
            "current_price": current_price,
            "atr": atr
        }
        
                                                                                  
        pivots_high = analysis_utils.find_pivot_points(high_prices, is_high=True)
        pivots_low = analysis_utils.find_pivot_points(low_prices, is_high=False)
        
                                                           
        supports = []
        resistances = []
        
        for pivot in pivots_low:
            supports.append(low_prices[pivot])
        
        for pivot in pivots_high:
            resistances.append(high_prices[pivot])
        
                                         
        supports = sorted(set(supports))
        resistances = sorted(set(resistances))
        
                                                                             
        supports = [s for s in supports if s < current_price]
        resistances = [r for r in resistances if r > current_price]
        
                                                      
        if 'bb_upper' in indicators_df.columns and 'bb_lower' in indicators_df.columns:
            bb_upper = indicators_df['bb_upper'].iloc[-1]
            bb_lower = indicators_df['bb_lower'].iloc[-1]
            resistances.append(bb_upper)
            supports.append(bb_lower)
        
                                                               
        for ma in ['sma_20', 'sma_50', 'sma_200', 'ema_21']:
            if ma in indicators_df.columns:
                ma_value = indicators_df[ma].iloc[-1]
                if ma_value < current_price:
                    supports.append(ma_value)
                else:
                    resistances.append(ma_value)
        
                                      
        supports = sorted(supports, reverse=True)
        resistances = sorted(resistances)
        
                                                                       
        if supports:
            key_levels["support_1"] = supports[0] if supports else current_price * 0.98
            key_levels["support_2"] = supports[1] if len(supports) > 1 else current_price * 0.95
            key_levels["support_3"] = supports[2] if len(supports) > 2 else current_price * 0.93
        
        if resistances:
            key_levels["resistance_1"] = resistances[0] if resistances else current_price * 1.02
            key_levels["resistance_2"] = resistances[1] if len(resistances) > 1 else current_price * 1.05
            key_levels["resistance_3"] = resistances[2] if len(resistances) > 2 else current_price * 1.07
        
                                                                        
        if len(klines_df) > 20:
            recent_high = klines_df['high'].iloc[-20:].max()
            recent_low = klines_df['low'].iloc[-20:].min()
            
            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            fib_retracements = {}
            
                                                                              
            if current_price > recent_low:
                price_range = recent_high - recent_low
                for fib in fib_levels:
                    level = recent_high - (price_range * fib)
                    fib_retracements[f"fib_{int(fib*1000)}"] = level
            
            key_levels["fib_retracements"] = fib_retracements
        
        return key_levels

    def analyze_market(self, symbol: str = "BTCUSDT", interval: str = "1h") -> Dict:
                   
        try:
            logger.info(f"Démarrage de l'analyse pour {symbol} ({interval})")
            
                                             
            klines_df, order_book, ticker_24h = self._get_market_data(symbol, interval)
            
            if klines_df is None or klines_df.empty:
                logger.error(f"Aucune donnée disponible pour {symbol}")
                return {"status": "error", "message": f"Aucune donnée disponible pour {symbol}"}
            
                                                 
            try:
                indicators_df = analysis_utils.calculate_all_indicators(klines_df)
            except Exception as e:
                logger.error(f"Erreur lors du calcul des indicateurs techniques: {str(e)}")
                indicators_df = pd.DataFrame()
            
                                                     
            bid_volume, ask_volume, order_book_ratio = self._calculate_order_book_pressure(order_book)
            
                                         
            key_levels = self._identify_key_levels(klines_df, indicators_df)
            
                                    
            trend_analysis = self._determine_trend(indicators_df)
            
                                 
            signals = self._generate_signals(indicators_df, order_book_ratio)
            
                                       
            recommendation = self._generate_recommendation(signals, trend_analysis, key_levels)
            
                                          
            current_price = klines_df['close'].iloc[-1] if not klines_df.empty else None
            volume_24h = float(ticker_24h.get('volume', 0)) if ticker_24h else 0
            price_change_24h = float(ticker_24h.get('priceChangePercent', 0)) if ticker_24h else 0
            
            market_data = {
                "symbol": symbol,
                "interval": interval,
                "current_price": current_price,
                "volume_24h": volume_24h,
                "price_change_24h": price_change_24h,
                "bid_volume": bid_volume,
                "ask_volume": ask_volume,
                "order_book_ratio": order_book_ratio
            }
            
            analysis_result = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "market_data": market_data,
                "trend": trend_analysis.get("primary", "neutre"),
                "trend_analysis": trend_analysis,
                "signals": signals,
                "key_levels": key_levels,
                "recommendation": recommendation,
                "technical_indicators": indicators_df.tail(1).to_dict('records')[0] if not indicators_df.empty else {}
            }
            
            logger.info(f"Analyse terminée pour {symbol} ({interval}). Recommandation: {recommendation['position']}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de marché: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
