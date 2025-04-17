import requests
import pandas as pd
from datetime import datetime

class BinanceAPIClient:
    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def _make_request(self, endpoint, params=None):
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la requête API vers {url}: {e}")
            return None
        except Exception as e:
            print(f"Erreur inattendue lors de la requête API: {e}")
            return None

    def get_klines(self, symbol="BTCUSDT", interval="1h", limit=200):
        endpoint = "/api/v3/klines"
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        data = self._make_request(endpoint, params)

        if data is None:
            return None

        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        try:
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

            numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                               'quote_asset_volume', 'taker_buy_base_asset_volume',
                               'taker_buy_quote_asset_volume', 'number_of_trades']

            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])

            df = df.drop(columns=['ignore'])
            return df
        except Exception as e:
            print(f"Erreur lors de la conversion des données klines en DataFrame: {e}")
            return None


    def get_order_book(self, symbol="BTCUSDT", limit=100):
        endpoint = "/api/v3/depth"
        params = {'symbol': symbol, 'limit': limit}
        return self._make_request(endpoint, params)

    def get_ticker_24h(self, symbol="BTCUSDT"):
        endpoint = "/api/v3/ticker/24hr"
        params = {'symbol': symbol}
        return self._make_request(endpoint, params)