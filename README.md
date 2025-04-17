# Market Analyzer & Trading Signal Generator

![](https://raw.githubusercontent.com/Dimitri-Blanchard/BITCOIN_BOT/refs/heads/main/images/crypto.avif)

## Description

This project provides a Python-based tool for analyzing cryptocurrency market data from the Binance public API. It calculates various technical indicators, identifies market trends and key support/resistance levels, analyzes order book pressure, and generates potential trading signals (LONG/SHORT/NEUTRE) with suggested Stop Loss (SL) and Take Profit (TP) levels.

The analysis results are displayed in the console and saved as detailed JSON files. The tool can be run for a single analysis or in continuous mode for periodic updates.

## Features

* **Data Fetching:** Retrieves K-line (candlestick), order book depth, and 24-hour ticker statistics from Binance API.
* **Technical Indicators:** Calculates a suite of indicators:
    * Moving Averages (SMA, EMA)
    * Relative Strength Index (RSI)
    * Moving Average Convergence Divergence (MACD)
    * Bollinger Bands (BBands)
    * Stochastic Oscillator (%K, %D)
    * Average True Range (ATR)
    * On-Balance Volume (OBV)
    * Momentum (MOM)
* **Level Identification:** Automatically identifies potential support and resistance levels using swing points and recent price action.
* **Trend Analysis:** Determines the short-term and long-term trend based on moving average configurations.
* **Order Book Analysis:** Calculates buy/sell pressure based on the top levels of the order book.
* **Signal Generation:** Aggregates data from indicators and analyses to generate weighted trading signals.
* **Trading Recommendation:** Provides a final recommendation (LONG, SHORT, or NEUTRE) including:
    * Entry Price suggestion (current market price)
    * Stop Loss (SL) level calculation (based on ATR and key levels)
    * Take Profit (TP) level calculation (based on ATR and key levels)
    * Calculated Risk/Reward Ratio
    * Confidence level (faible, moyenne, élevée)
* **Output:**
    * Concise summary printed to the console.
    * Detailed analysis report saved in JSON format in a specified output directory.
* **Flexibility:**
    * Configurable trading pair (symbol, e.g., `BTCUSDT`, `ETHUSDT`).
    * Configurable time interval (e.g., `1m`, `15m`, `1h`, `4h`, `1d`).
    * Option for continuous execution with adjustable delay.

## Technical Stack

* Python 3.13.3
* Pandas
* NumPy
* Requests

## Installation

1.  **Clone the repository:**
    ```bash
    git clone github.com/Dimitri-Blanchard/BITCOIN_BOT.git
    cd BITCOIN_BOT
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The script is run from the command line using `main.py`.

### Single Analysis

To run a single analysis for the default pair (`BTCUSDT`) and interval (`1h`):

```bash
python main.py
```

### To specify a different symbol or interval:

```bash
python main.py --symbol ETHUSDT --interval 4h
```

### Continuous Analysis
To run the analysis continuously, updating every hour (3600 seconds):

```bash
python main.py --continuous
```

To run continuously with a different symbol, interval, and update delay (e.g., every 15 minutes = 900 seconds):

```bash
python main.py --symbol BTCUSDT --interval 15m --continuous --delay 900
```

Press `Ctrl+C` to stop continuous analysis.

## Command-line Arguments

* `--interval`: Candlestick interval (default: `1h`). Examples: `1m`, `5m`, `1h`, `4h`, `1d`.
* `--symbol`: Trading pair symbol (default: `BTCUSDT`). Examples: `ETHUSDT`, `BNBBTC`.
* `--output`: Directory to save JSON analysis results (default: `analysis_results`).
* `-continuous`: Flag to enable continuous execution mode.
* `--delay`: Delay in seconds between analyses in continuous mode (default: `3600`).

## Output

1.  **Console Output:** Provides a real-time summary including:

* Current Price
* Determined Trend
* Key Signals detected
* Final Recommendation (Position, Confidence, Entry, SL, TP, R:R)

2. **JSON File Output:** Saves a detailed JSON file for each analysis run in the specified output directory (default: `analysis_results`). The filename includes the symbol, interval, and timestamp (e.g., `BTCUSDT_1h_20250417_195300.json`). This file contains:

* Timestamp, symbol, timeframe
* Market data (current price, 24h stats, key levels, ATR)
* Values of all calculated technical indicators
* Order book analysis details
* Trend information
* List of all generated signals
* The full recommendation object

## **Disclaimer**

This tool is for educational and informational purposes only. Trading cryptocurrencies involves significant risk. The signals and recommendations generated by this script are based on technical analysis algorithms and do not constitute financial advice. Past performance is not indicative of future results. Use this tool at your own risk and always conduct your own research before making any trading decisions. The authors are not responsible for any financial losses incurred.

## **Contributing**
Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.

# **License**
This project is licensed under the MIT License - see the LICENSE file for details
