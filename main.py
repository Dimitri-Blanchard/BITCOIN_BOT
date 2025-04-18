import argparse
import os
import time
import json
from datetime import datetime
import colorama
from colorama import Fore, Back, Style

from api_client import BinanceAPIClient
from analyzer import MarketAnalyzer

                                                      
colorama.init(autoreset=True)

def print_header(text):
    print(f"\n{Back.BLUE}{Fore.WHITE}{Style.BRIGHT} {text} {Style.RESET_ALL}")

def print_subheader(text):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{text}{Style.RESET_ALL}")

def print_success(text):
    print(f"{Fore.GREEN}{text}{Style.RESET_ALL}")

def print_warning(text):
    print(f"{Fore.YELLOW}{text}{Style.RESET_ALL}")

def print_error(text):
    print(f"{Fore.RED}{Style.BRIGHT}{text}{Style.RESET_ALL}")

def print_info(text):
    print(f"{Fore.WHITE}{text}")

def print_price(price, currency="$"):
    print(f"{Fore.YELLOW}{Style.BRIGHT}{currency}{price:.2f}{Style.RESET_ALL}")

def get_trend_color(trend):
    if "haussière" in trend.lower():
        return Fore.GREEN
    elif "baissière" in trend.lower():
        return Fore.RED
    else:
        return Fore.YELLOW

def get_position_color(position):
    if position == "LONG":
        return Fore.GREEN
    elif position == "SHORT":
        return Fore.RED
    else:
        return Fore.YELLOW

def get_signal_color(strength):
    if strength == "très fort":
        return Fore.MAGENTA + Style.BRIGHT
    elif strength == "fort":
        return Fore.RED + Style.BRIGHT
    elif strength == "modéré":
        return Fore.YELLOW
    else:
        return Fore.WHITE

def run_analysis(analyzer: MarketAnalyzer, symbol: str, interval: str, output_dir: str):
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_header(f"ANALYSE DE MARCHÉ: {symbol} ({interval}) - {timestamp_str}")
    
    try:
        analysis = analyzer.analyze_market(symbol=symbol, interval=interval)

        if analysis.get("status") == "error":
            print_error(f"Erreur: {analysis.get('message')}")
            return False

        current_price = analysis["market_data"]["current_price"]
        position = analysis["recommendation"]["position"]
        confidence = analysis["recommendation"]["confidence"]
        trend = analysis["trend"]
        price_change_24h = analysis["market_data"]["price_change_24h"]

                           
        print_subheader(f"Prix Actuel ({symbol}):")
        print_price(current_price)
        
                                                                 
        price_change_color = Fore.GREEN if price_change_24h >= 0 else Fore.RED
        print(f"Variation 24h: {price_change_color}{price_change_24h:+.2f}%{Style.RESET_ALL}")

                               
        print(f"Tendance ({interval}): {get_trend_color(trend)}{trend}{Style.RESET_ALL}")

                
        volume_24h = analysis["market_data"]["volume_24h"]
        print(f"Volume 24h: {Fore.CYAN}{volume_24h:,.0f} USDT{Style.RESET_ALL}")

                                
        book_ratio = analysis["market_data"]["order_book_ratio"]
        book_ratio_color = Fore.GREEN if book_ratio > 1 else Fore.RED
        print(f"Ratio Achat/Vente: {book_ratio_color}{book_ratio:.2f}{Style.RESET_ALL}")

                      
        print_subheader("Signaux Clés:")
        if analysis["signals"]:
            for signal in analysis["signals"][:5]:
                signal_color = get_signal_color(signal['strength'])
                print(f"• {Fore.CYAN}{signal['indicator']}{Style.RESET_ALL}: {signal_color}{signal['signal']} ({signal['strength']}){Style.RESET_ALL}")
        else:
            print_warning("Aucun signal détecté.")

                      
        print_subheader("Niveaux Clés:")
        key_levels = analysis["key_levels"]
        if "resistance_levels" in key_levels and key_levels["resistance_levels"]:
            print(f"{Fore.RED}Résistances:{Style.RESET_ALL} " + ", ".join([f"${r:.2f}" for r in key_levels["resistance_levels"][:3]]))
        if "support_levels" in key_levels and key_levels["support_levels"]:
            print(f"{Fore.GREEN}Supports:{Style.RESET_ALL} " + ", ".join([f"${s:.2f}" for s in key_levels["support_levels"][:3]]))

                        
        print_subheader("Recommandation:")
        position_color = get_position_color(position)
        print(f"{position_color}{Style.BRIGHT}{position}{Style.RESET_ALL} (Confiance: {confidence})")
        
        if position != "NEUTRE":
            reco = analysis["recommendation"]
            entry = reco["entry_price"]
            sl = reco["stop_loss"]
            tp = reco["take_profit"]
            rr = reco["risk_reward_ratio"]
            perc_sl = reco.get("percentage_move", {}).get("to_stop_loss", "N/A")
            perc_tp = reco.get("percentage_move", {}).get("to_take_profit", "N/A")

            print(f"  {Fore.WHITE}Entrée ~ ${entry:.2f}")
            print(f"  {Fore.RED}Stop Loss: ${sl:.2f} ({perc_sl if perc_sl != 'N/A' else 'N/A'}%)")
            print(f"  {Fore.GREEN}Take Profit: ${tp:.2f} ({perc_tp if perc_tp != 'N/A' else 'N/A'}%)")
            print(f"  {Fore.YELLOW}Ratio R/R: {rr if rr is not None else 'N/A'}")

                                    
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"{symbol}_{interval}_{timestamp}.json")

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=4, default=str)
            print_success(f"\nAnalyse détaillée sauvegardée: {filename}")
        except TypeError as e:
            print_error(f"Erreur lors de la sérialisation JSON: {e}. Tentative de sauvegarde partielle.")
            try:
                if 'technical_indicators' in analysis: del analysis['technical_indicators']
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(analysis, f, indent=4, default=str)
                print_warning(f"Sauvegarde partielle réussie dans: {filename}")
            except Exception as e_inner:
                print_error(f"Échec de la sauvegarde partielle: {e_inner}")
        except Exception as e:
            print_error(f"Erreur lors de la sauvegarde du fichier JSON: {e}")

        return True

    except Exception as e:
        print_error(f"Erreur inattendue durant l'analyse: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_progress_bar(iteration, total, length=50, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{Fore.CYAN}Prochaine analyse: {Style.RESET_ALL}{Fore.WHITE}|{bar}| {percent}% ({iteration}s/{total}s)', end='\r')
    if iteration == total:
        print()

def countdown(seconds):
    for i in range(seconds, 0, -1):
        print_progress_bar(seconds - i + 1, seconds)
        time.sleep(1)
    print()

if __name__ == "__main__":
    print(f"\n{Back.CYAN}{Fore.BLACK}{Style.BRIGHT} ANALYSEUR DE MARCHÉ CRYPTO {Style.RESET_ALL}")
    
    parser = argparse.ArgumentParser(description="Analyseur de marché Crypto via Binance API")
    parser.add_argument("--interval", type=str, default="1h", help="Intervalle (ex: 1m, 5m, 1h, 4h, 1d)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Paire de trading (ex: BTCUSDT, ETHUSDT)")
    parser.add_argument("--output", type=str, default="analysis_results", help="Répertoire pour les résultats")
    parser.add_argument("--continuous", action="store_true", help="Exécuter en continu")
    parser.add_argument("--delay", type=int, default=3600, help="Délai entre analyses (s)")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        try:
            os.makedirs(args.output)
            print_success(f"Répertoire de sortie créé: {args.output}")
        except OSError as e:
            print_error(f"Erreur: Impossible de créer le répertoire '{args.output}': {e}")
            exit(1)

    api_client = BinanceAPIClient()
    market_analyzer = MarketAnalyzer(api_client)

    if args.continuous:
        print_info(f"Démarrage de l'analyse continue pour {Fore.YELLOW}{args.symbol}{Style.RESET_ALL} "
                   f"({Fore.CYAN}{args.interval}{Style.RESET_ALL}) "
                   f"avec un délai de {Fore.GREEN}{args.delay}s{Style.RESET_ALL}")
        print_warning("Appuyez sur Ctrl+C pour arrêter.")
        
        iterations = 0
        while True:
            if iterations > 0:
                print_header(f"ANALYSE #{iterations+1}")
            success = run_analysis(market_analyzer, args.symbol, args.interval, args.output)
            iterations += 1
            
            wait_time = args.delay
            if not success:
                print_error("Une erreur s'est produite. Tentative de relance dans 60 secondes.")
                wait_time = 60
                
            try:
                countdown(wait_time)
            except KeyboardInterrupt:
                print("\n")
                print_warning("Arrêt demandé par l'utilisateur.")
                break
    else:
        run_analysis(market_analyzer, args.symbol, args.interval, args.output)
        print_success("\nAnalyse terminée.")
