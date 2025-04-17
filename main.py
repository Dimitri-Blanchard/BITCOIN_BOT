import argparse
import os
import time
import json
from datetime import datetime

from api_client import BinanceAPIClient
from analyzer import MarketAnalyzer

def run_analysis(analyzer: MarketAnalyzer, symbol: str, interval: str, output_dir: str):
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Lancement de l'analyse pour {symbol} ({interval})...")
    try:
        analysis = analyzer.analyze_market(symbol=symbol, interval=interval)

        if analysis.get("status") == "error":
            print(f"Erreur: {analysis.get('message')}")
            return False

        current_price = analysis["market_data"]["current_price"]
        position = analysis["recommendation"]["position"]
        confidence = analysis["recommendation"]["confidence"]
        trend = analysis["trend"]

        print(f"Prix Actuel ({symbol}): ${current_price:.2f}")
        print(f"Tendance ({interval}): {trend}")

        print("\nSignaux Clés:")
        if analysis["signals"]:
            for signal in analysis["signals"][:5]:
                print(f"- {signal['indicator']}: {signal['signal']} ({signal['strength']})")
        else:
            print("- Aucun signal détecté.")

        print(f"\nRecommandation: {position} (Confiance: {confidence})")
        if position != "NEUTRE":
            reco = analysis["recommendation"]
            entry = reco["entry_price"]
            sl = reco["stop_loss"]
            tp = reco["take_profit"]
            rr = reco["risk_reward_ratio"]
            perc_sl = reco.get("percentage_move", {}).get("to_stop_loss", "N/A")
            perc_tp = reco.get("percentage_move", {}).get("to_take_profit", "N/A")


            print(f"  Entrée ~ ${entry:.2f}")
            print(f"  Stop Loss: ${sl:.2f} ({perc_sl if perc_sl != 'N/A' else 'N/A'}%)")
            print(f"  Take Profit: ${tp:.2f} ({perc_tp if perc_tp != 'N/A' else 'N/A'}%)")
            print(f"  Ratio Risque/Récompense: {rr if rr is not None else 'N/A'}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"{symbol}_{interval}_{timestamp}.json")

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=4, default=str)
            print(f"\nAnalyse détaillée sauvegardée dans: {filename}")
        except TypeError as e:
             print(f"\nErreur lors de la sérialisation JSON: {e}. Tentative de sauvegarde partielle.")
             try:
                 if 'technical_indicators' in analysis: del analysis['technical_indicators']
                 with open(filename, 'w', encoding='utf-8') as f:
                     json.dump(analysis, f, indent=4, default=str)
                 print(f"Sauvegarde partielle réussie dans: {filename}")
             except Exception as e_inner:
                 print(f"Échec de la sauvegarde partielle: {e_inner}")

        except Exception as e:
            print(f"\nErreur lors de la sauvegarde du fichier JSON: {e}")

        return True

    except Exception as e:
        print(f"Erreur inattendue durant l'analyse: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyseur de marché Bitcoin (BTCUSDT) via Binance API")
    parser.add_argument("--interval", type=str, default="1h", help="Intervalle de temps (ex: 1m, 5m, 1h, 4h, 1d)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Paire de trading (ex: BTCUSDT, ETHUSDT)")
    parser.add_argument("--output", type=str, default="analysis_results", help="Répertoire de sortie pour les résultats JSON")
    parser.add_argument("--continuous", action="store_true", help="Exécuter en continu avec des mises à jour périodiques")
    parser.add_argument("--delay", type=int, default=3600, help="Délai entre les mises à jour en secondes (défaut: 3600 = 1 heure)")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        try:
            os.makedirs(args.output)
            print(f"Répertoire de sortie créé: {args.output}")
        except OSError as e:
            print(f"Erreur: Impossible de créer le répertoire de sortie '{args.output}': {e}")
            exit(1)

    api_client = BinanceAPIClient()
    market_analyzer = MarketAnalyzer(api_client)

    if args.continuous:
        print(f"Démarrage de l'analyse continue pour {args.symbol} ({args.interval}) avec un délai de {args.delay} secondes.")
        print("Appuyez sur Ctrl+C pour arrêter.")
        while True:
            success = run_analysis(market_analyzer, args.symbol, args.interval, args.output)
            wait_time = args.delay
            if not success:
                print("Une erreur s'est produite. Tentative de relance dans 60 secondes.")
                wait_time = 60
            print(f"\nProchaine analyse dans {wait_time} secondes...")
            try:
                time.sleep(wait_time)
            except KeyboardInterrupt:
                print("\nArrêt demandé par l'utilisateur.")
                break
    else:
        run_analysis(market_analyzer, args.symbol, args.interval, args.output)
        print("\nAnalyse terminée.")