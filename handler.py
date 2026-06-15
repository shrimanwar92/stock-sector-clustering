import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from nselib import capital_market

warnings.filterwarnings("ignore")


class AuditableMomentumPipeline:

    def __init__(self, symbols: list, lookback_years: float = 1.2):
        self.symbols = symbols
        self.lookback_years = lookback_years
        self.feature_cols = [
            "Feature_ATR_Ratio",
            "Feature_Volume_Ratio",
            "Feature_ROC_20",
            "Feature_EMA_Dist",
            "Feature_MACD_Hist_Accel",
            "Feature_IBA_Score"
        ]

    def fetch_universe_data(self) -> pd.DataFrame:
        """
        Silver Layer: Ingests raw asset history for targeted symbols in parallel.
        Uses multi-threaded concurrency to speed up network requests by ~10x.
        """
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=int(365 * self.lookback_years))
        formatted_start = start_date.strftime("%d-%m-%Y")
        formatted_end = end_date.strftime("%d-%m-%Y")

        all_data = []
        total_symbols = len(self.symbols)
        print(f"[INGEST] Parallelizing micro technical chart downloads for {total_symbols} active tickers...")

        def fetch_single_symbol(symbol):
            """Target worker function to execute within individual thread pools."""
            try:
                df = capital_market.price_volume_and_deliverable_position_data(
                    symbol=symbol, from_date=formatted_start, to_date=formatted_end
                )
                if df is not None and not df.empty:
                    df.columns = [col.strip() for col in df.columns]
                    df["Symbol"] = symbol
                    return df
            except Exception:
                pass
            return None

        # Restrict concurrent workers to 12 to maximize performance without triggering DDOS/Rate-Limit walls
        max_workers = min(12, total_symbols)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map futures to track which thread represents which asset
            future_to_symbol = {executor.submit(fetch_single_symbol, sym): sym for sym in self.symbols}
            
            completed_count = 0
            for future in as_completed(future_to_symbol):
                completed_count += 1
                result_df = future.result()
                if result_df is not None:
                    all_data.append(result_df)
                
                # Dynamic terminal progress indicator
                if completed_count % 10 == 0 or completed_count == total_symbols:
                    print(f" -> Processed {completed_count}/{total_symbols} assets...")

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def engineer_gold_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Gold Layer: Validates schemas and constructs multi-variable momentum vectors."""
        if raw_df.empty: return pd.DataFrame()
        
        close_col = 'ClosePrice' if 'ClosePrice' in raw_df.columns else 'Close'
        high_col = 'HighPrice' if 'HighPrice' in raw_df.columns else 'High'
        low_col = 'LowPrice' if 'LowPrice' in raw_df.columns else 'Low'
        volume_col = 'TotalTradedQty' if 'TotalTradedQty' in raw_df.columns else 'TotalTradedQuantity'
        delivery_col = 'DeliverableQtyPct' if 'DeliverableQtyPct' in raw_df.columns else '%DlyQttoTradedQty'

        raw_df["Close"] = pd.to_numeric(raw_df[close_col].astype(str).str.replace(",", ""), errors='coerce')
        raw_df["High"] = pd.to_numeric(raw_df[high_col].astype(str).str.replace(",", ""), errors='coerce')
        raw_df["Low"] = pd.to_numeric(raw_df[low_col].astype(str).str.replace(",", ""), errors='coerce')
        raw_df["Volume"] = pd.to_numeric(raw_df[volume_col].astype(str).str.replace(",", ""), errors='coerce')
        raw_df["DeliveryPct"] = pd.to_numeric(raw_df[delivery_col].astype(str).str.replace(",", ""), errors='coerce')
        raw_df['Date'] = pd.to_datetime(raw_df['Date'], errors='coerce')

        processed_stocks = []

        for symbol, group in raw_df.groupby("Symbol"):
            group = group.sort_values(by="Date").copy()
            
            # Simple Block Trade proxy identification using volume deviations
            group["Is_Large_Block"] = (group["Volume"] > group["Volume"].rolling(20).mean() * 2.5).astype(int)
            group["Feature_IBA_Score"] = group["Is_Large_Block"].rolling(10, min_periods=1).sum()

            # Volatility and Volume Accelerations
            high_low = group["High"] - group["Low"]
            high_close = np.abs(group["High"] - group["Close"].shift())
            low_close = np.abs(group["Low"] - group["Close"].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            group["Feature_ATR_Ratio"] = tr.rolling(10).mean() / (tr.rolling(50).mean() + 1e-9)
            group["Feature_Volume_Ratio"] = group["Volume"].rolling(3).mean() / (group["Volume"].rolling(20).mean() + 1e-9)
            group["Feature_ROC_20"] = group["Close"].pct_change(periods=20) * 100
            
            ema_20 = group["Close"].ewm(span=20, adjust=False).mean()
            group["Feature_EMA_Dist"] = ((group["Close"] - ema_20) / (ema_20 + 1e-9)) * 100

            # MACD Momentum Velocity Tracking
            ema_12 = group["Close"].ewm(span=12, adjust=False).mean()
            ema_26 = group["Close"].ewm(span=26, adjust=False).mean()
            macd_hist = (ema_12 - ema_26) - (ema_12 - ema_26).ewm(span=9, adjust=False).mean()
            group["Feature_MACD_Hist_Accel"] = macd_hist.diff(periods=3)

            # Classic Relative Strength Index Component
            delta = group["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            group["Feature_RSI"] = 100 - (100 / (1 + rs))

            group = group.dropna(subset=self.feature_cols + ["Feature_RSI"])
            processed_stocks.append(group)

        return pd.concat(processed_stocks, ignore_index=True) if processed_stocks else pd.DataFrame()

    def _apply_hybrid_guardrails_with_audit(self, row) -> pd.Series:
        """Deterministic Execution Firewall Engine with Natural Language Explanation String Generators."""
        rsi = row["Feature_RSI"]
        ema_dist = row["Feature_EMA_Dist"]
        vol_ratio = row["Feature_Volume_Ratio"]
        macd_accel = row["Feature_MACD_Hist_Accel"]
        iba_score = row["Feature_IBA_Score"]
        
        # Rule 1: Absolute Overextended Exhaustion Top
        if rsi >= 75.0 or ema_dist > 8.0:
            return pd.Series(["🛑 OVEREXTENDED", f"Buy Veto triggered. RSI ({rsi:.1f} >= 75) or price distance to 20EMA ({ema_dist:.1f}%) indicates an overextended exhaustion zone."])
            
        # Rule 2: Deep capitulation/flush vertical dropdown 
        if rsi <= 25.0 or ema_dist < -10.0:
            return pd.Series(["📉 DEEP FLUSH", f"Avoid Asset. RSI ({rsi:.1f} <= 25) or price extension below 20EMA ({ema_dist:.1f}%) signals steep capitulation trend."])

        # Rule 3: High-Probability Breakout Validation
        if vol_ratio >= 1.25 and macd_accel > 0 and ema_dist > 0:
            if iba_score >= 1:
                return pd.Series(["🚀 INSIDER BREAKOUT", f"Confirmed entry signal. Volume Expansion ({vol_ratio:.2f}x) backed by positive MACD velocity and {int(iba_score)} institutional block trades."])
            return pd.Series(["🚀 ACTIVE BREAKOUT", f"Momentum entry confirmation. Volume Expansion ({vol_ratio:.2f}x) and positive MACD directional velocity. No major blocks detected."])

        # Rule 4: Quiet Consolidation/Launchpad Entry Fields
        if -3.5 <= ema_dist <= 3.5:
            if iba_score >= 2:
                return pd.Series(["🏢 INSTITUTIONAL LAUNCHPAD", f"High Priority Accumulation. Price coiling tight to core average ({ema_dist:.1f}%) with significant institutional block absorption ({int(iba_score)} hits)."])
            return pd.Series(["🏢 LAUNCHPAD", f"Standard Compression Watch. Asset consolidating cleanly within historical moving average structural boundaries ({ema_dist:.1f}%)."])

        return pd.Series(["⏳ UNCONFIRMED CHOP", f"Neutral/No Trade Zone. Indicators did not meet precise strategy parameters (RSI: {rsi:.1f}, EMA_Dist: {ema_dist:.1f}%, Vol_Ratio: {vol_ratio:.2f}x)."])

    def export_execution_signals(self, gold_df: pd.DataFrame, output_filename: str = "execution_signals_report.txt"):
        """Compiles technical metadata arrays and writes clean audit logs directly to a text file."""
        if gold_df.empty: return

        latest_snapshot = gold_df.groupby("Symbol").last().reset_index()
        latest_snapshot[["Strategic_Label", "Decision_Reason"]] = latest_snapshot.apply(
            self._apply_hybrid_guardrails_with_audit, axis=1
        )
        
        # Sorting outputs to surface high-priority trade setups right at the top of the file
        priority_map = {"🚀 INSIDER BREAKOUT": 0, "🚀 ACTIVE BREAKOUT": 1, "🏢 INSTITUTIONAL LAUNCHPAD": 2, "🏢 LAUNCHPAD": 3, "⏳ UNCONFIRMED CHOP": 4, "🛑 OVEREXTENDED": 5, "📉 DEEP FLUSH": 6}
        latest_snapshot["Priority"] = latest_snapshot["Strategic_Label"].map(priority_map).fillna(9)
        latest_snapshot = latest_snapshot.sort_values(by=["Priority", "Feature_ROC_20"], ascending=[True, False])

        print(f"[EXPORT] Writing down technical execution audits to text file: '{output_filename}'...")
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("=" * 140 + "\n")
            f.write(f" PRODUCTION QUANT DATA LOGS: MICRO TECHNICAL EXECUTION INTERACTION FIREWALL\n")
            f.write(f" Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Scored Workload Universe Size: {len(latest_snapshot)} Tickers\n")
            f.write("=" * 140 + "\n\n")

            f.write("## SECTION 1: EXECUTION TRADING LEDGER\n")
            f.write("-" * 140 + "\n")
            f.write(latest_snapshot[["Symbol", "Strategic_Label", "Close", "Feature_RSI", "Feature_EMA_Dist"]].to_string(index=False))
            f.write("\n\n" + "=" * 140 + "\n\n")

            f.write("## SECTION 2: SYSTEM RAW FEATURE METADATA MATRICES\n")
            f.write("-" * 140 + "\n")
            f.write(f"{'Symbol':<12} | {'Close':<10} | {'RSI (14)':<8} | {'EMA Dist %':<10} | {'Vol Ratio':<10} | {'MACD Accel':<12} | {'Block Score':<12}\n")
            f.write("-" * 140 + "\n")
            for _, r in latest_snapshot.iterrows():
                f.write(f"{r['Symbol']:<12} | {r['Close']:<10.2f} | {r['Feature_RSI']:<8.1f} | {r['Feature_EMA_Dist']:<10.2f} | {r['Feature_Volume_Ratio']:<10.2f} | {r['Feature_MACD_Hist_Accel']:<12.4f} | {r['Feature_IBA_Score']:<12.0f}\n")
            f.write("\n\n" + "=" * 140 + "\n\n")

            f.write("## SECTION 3: EXPANDED MICRO SYSTEM DECISION REASONS\n")
            f.write("-" * 140 + "\n")
            for _, r in latest_snapshot.iterrows():
                f.write(f"🎯 TICKER  : {r['Symbol']}\n")
                f.write(f"   SIGNAL  : {r['Strategic_Label']}\n")
                f.write(f"   FIREWALL: {r['Decision_Reason']}\n")
                f.write("-" * 80 + "\n")

        return latest_snapshot