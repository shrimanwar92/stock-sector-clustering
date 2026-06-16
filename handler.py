import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from nselib import capital_market

warnings.filterwarnings("ignore")

class AuditableMomentumPipeline:

    def __init__(self, symbols: list, market_cap_map: dict = None, symbol_to_sector_map: dict = None, sector_regime_map: dict = None, lookback_years: float = 2.0):
        """
        Args:
            symbols: Raw universe pool of equity tickers.
            market_cap_map: Categorization dict mapping symbol -> index bracket string.
            symbol_to_sector_map: Mapping dict linking symbol -> Sector Name string.
            sector_regime_map: Mapping dict linking Sector Name -> Macro Regime string.
            lookback_years: Keeping at 2.0+ years to guarantee 252 historical trading rows.
        """
        self.symbols = symbols
        self.market_cap_map = market_cap_map or {}
        self.symbol_to_sector_map = symbol_to_sector_map or {}
        self.sector_regime_map = sector_regime_map or {}
        self.lookback_years = lookback_years
        self.allowed_categories = ['MIDCAP', 'SMALLCAP_100']
        
        self.approved_regimes = [
            "🔥 ULTRA_MOMENTUM_LEADERS", 
            "🚀 ACTIVE_BREAKOUT_FIELDS", 
            "📈 STABLE_UPWARD_ACCUMULATION"
        ]

    def _normalize_string(self, val: str) -> str:
        """Converts text safely to strip out symbols, underscores, and spacing variance."""
        return "".join(str(val).replace("_", "").replace(" ", "").upper().split())

    def fetch_universe_data(self) -> pd.DataFrame:
        """Silver Layer: Filters universe by Market Cap BEFORE downloading with Local Disk Cache Layer."""
        import os
        import gzip

        today_str = datetime.date.today().strftime("%d-%m-%Y")
        cache_dir = f"reports/[{today_str}]"
        cache_file_path = os.path.join(cache_dir, ".micro_universe_cache.json.gz")

        if os.path.exists(cache_file_path):
            print(f"💾 [CACHE READ] Hydrating raw data from today's disk cache: '{cache_file_path}'")
            try:
                with gzip.open(cache_file_path, "rt", encoding="utf-8") as f:
                    cached_df = pd.read_json(f, orient="records")
                if not cached_df.empty:
                    cached_df.columns = [str(col).replace('ï»¿', '').strip() for col in cached_df.columns]
                    return cached_df
            except Exception as ce:
                print(f"[WARN] Cache read collision ({str(ce)}). Falling back to exchange engine...")

        if self.market_cap_map:
            filtered_symbols = [
                s for s in self.symbols 
                if str(self.market_cap_map.get(s, '')).upper() in self.allowed_categories
            ]
            print(f"[GATEKEEPER] Universe refined: {len(self.symbols)} -> {len(filtered_symbols)} active candidates.")
        else:
            filtered_symbols = self.symbols
            print("[GATEKEEPER] No market_cap_map provided. Processing full input list.")

        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=int(365 * self.lookback_years))
        formatted_start = start_date.strftime("%d-%m-%Y")
        formatted_end = end_date.strftime("%d-%m-%Y")

        clean_symbols = [str(sym).split('.')[0].strip().upper() for sym in filtered_symbols]
        fetch_targets = list(set(clean_symbols + ["NIFTY 500"]))
        all_data = []

        def fetch_single_symbol(symbol):
            try:
                df = capital_market.price_volume_and_deliverable_position_data(
                    symbol=symbol, from_date=formatted_start, to_date=formatted_end
                )
                if df is not None and not df.empty:
                    df = df.copy().reset_index(drop=True)
                    df.columns = [str(col).replace('ï»¿', '').strip() for col in df.columns]
                    if "Symbol" in df.columns:
                        df = df.drop(columns=["Symbol"])
                    df["Symbol"] = str(symbol).upper()
                    return df
            except Exception: pass
            return None

        print(f"[START] Requesting historical streams across {len(fetch_targets)} target tickers...")
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_symbol = {executor.submit(fetch_single_symbol, sym): sym for sym in fetch_targets}
            for future in as_completed(future_to_symbol):
                res = future.result()
                if res is not None: all_data.append(res)

        if all_data:
            final_df = pd.concat(all_data, ignore_index=True, sort=False)
            final_df.columns = [str(col).replace('ï»¿', '').strip() for col in final_df.columns]
            final_df = final_df.loc[:, ~final_df.columns.duplicated()].copy()
            
            try:
                os.makedirs(cache_dir, exist_ok=True)
                with gzip.open(cache_file_path, "wt", encoding="utf-8") as f:
                    final_df.to_json(f, orient="records", date_format="iso")
                print(f"💾 [CACHE WRITE] Successfully stored today's raw micro universe data.")
            except Exception: pass
            
            return final_df
        return pd.DataFrame()

    def engineer_gold_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Gold Layer: Strict mathematical feature extraction rejecting any look-ahead or data-fill biases."""
        if raw_df.empty: return pd.DataFrame()
        
        raw_df.columns = [str(col).replace('ï»¿', '').strip() for col in raw_df.columns]
        
        possible_closes = ['ClosePrice', 'Close', 'close', 'CLOSE', 'ClosePriceParticulars']
        close_col = next((c for c in possible_closes if c in raw_df.columns), None)
        if not close_col: return pd.DataFrame()
            
        raw_df["Close"] = pd.to_numeric(raw_df[close_col].astype(str).str.replace(",", ""), errors='coerce')
        
        possible_vols = ['TotalTradedQty', 'Volume', 'volume', 'VOLUME', 'TotalTradedQuantity']
        vol_col = next((v for v in possible_vols if v in raw_df.columns), "Volume")
        raw_df["Volume"] = pd.to_numeric(raw_df[vol_col].astype(str).str.replace(",", ""), errors='coerce') if vol_col in raw_df.columns else np.nan
            
        possible_deliv = ['DeliverableQty', 'Delivery', 'delivery', '%DlyQttoTradedQty', 'DeliverableQuantity']
        deliv_col = next((d for d in possible_deliv if d in raw_df.columns), "Delivery")
        raw_df["Delivery"] = pd.to_numeric(raw_df[deliv_col].astype(str).str.replace(",", ""), errors='coerce') if deliv_col in raw_df.columns else np.nan

        raw_df['Date'] = pd.to_datetime(raw_df['Date'], errors='coerce')
        raw_df["Symbol"] = raw_df["Symbol"].astype(str).str.upper().str.strip()
        raw_df["Sector"] = raw_df["Symbol"].map(self.symbol_to_sector_map).fillna("UNKNOWN")

        # 1. Benchmark Matrix Ingestion
        index_df = raw_df[raw_df["Symbol"] == "NIFTY 500"].sort_values(by="Date").copy()
        if not index_df.empty:
            index_df["Index_EMA_50"] = index_df["Close"].ewm(span=50, adjust=False).mean()
            index_df["Market_Regime_Risk_Off"] = (index_df["Close"] < index_df["Index_EMA_50"]).astype(int)
            index_df["Index_ROC_20"] = index_df["Close"].pct_change(periods=20) * 100
            index_df["Index_ROC_252"] = index_df["Close"].pct_change(periods=252) * 100
            
            index_roc_20_map = dict(zip(index_df["Date"], index_df["Index_ROC_20"]))
            index_roc_252_map = dict(zip(index_df["Date"], index_df["Index_ROC_252"]))
            regime_risk_map = dict(zip(index_df["Date"], index_df["Market_Regime_Risk_Off"]))
        else:
            index_roc_20_map, index_roc_252_map, regime_risk_map = {}, {}, {}

        # 2. Dynamic Sector Matrix Synthesis (FIX #5: Compounding Unweighted Returns)
        stock_pool_df = raw_df[raw_df["Symbol"] != "NIFTY 500"].sort_values(by="Date").copy()
        if stock_pool_df.empty: return pd.DataFrame()
            
        # Compute individual asset daily returns to break absolute price weight dominance
        stock_pool_df["Daily_Return"] = stock_pool_df.groupby("Symbol")["Close"].pct_change()
        
        # Aggregate equally weighted arithmetic average of daily returns per sector per date
        sector_daily_returns = stock_pool_df.groupby(["Sector", "Date"])["Daily_Return"].mean().reset_index()
        sector_daily_returns = sector_daily_returns.sort_values(by="Date")
        
        sector_trends = {}
        for sector_name, s_group in sector_daily_returns.groupby("Sector"):
            s_group = s_group.copy()
            # Construct synthetic cumulative value index baseline (starting at 100)
            s_group["Synthetic_Index"] = 100.0 * (1.0 + s_group["Daily_Return"].fillna(0.0)).cumprod()
            s_group["Sector_EMA_50"] = s_group["Synthetic_Index"].ewm(span=50, adjust=False).mean()
            s_group["Sector_Bullish"] = (s_group["Synthetic_Index"] > s_group["Sector_EMA_50"]).astype(int)
            sector_trends[sector_name] = dict(zip(s_group["Date"], s_group["Sector_Bullish"]))

        # 3. Micro Asset Engineering Execution Block
        processed_stocks = []
        for symbol, group in stock_pool_df.groupby("Symbol"):
            group = group.sort_values(by="Date").copy()
            
            # FIX #1: Strict History Requirement (Enforce standard trading year history baseline)
            if len(group) < 252:
                continue

            # Alpha Relative Strength Engine
            group["Feature_ROC_20"] = group["Close"].pct_change(periods=20) * 100
            # FIX #2: Removed .bfill() completely to eliminate forward look-ahead lookups
            group["Feature_ROC_252"] = group["Close"].pct_change(periods=252) * 100
            
            group["RS_Short"] = group["Feature_ROC_20"] - group["Date"].map(index_roc_20_map).fillna(0)
            group["RS_Long"] = group["Feature_ROC_252"] - group["Date"].map(index_roc_252_map).fillna(0)
            group["Feature_Relative_Strength"] = (group["RS_Short"] * 0.4) + (group["RS_Long"] * 0.6)
            
            # FIX #3: Strict Alpha Filter Guardrail (Asset must beat broad baseline indexes)
            group["is_tradable"] = (group["Feature_Relative_Strength"] > 0).astype(int)
            
            # Trend Alignment Matrix
            group["EMA_20"] = group["Close"].ewm(span=20, adjust=False).mean()
            group["EMA_50"] = group["Close"].ewm(span=50, adjust=False).mean()
            group["EMA_200"] = group["Close"].ewm(span=200, adjust=False).mean()
            group["Feature_Trend_Aligned"] = ((group["EMA_20"] > group["EMA_50"]) & (group["EMA_50"] > group["EMA_200"])).astype(int)
            group["Feature_EMA_Dist"] = ((group["Close"] - group["EMA_20"]) / (group["EMA_20"] + 1e-9)) * 100

            # TradingView-Compliant Wilder's RSI Implementation
            delta = group["Close"].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-9)
            group["Feature_RSI"] = 100 - (100 / (1 + rs))

            # Volume Ratio Tracking
            if not group["Volume"].isna().all():
                group["Vol_MA3"] = group["Volume"].rolling(window=3, min_periods=1).mean()
                group["Vol_MA20"] = group["Volume"].rolling(window=20, min_periods=1).mean()
                group["Feature_Volume_Ratio"] = group["Vol_MA3"] / (group["Vol_MA20"] + 1e-9)
            else:
                group["Feature_Volume_Ratio"] = np.nan

            # Delivery Absorption Processing
            if not group["Delivery"].isna().all() and not group["Volume"].isna().all():
                group["Raw_Delivery_Pct"] = group["Delivery"] / (group["Volume"] + 1e-9)
                group["Delivery_Pct_MA20"] = group["Raw_Delivery_Pct"].rolling(window=20, min_periods=1).mean()
                group["Feature_Delivery_Ratio"] = group["Raw_Delivery_Pct"] / (group["Delivery_Pct_MA20"] + 1e-9)
            else:
                group["Feature_Delivery_Ratio"] = np.nan

            # Responsive Wilder's ATR Calculation Framework
            high_col = next((h for h in ['HighPrice', 'High', 'HIGH'] if h in group.columns), None)
            low_col = next((l for l in ['LowPrice', 'Low', 'LOW'] if l in group.columns), None)
            if high_col and low_col:
                group["H"] = pd.to_numeric(group[high_col], errors='coerce')
                group["L"] = pd.to_numeric(group[low_col], errors='coerce')
                group["C_prev"] = group["Close"].shift(1)
                group["TR"] = np.maximum(group["H"] - group["L"], np.maximum(np.abs(group["H"] - group["C_prev"]), np.abs(group["L"] - group["C_prev"])))
                
                group["ATR_14"] = group["TR"].ewm(alpha=1/14, adjust=False).mean()
                group["ATR_50"] = group["TR"].ewm(alpha=1/50, adjust=False).mean()
                group["Feature_ATR_Ratio"] = group["ATR_14"] / (group["ATR_50"] + 1e-9)
            else:
                group["Feature_ATR_Ratio"] = np.nan

            # MACD Acceleration 
            ema12 = group["Close"].ewm(span=12, adjust=False).mean()
            ema26 = group["Close"].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            group["MACD_Hist"] = macd_line - signal_line
            group["Feature_MACD_Hist_Accel"] = group["MACD_Hist"].diff().fillna(0)

            # Mapping Environmental Layouts
            group["Market_Regime_Risk_Off"] = group["Date"].map(regime_risk_map).fillna(0)
            sector_map = sector_trends.get(group["Sector"].iloc[0], {})
            group["Feature_Sector_Aligned"] = group["Date"].map(sector_map).fillna(1)

            # Intraday Close Location
            if high_col and low_col:
                group["Feature_Close_Strength"] = (group["Close"] - group["L"]) / (group["H"] - group["L"] + 1e-9)
            else:
                group["Feature_Close_Strength"] = 0.5

            # FIX #4: Replace silent .fillna() fabrications with strict .dropna() execution clearing
            # Any rows without complete window calculations are entirely removed from historical output
            group = group.dropna(subset=[
                "Feature_ROC_252", 
                "Feature_RSI", 
                "Feature_Volume_Ratio", 
                "Feature_Delivery_Ratio", 
                "Feature_ATR_Ratio"
            ])
            
            if not group.empty:
                processed_stocks.append(group)

        return pd.concat(processed_stocks, ignore_index=True) if processed_stocks else pd.DataFrame()

    def _apply_hybrid_guardrails_with_audit(self, row) -> tuple:
        """Deterministic Execution Firewall enforcing strict parameter criteria rules."""
        
        if row.get("is_tradable", 0) == 0:
            return ("⏳ FILTERED_OUT", "Failed structural Relative Strength Alpha threshold (RS <= 0).", 0.0, 0.0)

        current_sector = self._normalize_string(row.get("Sector", "UNKNOWN"))
        normalized_regime_map = {self._normalize_string(k): v for k, v in self.sector_regime_map.items()}
        sector_label = normalized_regime_map.get(current_sector, "UNKNOWN")
        
        if sector_label not in self.approved_regimes:
            return ("⏳ SECTOR_VETO", f"Veto: Sector '{current_sector}' mapped to unapproved cluster ({sector_label}).", 0.0, 0.0)

        if row.get("Market_Regime_Risk_Off", 0) == 1:
            return ("❌ REGIME_VETO", "Halted: Market structure macro environment is Risk-Off.", 0.0, 0.0)

        rsi = row.get("Feature_RSI", 50.0)
        ema_dist = row.get("Feature_EMA_Dist", 0.0)
        vol_ratio = row.get("Feature_Volume_Ratio", 1.0)
        macd_accel = row.get("Feature_MACD_Hist_Accel", 0.0)
        trend_aligned = row.get("Feature_Trend_Aligned", 0)
        delivery_ratio = row.get("Feature_Delivery_Ratio", 1.0)
        close_strength = row.get("Feature_Close_Strength", 0.5)
        roc_20 = row.get("Feature_ROC_20", 0.0)
        atr_ratio = row.get("Feature_ATR_Ratio", 1.0)
        close_price = row.get("Close", 0.0)

        atr_value = row.get("ATR_14", close_price * 0.03)
        stop_loss = round(close_price - (2 * atr_value), 2)
        target_price = round(close_price + (4 * atr_value), 2)

        if rsi >= 78.0 or ema_dist > 12.0:
            return ("🛑 OVEREXTENDED", f"Buy Veto: Overextended RSI ({rsi:.1f}).", 0.0, 0.0)
            
        if rsi <= 25.0 or ema_dist < -12.0:
            return ("📉 DEEP FLUSH", "Avoid: Capitulation structural drop.", 0.0, 0.0)

        # FIX #6 & #7: Strict Multi-Timeframe Trend and Acceleration Requirements
        # Requires strictly trend_aligned == 1 (EMA 20 > EMA 50 > EMA 200) and positive MACD histogram acceleration (> 0)
        if vol_ratio >= 1.25 and macd_accel > 0.0 and ema_dist > -1.0 and roc_20 > 2.0:
            if trend_aligned == 1:  # Enforce strict matrix alignment
                if delivery_ratio >= 1.15 or close_strength >= 0.65:
                    return ("🚀 INSIDER BREAKOUT", "Confirmed Entry: Vol + Alpha + Delivery + Sector Confirmation.", stop_loss, target_price)
                return ("🚀 ACTIVE BREAKOUT", "Momentum Entry: Vol + Macro structure confirmed with Sector support.", stop_loss, target_price)
            return ("⏳ COUNTER_TREND_MOMENTUM", "Halted: Disaligned moving average structure.", 0.0, 0.0)

        if -4.5 <= ema_dist <= 4.5 and trend_aligned == 1 and atr_ratio < 0.98:
            if delivery_ratio >= 1.20:
                return ("🏢 INSTITUTIONAL LAUNCHPAD", "High Priority: Coiling base compression with institutional footprints.", stop_loss, target_price)
            return ("🏢 LAUNCHPAD", "Standard Compression base consolidation.", stop_loss, target_price)

        return ("⏳ UNCONFIRMED CHOP", "Neutral: No trade zone matching parameters.", 0.0, 0.0)

    def export_execution_signals(self, gold_df: pd.DataFrame, output_filename: str = "execution_signals_report.txt"):
        """Compiles final report, ignoring filtered-out tickers."""
        if gold_df.empty:
            print("[WARN] No processing data passed to signal exporter.")
            return pd.DataFrame()

        latest_snapshot = gold_df.groupby("Symbol").last().reset_index()
        latest_snapshot["Sector"] = latest_snapshot["Symbol"].map(self.symbol_to_sector_map).fillna("UNKNOWN")
        
        print(f"[ENGINE] Evaluating {len(latest_snapshot)} assets through Technical Execution Firewall...")
        audit_df = latest_snapshot.apply(self._apply_hybrid_guardrails_with_audit, axis=1, result_type='expand')
        
        latest_snapshot["Strategic_Label"] = audit_df[0]
        latest_snapshot["Decision_Reason"] = audit_df[1]
        latest_snapshot["Stop_Loss"] = audit_df[2]
        latest_snapshot["Profit_Target"] = audit_df[3]
        
        invalid_states = ["⏳ FILTERED_OUT", "❌ REGIME_VETO", "⏳ SECTOR_MISALIGNED", "⏳ SECTOR_VETO", "⏳ UNCONFIRMED CHOP"]
        final_report = latest_snapshot[~latest_snapshot["Strategic_Label"].isin(invalid_states)].copy()
        
        print(f"[EXPORT] Writing audit log for {len(final_report)} qualified candidates to {output_filename}.")
        
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("=" * 150 + "\n")
            f.write(" PRODUCTION QUANT DATA LOGS: FILTERED SWING EXECUTION FIREWALL\n")
            f.write(f" Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 150 + "\n\n")
            if not final_report.empty:
                f.write(final_report[["Symbol", "Strategic_Label", "Close", "Stop_Loss", "Profit_Target"]].to_string(index=False))
                f.write("\n\n---\n")
                for _, r in final_report.iterrows():
                    f.write(f"{r['Symbol']}: {r['Strategic_Label']} -> SL: ₹{r['Stop_Loss']} | PT: ₹{r['Profit_Target']} | Reason: {r['Decision_Reason']}\n")
            else:
                f.write("No qualified tickers passed technical firewall triggers in this runtime window.")
            
        return final_report