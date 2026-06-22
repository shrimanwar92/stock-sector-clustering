import os
import warnings
import numpy as np
import pandas as pd
import gzip
from xgboost import XGBClassifier
from constants import (
    MODEL_PATH, 
    FEATURE_COLUMNS,
    REPORTS_DIR,
    fetch_data_from_nse
)

warnings.filterwarnings("ignore")


class StocksRuleEngine:

    def __init__(
        self, 
        symbols: list, 
        market_cap_map: dict = None, 
        symbol_to_sector_map: dict = None, 
        sector_regime_map: dict = None, 
        sector_score_map: dict = None,  
        lookback_years: float = 2.0,
        macro_score_threshold: float = 0.55
    ):
        self.symbols = symbols
        self.market_cap_map = market_cap_map or {}
        self.symbol_to_sector_map = symbol_to_sector_map or {}
        self.sector_regime_map = sector_regime_map or {}
        self.sector_score_map = sector_score_map or {}  
        self.lookback_years = lookback_years
        self.macro_score_threshold = macro_score_threshold  
        self.allowed_categories = ['MIDCAP', 'SMALLCAP_100']
        self.cache_file_path = os.path.join(REPORTS_DIR, ".micro_universe_cache.json.gz")
    
    def load_stocks_from_cache(self):
        cached_df = pd.DataFrame()
        if os.path.exists(self.cache_file_path):
            print(f"💾 [CACHE READ] Hydrating raw data from today's disk cache: '{self.cache_file_path}'")
            try:
                with gzip.open(self.cache_file_path, "rt", encoding="utf-8") as f:
                    cached_df = pd.read_json(f, orient="records")
                if not cached_df.empty:
                    cached_df.columns = [str(col).replace('ï»¿', '').strip() for col in cached_df.columns]
            except Exception as ce:
                print(f"[WARN] Cache read collision ({str(ce)}). Falling back to exchange engine...")
        return cached_df
    
    def save_stocks_to_cache(self, df):
        with gzip.open(self.cache_file_path, "wt", encoding="utf-8") as f:
            df.to_json(f, orient="records", date_format="iso")
        print(f"💾 [CACHE WRITE] Successfully stored today's raw micro universe data.")

    def fetch_universe_data(self) -> pd.DataFrame:
        filtered_symbols = self.symbols
        nse_df = self.load_stocks_from_cache()
        if nse_df.empty:
            filtered_symbols = list(set(filtered_symbols + ["NIFTY 500"]))
            nse_df = fetch_data_from_nse(filtered_symbols, self.symbol_to_sector_map)
        return nse_df

    def _parse_and_sanitize_columns(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()
        df.columns = [str(col).replace('ï»¿', '').strip() for col in df.columns]
        
        possible_closes = ['ClosePrice', 'Close', 'close', 'CLOSE', 'ClosePriceParticulars']
        close_col = next((c for c in possible_closes if c in df.columns), None)
        if not close_col: return pd.DataFrame()
            
        df["Close"] = pd.to_numeric(df[close_col].astype(str).str.replace(",", ""), errors='coerce')
        
        possible_highs = ['HighPrice', 'High', 'HIGH', 'HighPriceParticulars']
        high_col = next((h for h in possible_highs if h in df.columns), None)
        if high_col:
            df["High"] = pd.to_numeric(df[high_col].astype(str).str.replace(",", ""), errors='coerce')
            
        possible_lows = ['LowPrice', 'Low', 'LOW', 'LowPriceParticulars']
        low_col = next((l for l in possible_lows if l in df.columns), None)
        if low_col:
            df["Low"] = pd.to_numeric(df[low_col].astype(str).str.replace(",", ""), errors='coerce')

        possible_vols = ['TotalTradedQty', 'Volume', 'volume', 'VOLUME', 'TotalTradedQuantity']
        vol_col = next((v for v in possible_vols if v in df.columns), "Volume")
        df["Volume"] = pd.to_numeric(df[vol_col].astype(str).str.replace(",", ""), errors='coerce') if vol_col in df.columns else np.nan
            
        possible_deliv = ['DeliverableQty', 'Delivery', 'delivery', '%DlyQttoTradedQty', 'DeliverableQuantity']
        deliv_col = next((d for d in possible_deliv if d in df.columns), "Delivery")
        df["Delivery"] = pd.to_numeric(df[deliv_col].astype(str).str.replace(",", ""), errors='coerce') if deliv_col in df.columns else np.nan

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
        df["Sector"] = df["Symbol"].map(self.symbol_to_sector_map).fillna("UNKNOWN")
        return df

    def _generate_benchmark_regime_maps(self, df: pd.DataFrame) -> tuple:
        index_df = df[df["Symbol"] == "NIFTY 500"].sort_values(by="Date").copy()
        if index_df.empty: return {}, {}, {}, {}
            
        index_df["Index_EMA_200"] = index_df["Close"].ewm(span=200, adjust=False).mean()
        index_df["Market_Regime_Risk_Off"] = (index_df["Close"] < index_df["Index_EMA_200"]).astype(int)
        
        index_df["Index_ROC_20"] = index_df["Close"].pct_change(periods=20) * 100
        index_df["Index_ROC_252"] = index_df["Close"].pct_change(periods=252) * 100
        
        index_df["Dynamic_Alpha_Cutoff"] = index_df["Index_ROC_20"].apply(
            lambda x: -6.0 if x > 5.0 else (-3.0 if x > 2.0 else 0.0)
        )
        
        return (dict(zip(index_df["Date"], index_df["Index_ROC_20"])),
                dict(zip(index_df["Date"], index_df["Index_ROC_252"])),
                dict(zip(index_df["Date"], index_df["Market_Regime_Risk_Off"])),
                dict(zip(index_df["Date"], index_df["Dynamic_Alpha_Cutoff"])))

    def _generate_sector_trend_maps(self, stock_pool_df: pd.DataFrame) -> dict:
        stock_pool = stock_pool_df.copy()
        stock_pool["Daily_Return"] = stock_pool.groupby("Symbol")["Close"].pct_change()
        
        sector_daily_returns = stock_pool.groupby(["Sector", "Date"])["Daily_Return"].mean().reset_index()
        sector_daily_returns = sector_daily_returns.sort_values(by="Date")
        
        sector_trends = {}
        for sector_name, s_group in sector_daily_returns.groupby("Sector"):
            s_group = s_group.copy()
            s_group["Synthetic_Index"] = 100.0 * (1.0 + s_group["Daily_Return"].fillna(0.0)).cumprod()
            s_group["Sector_EMA_50"] = s_group["Synthetic_Index"].ewm(span=50, adjust=False).mean()
            s_group["Sector_Bullish"] = (s_group["Synthetic_Index"] > s_group["Sector_EMA_50"]).astype(int)
            sector_trends[sector_name] = dict(zip(s_group["Date"], s_group["Sector_Bullish"]))
        return sector_trends

    def _engineer_single_asset_features(self, group: pd.DataFrame, index_roc_20_map: dict, index_roc_252_map: dict, regime_risk_map: dict, sector_trends: dict, dynamic_alpha_map: dict) -> pd.DataFrame:
        group = group.sort_values(by="Date").copy()
        if len(group) < 252: return pd.DataFrame()

        group["Feature_ROC_20"] = group["Close"].pct_change(periods=20) * 100
        group["Feature_ROC_252"] = group["Close"].pct_change(periods=252) * 100
        group["RS_Short"] = group["Feature_ROC_20"] - group["Date"].map(index_roc_20_map).fillna(0)
        group["RS_Long"] = group["Feature_ROC_252"] - group["Date"].map(index_roc_252_map).fillna(0)
        group["Feature_Relative_Strength"] = (group["RS_Short"] * 0.4) + (group["RS_Long"] * 0.6)
        
        group["Assigned_Alpha_Floor"] = group["Date"].map(dynamic_alpha_map).fillna(0.0)
        group["is_tradable"] = (group["Feature_Relative_Strength"] > group["Assigned_Alpha_Floor"]).astype(int)
        
        group["EMA_20"] = group["Close"].ewm(span=20, adjust=False).mean()
        group["EMA_50"] = group["Close"].ewm(span=50, adjust=False).mean()
        group["EMA_200"] = group["Close"].ewm(span=200, adjust=False).mean()
        group["Feature_Trend_Aligned"] = ((group["EMA_20"] > group["EMA_50"]) & (group["EMA_50"] > group["EMA_200"])).astype(int)
        group["Feature_EMA_Dist"] = ((group["Close"] - group["EMA_20"]) / (group["EMA_20"] + 1e-9)) * 100

        delta = group["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        group["Feature_RSI"] = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-9)) + 1e-9))

        if not group["Volume"].isna().all():
            group["Vol_MA3"] = group["Volume"].rolling(window=3, min_periods=1).mean()
            group["Vol_MA20"] = group["Volume"].rolling(window=20, min_periods=1).mean()
            group["Feature_Volume_Ratio"] = group["Vol_MA3"] / (group["Vol_MA20"] + 1e-9)
        else:
            group["Feature_Volume_Ratio"] = np.nan

        if not group["Delivery"].isna().all() and not group["Volume"].isna().all():
            group["Raw_Delivery_Pct"] = group["Delivery"] / (group["Volume"] + 1e-9)
            group["Delivery_Pct_MA20"] = group["Raw_Delivery_Pct"].rolling(window=20, min_periods=1).mean()
            group["Feature_Delivery_Ratio"] = group["Raw_Delivery_Pct"] / (group["Delivery_Pct_MA20"] + 1e-9)
        else:
            group["Feature_Delivery_Ratio"] = np.nan

        if "High" in group.columns and "Low" in group.columns:
            group["H"] = group["High"]
            group["L"] = group["Low"]
            group["C_prev"] = group["Close"].shift(1)
            group["TR"] = np.maximum(group["H"] - group["L"], np.maximum(np.abs(group["H"] - group["C_prev"]), np.abs(group["L"] - group["C_prev"])))
            
            group["ATR_14"] = group["TR"].ewm(alpha=1/14, adjust=False).mean()
            group["ATR_50"] = group["TR"].ewm(alpha=1/50, adjust=False).mean()
            group["Feature_ATR_Ratio"] = group["ATR_14"] / (group["ATR_50"] + 1e-9)
            group["Feature_Close_Strength"] = (group["Close"] - group["L"]) / (group["H"] - group["L"] + 1e-9)
            
            up_move = group["H"] - group["H"].shift(1)
            down_move = group["L"].shift(1) - group["L"]
            p_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            m_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
            
            smooth_tr = group["TR"].ewm(alpha=1/14, adjust=False).mean()
            smooth_p_dm = pd.Series(p_dm, index=group.index).ewm(alpha=1/14, adjust=False).mean()
            smooth_m_dm = pd.Series(m_dm, index=group.index).ewm(alpha=1/14, adjust=False).mean()
            
            p_di = 100.0 * (smooth_p_dm / (smooth_tr + 1e-9))
            m_di = 100.0 * (smooth_m_dm / (smooth_tr + 1e-9))
            dx = 100.0 * (np.abs(p_di - m_di) / (p_di + m_di + 1e-9))
            group["Feature_ADX_14"] = dx.ewm(alpha=1/14, adjust=False).mean()
            group["Pivot_Low_30"] = group["L"].rolling(window=30, min_periods=10).min()
        else:
            group["Feature_ATR_Ratio"] = np.nan
            group["Feature_Close_Strength"] = 0.5
            group["Feature_ADX_14"] = 0.0
            group["Pivot_Low_30"] = group["Close"] * 0.95

        ema12 = group["Close"].ewm(span=12, adjust=False).mean()
        ema26 = group["Close"].ewm(span=26, adjust=False).mean()
        group["MACD_Hist"] = ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean()
        group["Feature_MACD_Hist_Accel"] = group["MACD_Hist"].rolling(window=3, min_periods=1).mean().diff().fillna(0)

        group["Market_Regime_Risk_Off"] = group["Date"].map(regime_risk_map).fillna(0)
        group["Feature_Sector_Aligned"] = group["Date"].map(sector_trends.get(group["Sector"].iloc[0], {})).fillna(0)

        return group.dropna(subset=["Feature_ROC_252", "Feature_RSI", "Feature_Volume_Ratio", "Feature_Delivery_Ratio", "Feature_ATR_Ratio"])

    def engineer_gold_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = self._parse_and_sanitize_columns(raw_df)
        if df.empty: return pd.DataFrame()
            
        index_roc_20, index_roc_252, regime_risk, dynamic_alpha = self._generate_benchmark_regime_maps(df)
        stock_pool_df = df[df["Symbol"] != "NIFTY 500"].sort_values(by="Date").copy()
        if stock_pool_df.empty: return pd.DataFrame()
            
        sector_trends = self._generate_sector_trend_maps(stock_pool_df)
        
        processed_stocks = []
        for _, group in stock_pool_df.groupby("Symbol"):
            feat_df = self._engineer_single_asset_features(group, index_roc_20, index_roc_252, regime_risk, sector_trends, dynamic_alpha)
            if not feat_df.empty: processed_stocks.append(feat_df)
                
        return pd.concat(processed_stocks, ignore_index=True) if processed_stocks else pd.DataFrame()

    # -------------------------------------------------------------------------
    # UPGRADE: Comprehensive López de Prado Path Engine (Typo Fixed + Timeout Return Rules)
    # -------------------------------------------------------------------------
    def _apply_lopez_de_prado_barriers(self, group: pd.DataFrame, pt_horizon: int, base_pt_mult: float, base_sl_mult: float) -> pd.Series:
        """
        Executes dynamic path analysis. Incorporates the Upgrade for Sector-Regime 
        Adaptive Multipliers and explicit Timeout Return Classification.
        """
        labels = np.zeros(len(group), dtype=int)
        close_prices = group['Close'].values
        high_prices = group['High'].values if 'High' in group.columns else close_prices
        low_prices = group['Low'].values if 'Low' in group.columns else close_prices
        atr_values = group['ATR_14'].values if 'ATR_14' in group.columns else (close_prices * 0.02)
        
        # Pull Sector alignment arrays to perform structural barrier modifications
        sector_aligned = group['Feature_Sector_Aligned'].values if 'Feature_Sector_Aligned' in group.columns else np.ones(len(group))
        total_bars = len(group)
        
        for i in range(total_bars):
            current_close = close_prices[i]
            current_atr = atr_values[i] if not np.isnan(atr_values[i]) else (current_close * 0.02)
            
            # Upgrade: Sector Regime-Based Sizing (Wider targets during active sector trends)
            if sector_aligned[i] == 1:
                pt_mult = base_pt_mult * 1.2   # e.g., 2.5 * 1.2 = 3.0 ATR
                sl_mult = base_sl_mult         # e.g., 1.5 ATR
            else:
                pt_mult = base_pt_mult * 0.8   # e.g., 2.5 * 0.8 = 2.0 ATR
                sl_mult = base_sl_mult * 0.66  # e.g., 1.5 * 0.66 = 1.0 ATR
                
            pt_barrier = current_close + (pt_mult * current_atr)
            sl_barrier = current_close - (sl_mult * current_atr)
            
            end_idx = min(i + pt_horizon + 1, total_bars)
            barrier_triggered = False
            
            for j in range(i + 1, end_idx):
                current_high = high_prices[j]
                current_low = low_prices[j]
                
                hit_pt = current_high >= pt_barrier
                hit_sl = current_low <= sl_barrier
                
                if hit_pt and hit_sl:
                    labels[i] = 0  
                    barrier_triggered = True
                    break
                elif hit_pt:
                    labels[i] = 2  # Class 2: Upside Expansion
                    barrier_triggered = True
                    break
                elif hit_sl:
                    labels[i] = 0  # Class 0: Downside Risk
                    barrier_triggered = True
                    break
            
            # Upgrade: Granular Timeout Threshold Logic (Handles steady, crawling trends)
            if not barrier_triggered:
                terminal_idx = end_idx - 1
                final_return = (close_prices[terminal_idx] - current_close) / (current_close + 1e-9)
                atr_pct_hurdle = (current_atr / (current_close + 1e-9)) * 0.5  # Dynamic 0.5x ATR equivalent hurdle
                
                if final_return > atr_pct_hurdle:
                    labels[i] = 2  # Gradual uptrend is counted as a Success
                elif final_return < -atr_pct_hurdle:
                    labels[i] = 0  # Gradual degradation is counted as a Failure
                else:
                    labels[i] = 1  # Standard Stagnation/Timeout Lockup
                
        return pd.Series(labels, index=group.index)

    def engineer_training_labels(
        self, 
        df: pd.DataFrame, 
        pt_horizon: int = 20, 
        pt_mult: float = 2.5, 
        sl_mult: float = 1.5
    ) -> pd.DataFrame:
        """
        Public matrix labeling gateway.
        CRITICAL FIX: Retains all risk-off crash periods to eliminate optimization bias.
        """
        print(f"⚙️ [LABEL ENGINE] Executing 3-Class Self-Adaptive System Matrix Logic...")
        
        if 'Strategic_Label' in df.columns:
            df = df.drop(columns=['Strategic_Label'])
            
        df = df.sort_values(by=['Symbol', 'Date']).copy()
        
        df['Strategic_Label'] = df.groupby('Symbol', group_keys=False).apply(
            lambda x: self._apply_lopez_de_prado_barriers(x, pt_horizon, pt_mult, sl_mult)
        )
        
        # Upgrade: Imbalance Diagnostic Engine Logs
        print("\n📊 [DIAGNOSTIC] Class Distribution Breakdown across Entire Matrix:")
        distribution = df['Strategic_Label'].value_counts(normalize=True).sort_index()
        for cls, pct in distribution.items():
            name = "0 (Downside/Failure)" if cls == 0 else ("1 (True Stagnation)" if cls == 1 else "2 (Upside/Success)")
            print(f" -> Class {name}: {pct*100:.2f}%")
            
        return df

    def execute_ml_signals(self, gold_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculates probabilities, handles asymmetry parameters, and ranks positions 
        by mathematical Expected Value (EV) rather than raw win-rate probability.
        """
        if gold_df is None or gold_df.empty:
            print("[WARN] Empty data frame passed to execution engine. Aborting.")
            return pd.DataFrame()
        
        model = XGBClassifier()
        model.load_model(MODEL_PATH)
        self.model = model  
            
        working_df = gold_df.copy()
        working_df = working_df.sort_values(by="Date")
        latest_snapshot = working_df.groupby("Symbol").tail(1).reset_index(drop=True)
        
        sector_regime_map = getattr(self, "sector_regime_map", {})
        sector_score_map = getattr(self, "sector_score_map", {})
        
        latest_snapshot["Sector_Regime_Label"] = latest_snapshot["Sector"].map(sector_regime_map).fillna("📈 NEUTRAL_CONSOLIDATION")
        latest_snapshot["Sector_GMM_Factor"] = latest_snapshot["Sector"].map(sector_score_map).fillna(0.0)

        # Production Execution Hard Constraints Block
        domain_mask = (
            (latest_snapshot["is_tradable"] == 1) &
            (latest_snapshot["Feature_Sector_Aligned"] == 1) & 
            (latest_snapshot["Market_Regime_Risk_Off"] == 0) &   
            (latest_snapshot["Feature_RSI"] < 82.0) &
            (latest_snapshot["Close"] >= 15.0)
        )
        
        cluster_gate = latest_snapshot["Sector_GMM_Factor"] >= self.macro_score_threshold
        domain_mask = domain_mask & cluster_gate
        
        valid_universe = latest_snapshot[domain_mask].copy()
        if valid_universe.empty:
            print(f"[WARN] Zero portfolio elements passed the macro criteria threshold ({self.macro_score_threshold}). Preservation enforced.")
            return pd.DataFrame()

        # Align feature store array to drop features excluded during training
        active_features = [
            f for f in FEATURE_COLUMNS 
            if f not in ["Feature_Delivery_Ratio", "Feature_Close_Strength"]
        ]
        
        # Probabilities Matrix Extraction (Shape: N, 3)
        X_live = valid_universe[active_features]
        probabilities_matrix = model.predict_proba(X_live)
        
        # Verify structural footprint before slicing
        if probabilities_matrix.shape[1] < 3:
            raise ValueError(
                f"❌ [CRITICAL CONFIG ERROR] Loaded model binary at '{MODEL_PATH}' returned only "
                f"{probabilities_matrix.shape[1]} classes. The upgraded production pipeline requires a 3-class "
                f"model ('multi:softprob' with num_class=3). Please verify/re-run training."
            )
        
        valid_universe["Prob_Failure_SL"] = probabilities_matrix[:, 0]  
        valid_universe["Prob_Stagnation"] = probabilities_matrix[:, 1]  
        valid_universe["Alpha_ML_Score"]  = probabilities_matrix[:, 2]  # P(Success)
        
        # Calculate Dynamically Mutated Targets to Match Matrix Logic
        close_prices = valid_universe["Close"].values
        atr_14 = valid_universe.get("ATR_14", close_prices * 0.03).values  
        pivot_lows = valid_universe.get("Pivot_Low_30", close_prices * 0.95).values
        sector_aligned_vals = valid_universe["Feature_Sector_Aligned"].values
        
        # Match sector scaling multipliers configured inside your labeling engine
        pt_multipliers = np.where(sector_aligned_vals == 1, 2.5 * 1.2, 2.5 * 0.8)
        sl_multipliers = np.where(sector_aligned_vals == 1, 1.5, 1.5 * 0.66)
        
        rupee_rewards = pt_multipliers * atr_14
        rupee_risks = sl_multipliers * atr_14
        
        # Dynamic Boundaries Sizing Configuration
        profit_targets = close_prices + rupee_rewards
        volatility_sl = close_prices - rupee_risks
        stop_losses = np.maximum(pivot_lows, volatility_sl)
        actual_rupee_risks = np.maximum(close_prices - stop_losses, 1e-9)
        
        valid_universe["Stop_Loss"] = np.round(stop_losses, 2)
        valid_universe["Profit_Target"] = np.round(profit_targets, 2)
        
        # Upgrade: Core Expected Value Calculation Engine
        valid_universe["Expected_Value"] = (valid_universe["Alpha_ML_Score"] * rupee_rewards) - (valid_universe["Prob_Failure_SL"] * actual_rupee_risks)
        
        # Re-Rank Portfolio Optimization Core on Expected Value Metrics
        valid_universe = valid_universe.sort_values(by="Expected_Value", ascending=False).reset_index(drop=True)
        top_20_signals = valid_universe.head(20).copy()
        top_20_signals["Alpha_Rank"] = top_20_signals.index + 1
        
        # Blended UI Score Construction
        top_20_signals["Confidence_Score"] = np.round(
            (0.6 * top_20_signals["Alpha_ML_Score"]) + (0.4 * top_20_signals["Sector_GMM_Factor"]), 2
        )
        
        def assign_presentation_tags(row):
            p_success = row["Alpha_ML_Score"]
            p_fail = row["Prob_Failure_SL"]
            p_stagnate = row["Prob_Stagnation"]
            ev_val = row["Expected_Value"]
            deliv_ratio = row.get("Feature_Delivery_Ratio", 1.0)
            close_strength = row.get("Feature_Close_Strength", 0.5)
            
            base_reason = f"EV: +₹{ev_val:.2f} | P(🚀): {p_success*100:.1f}% | P(🛑): {p_fail*100:.1f}%"
            
            if p_fail > 0.40 or ev_val < 0:
                return pd.Series(["⚠️ HIGH ASYMMETRY HAZARD", f"Negative or dangerous asymmetry profile. | {base_reason}"])
            
            if p_success >= 0.60:
                if deliv_ratio >= 1.15 or close_strength >= 0.65:
                    return pd.Series(["🚀 INSIDER BREAKOUT", f"Institutional volume profile confirmed. | {base_reason}"])
                return pd.Series(["🚀 ACTIVE BREAKOUT", f"Velocity price breakout validated. | {base_reason}"])
            
            if p_stagnate >= 0.50:
                return pd.Series(["🏢 COILING LAUNCHPAD", f"Tight compression breakout candidate. | {base_reason}"])
                
            return pd.Series(["🏢 LAUNCHPAD", f"Standard continuous accumulation setup. | {base_reason}"])

        # Safe multi-column array assignment pattern
        top_20_signals[["Strategic_Label", "Decision_Reason"]] = top_20_signals.apply(assign_presentation_tags, axis=1)

        print(f"🎯 Portfolio Allocation Engine sorting complete. Dispatched {len(top_20_signals)} positions.")
        
        # 🛠️ FIX: Dynamically guarantee that ALL columns required downstream are preserved.
        # Order is preserved using a dictionary, duplicates are dropped, and missing keys are avoided.
        required_ui_cols = [
            "Symbol", "Sector", "Close", "Expected_Value", "Alpha_ML_Score", "Prob_Failure_SL", "Prob_Stagnation",
            "Confidence_Score", "Alpha_Rank", "Stop_Loss", "Profit_Target", "Strategic_Label", "Decision_Reason"
        ]
        
        final_return_columns = list(dict.fromkeys(required_ui_cols + list(FEATURE_COLUMNS)))
        existing_return_columns = [col for col in final_return_columns if col in top_20_signals.columns]
        
        return top_20_signals[existing_return_columns]