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
        macro_score_threshold: float = 0.55  # UPGRADE: Dynamic control dial for macro filtering
    ):
        self.symbols = symbols
        self.market_cap_map = market_cap_map or {}
        self.symbol_to_sector_map = symbol_to_sector_map or {}
        self.sector_regime_map = sector_regime_map or {}
        self.sector_score_map = sector_score_map or {}  
        self.lookback_years = lookback_years
        self.macro_score_threshold = macro_score_threshold  # Blended expectation cutoff (0.55 = Strong + Top Neutral)
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

    def execute_ml_signals(self, gold_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        HYBRID SYSTEM PRODUCTION ENGINE (3-Component GMM Continuous Edition)
        Scores, filters, and scales positions using localized stock mechanics 
        cross-referenced with continuous soft macro sector distributions.
        """
        if gold_df is None or gold_df.empty:
            print("[WARN] Empty candidate universe passed to rule engine. Skipping allocation.")
            return pd.DataFrame()
        
        model = XGBClassifier()
        model.load_model(MODEL_PATH)
            
        working_df = gold_df.copy()
        working_df = working_df.sort_values(by="Date")
        latest_snapshot = working_df.groupby("Symbol").tail(1).reset_index(drop=True)
        
        # -------------------------------------------------------------------------
        # UPGRADE: Continuous Factor Map Extraction with Explicit Zero Fallbacks
        # -------------------------------------------------------------------------
        sector_regime_map = getattr(self, "sector_regime_map", {})
        sector_score_map = getattr(self, "sector_score_map", {})

        print("\n🔍 [RULE ENGINE DIAGNOSTIC] Inspecting 3-Tier Map Alignment:")
        print(f" -> Active Macro Score Hurdle Rate: {self.macro_score_threshold}")
        print(f" -> Unique sectors present in today's stock pool: {latest_snapshot['Sector'].unique().tolist()}")
        
        latest_snapshot["Sector_Regime_Label"] = latest_snapshot["Sector"].map(sector_regime_map).fillna("📈 NEUTRAL_SIDEWAYS_CONSOLIDATION")
        latest_snapshot["Sector_GMM_Factor"] = latest_snapshot["Sector"].map(sector_score_map).fillna(0.0)

        # Base Domain Technical Constraints
        domain_mask = (
            (latest_snapshot["is_tradable"] == 1) &
            (latest_snapshot["Feature_Sector_Aligned"] == 1) & 
            (latest_snapshot["Market_Regime_Risk_Off"] == 0) &   
            (latest_snapshot["Feature_RSI"] < 82.0) &
            (latest_snapshot["Close"] >= 15.0)
        )
        
        # -------------------------------------------------------------------------
        # UPGRADE: Continuous Score Threshold Gate (Replaces old K=5 text labels)
        # -------------------------------------------------------------------------
        cluster_gate = latest_snapshot["Sector_GMM_Factor"] >= self.macro_score_threshold
        domain_mask = domain_mask & cluster_gate
        
        rejected_sectors = latest_snapshot[latest_snapshot["Sector_GMM_Factor"] < self.macro_score_threshold]["Sector"].unique()
        if len(rejected_sectors) > 0:
            print(f"🛡️ [GMM FILTER] Guard closed. Excluded assets from macro groups failing hurdle rate: {rejected_sectors}")
            
        valid_universe = latest_snapshot[domain_mask].copy()
        if valid_universe.empty:
            print(f"[WARN] Zero assets cleared GMM macro hurdle ({self.macro_score_threshold}). Enforcing portfolio preservation.")
            return pd.DataFrame()

        # Compute Micro Model Class Probabilities
        X_live = valid_universe[FEATURE_COLUMNS]
        valid_universe["Alpha_ML_Score"] = model.predict_proba(X_live)[:, 1]
        
        # Isolate top candidates and apply structural sequence ranking
        valid_universe = valid_universe.sort_values(by="Alpha_ML_Score", ascending=False).reset_index(drop=True)
        top_20_signals = valid_universe.head(20).copy()
        top_20_signals["Alpha_Rank"] = top_20_signals.index + 1
        
        # Vectorized Risk Stop & Target Boundaries
        close_prices = top_20_signals["Close"].values
        atr_14 = top_20_signals.get("ATR_14", close_prices * 0.03).values  
        pivot_lows = top_20_signals.get("Pivot_Low_30", close_prices * 0.95).values
        
        volatility_sl = close_prices - (2 * atr_14)
        stop_losses = np.maximum(pivot_lows, volatility_sl)
        risk_distances = np.maximum(close_prices - stop_losses, 1e-9)
        profit_targets = close_prices + (2.0 * risk_distances)
        
        top_20_signals["Stop_Loss"] = np.round(stop_losses, 2)
        top_20_signals["Profit_Target"] = np.round(profit_targets, 2)
        
        # -------------------------------------------------------------------------
        # UPGRADE: Continuous Portfolio Sizing Multiplier Blend
        # -------------------------------------------------------------------------
        # Smoothly blends 60% individual asset alpha with 40% GMM mathematical expectation
        top_20_signals["Confidence_Score"] = np.round(
            (0.6 * top_20_signals["Alpha_ML_Score"]) + (0.4 * top_20_signals["Sector_GMM_Factor"]), 2
        )
        
        # Presentation Layer Custom Labels
        def assign_presentation_tags(row):
            score = row["Alpha_ML_Score"]
            conf = row["Confidence_Score"]
            deliv_ratio = row.get("Feature_Delivery_Ratio", 1.0)
            close_strength = row.get("Feature_Close_Strength", 0.5)
            base_reason = f"Model Prob: {score*100:.1f}% | Macro Blend Factor: {conf} | Stop: ₹{row['Stop_Loss']}"
            
            if score >= 0.70:
                if deliv_ratio >= 1.15 or close_strength >= 0.65:
                    return pd.Series(["🚀 INSIDER BREAKOUT", f"Velocity Vol + Inst. Delivery. | {base_reason}"])
                return pd.Series(["🚀 ACTIVE BREAKOUT", f"Validated breakout structures. | {base_reason}"])
            else:
                if deliv_ratio >= 1.20:
                    return pd.Series(["🏢 INSTITUTIONAL LAUNCHPAD", f"Coiling compression with accumulation. | {base_reason}"])
                return pd.Series(["🏢 LAUNCHPAD", f"Standard accumulation layout. | {base_reason}"])

        ui_metadata = top_20_signals.apply(assign_presentation_tags, axis=1)
        top_20_signals["Strategic_Label"] = ui_metadata[0]
        top_20_signals["Decision_Reason"] = ui_metadata[1]

        print(f"🎯 Production Rank Complete. Dispatched {len(top_20_signals)} dynamically scaled assets to allocation framework.")
        
        return top_20_signals[[
            "Symbol", "Sector", "Close", "Alpha_ML_Score", "Confidence_Score", "Alpha_Rank", 
            "Stop_Loss", "Profit_Target", "Strategic_Label", "Decision_Reason",
            "Feature_RSI", "Feature_ATR_Ratio", "Feature_Delivery_Ratio", "Feature_Close_Strength",
            "Feature_Relative_Strength", "Feature_EMA_Dist", "Feature_Volume_Ratio"
        ]]