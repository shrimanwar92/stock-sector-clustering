import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import json
from constants import (
    CACHE_FILE, TODAY, NSE_DATASET_PATH, fetch_data_from_nse,
    SECTOR_REGIMES
)

warnings.filterwarnings("ignore")

class SectorClusterEngine:

    def __init__(self, lookback_years: float = 1.2):
        self.csv_filename = NSE_DATASET_PATH
        self.lookback_years = lookback_years
        self.scaler = RobustScaler(with_centering=True, with_scaling=True)
        
        # STRATEGY: Core 4-Dimensional Trend-Quality Features
        self.sector_features = [
            "Sector_Relative_Strength", 
            "Sector_Trend_Breadth", 
            "Sector_Long_Term_Breadth",
            "Sector_ADX"
        ]
        self.sector_mapping = {}
        self.cache_filename = CACHE_FILE

    def _calculate_native_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Computes mathematically precise Wilder's Average Directional Index (ADX) via EMA smoothing."""
        df = df.copy()
        
        prev_close = df["Close"].shift(1)
        prev_high = df["High"].shift(1)
        prev_low = df["Low"].shift(1)
        
        tr1 = df["High"] - df["Low"]
        tr2 = (df["High"] - prev_close).abs()
        tr3 = (df["Low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        up_move = df["High"] - prev_high
        down_move = prev_low - df["Low"]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        smoothed_plus_dm = pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()
        smoothed_minus_dm = pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()
        
        plus_di = 100 * (smoothed_plus_dm / (atr + 1e-9))
        minus_di = 100 * (smoothed_minus_dm / (atr + 1e-9))
        
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
        adx = dx.ewm(alpha=alpha, adjust=False).mean()
        
        return adx.fillna(0.0)

    def load_cached_sectors(self) -> pd.DataFrame:
        """Checks the local hidden JSON cache file for valid, same-day sector tracking frames."""
        if not os.path.exists(self.cache_filename):
            return pd.DataFrame()

        try:
            with open(self.cache_filename, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            if cache_data.get("execution_date") == TODAY:
                print(f"✅ [CACHE HIT] Same-day record discovered ({TODAY}). Restoring sector matrix from disk cache.")
                self.sector_mapping = cache_data["sector_mapping"]
                return pd.DataFrame(cache_data["records"])
        except Exception as e:
            print(f"[WARN] Failed reading local workspace cache: {e}. Defaulting to live calculation path.")
        
        return pd.DataFrame()
    
    def save_sectors_to_cache(self, sector_matrix: pd.DataFrame):
        """Serializes the active sector matrix and current system parameters to a text dictionary layout."""
        if sector_matrix.empty:
            return

        try:
            cache_payload = {
                "execution_date": TODAY,
                "sector_mapping": self.sector_mapping,
                "records": sector_matrix.to_dict(orient="records")
            }
            with open(self.cache_filename, "w", encoding="utf-8") as f:
                json.dump(cache_payload, f, indent=4)
            print(f"💾 [CACHE WRITE] Successfully stored today's cluster dictionary parameters to '{self.cache_filename}'.")
        except Exception as e:
            print(f"[WARN] Execution warning: Could not save sector state variables to disk cache file: {e}")

    def load_mappings_from_csv(self):
        """Loads ticker-to-sector configurations from local CSV repository."""
        if not os.path.exists(self.csv_filename):
            raise FileNotFoundError(f"Missing critical source file! Please save '{self.csv_filename}' here.")
        
        df = pd.read_csv(self.csv_filename)
        df.columns = [c.strip() for c in df.columns]
        for _, row in df.dropna(subset=['Symbol', 'Industry']).iterrows():
            symbol = str(row['Symbol']).strip().upper()
            industry = str(row['Industry']).strip().upper().replace(" ", "_")
            self.sector_mapping[symbol] = industry

    def fetch_universe_market_data(self) -> pd.DataFrame:
        """Silver Layer: Collects historical data patterns from market streams."""
        if not self.sector_mapping:
            self.load_mappings_from_csv()
            
        sample_df = pd.DataFrame(list(self.sector_mapping.items()), columns=["Symbol", "Sector"])
        symbols = sample_df["Symbol"].tolist()
        symbols = list(set(symbols + ["NIFTY 500", "NIFTY_500", "^CNX500"]))
        return fetch_data_from_nse(symbols, self.sector_mapping)

    def discover_sectors(self) -> pd.DataFrame:
        """Gold Layer: Executes deterministic multi-factor ranking and assigns dynamic regime tags."""
        self.load_mappings_from_csv()
        
        cached_df = self.load_cached_sectors()
        if not cached_df.empty:
            return cached_df

        raw_df = self.fetch_universe_market_data()
        if raw_df.empty:
            print("[ERR] Combined market matrix generated empty records.")
            return pd.DataFrame()

        if "Symbol" in raw_df.columns:
            raw_df["Symbol"] = raw_df["Symbol"].astype(str).str.strip().str.upper()

        close_col = "ClosePrice" if "ClosePrice" in raw_df.columns else "Close"
        high_col = "HighPrice" if "HighPrice" in raw_df.columns else "High"
        low_col = "LowPrice" if "LowPrice" in raw_df.columns else "Low"

        raw_df["Close"] = pd.to_numeric(raw_df[close_col].astype(str).str.replace(",", ""), errors='coerce')
        raw_df["High"] = pd.to_numeric(raw_df[high_col].astype(str).str.replace(",", ""), errors='coerce')
        raw_df["Low"] = pd.to_numeric(raw_df[low_col].astype(str).str.replace(",", ""), errors='coerce')
        raw_df['Date'] = pd.to_datetime(raw_df['Date'], errors='coerce').dt.normalize()

        index_mask = raw_df["Symbol"].isin(["NIFTY 500", "NIFTY_500", "^CNX500"])
        nifty_df = raw_df[index_mask].sort_values("Date").copy()
        
        if not nifty_df.empty:
            nifty_df = nifty_df.groupby("Date").last().reset_index()
            nifty_df["Nifty_Return_3M"] = nifty_df["Close"].pct_change(60) * 100
            nifty_return_map = dict(zip(nifty_df["Date"], nifty_df["Nifty_Return_3M"]))
        else:
            print("[WARN] Benchmark index match failed. Defaulting to median universe returns.")
            nifty_return_map = {}

        stock_pool_df = raw_df[~index_mask].copy()
        processed_list = []
        
        for symbol, group in stock_pool_df.groupby("Symbol"):
            group = group.sort_values("Date").copy()
            if len(group) < 60:
                continue
                
            group["Return_3M"] = group["Close"].pct_change(60) * 100
            
            if nifty_return_map:
                group["Nifty_Ret_3M"] = group["Date"].map(nifty_return_map)
            else:
                group["Nifty_Ret_3M"] = np.nan
            
            processed_list.append(group)
            
        if not processed_list:
            print("[ERR] Insufficient data streams to execute feature mapping.")
            return pd.DataFrame()
            
        master_df = pd.concat(processed_list, ignore_index=True)
        
        if master_df["Nifty_Ret_3M"].isna().all():
            median_market_returns = master_df.groupby("Date")["Return_3M"].transform("median")
            master_df["Nifty_Ret_3M"] = median_market_returns.fillna(0.0)
        else:
            master_df["Nifty_Ret_3M"] = master_df["Nifty_Ret_3M"].bfill().ffill().fillna(0.0)

        master_df["Relative_Strength_3M"] = master_df["Return_3M"] - master_df["Nifty_Ret_3M"]

        finalized_stocks = []
        for symbol, group in master_df.groupby("Symbol"):
            group["ADX"] = self._calculate_native_adx(group, period=14)
            group["EMA_50"] = group["Close"].ewm(span=50, adjust=False).mean()
            group["EMA_200"] = group["Close"].ewm(span=200, adjust=False).mean()
            
            group["Above_EMA_50"] = (group["Close"] > group["EMA_50"]).astype(int)
            group["Above_EMA_200"] = (group["Close"] > group["EMA_200"]).astype(int)
            
            finalized_stocks.append(group)

        master_df = pd.concat(finalized_stocks, ignore_index=True)
        latest_snapshot = master_df.groupby("Symbol").last().reset_index()
        
        # Defensive Minimum Size Guard
        sector_counts = latest_snapshot["Sector"].value_counts()
        valid_sectors = sector_counts[sector_counts >= 5].index
        latest_snapshot = latest_snapshot[latest_snapshot["Sector"].isin(valid_sectors)]
        
        if latest_snapshot.empty:
            print("[ERR] No macro sectors matched minimum asset footprint boundaries.")
            return pd.DataFrame()

        # Generate Core Structural Matrix
        sector_matrix = latest_snapshot.groupby("Sector").agg(
            Sector_Rolling_Return=("Return_3M", "mean"),
            Sector_Relative_Strength=("Relative_Strength_3M", "mean"),
            Sector_Trend_Breadth=("Above_EMA_50", "mean"),
            Sector_Long_Term_Breadth=("Above_EMA_200", "mean"),
            Sector_ADX=("ADX", "mean")
        ).reset_index().dropna()

        # Standardize features using Robust scaling
        scaled_features = self.scaler.fit_transform(sector_matrix[self.sector_features])
        
        # -------------------------------------------------------------------------
        # UPGRADE: Deterministic Logistic Factor Scoring Engine
        # Replaces unstable GMM optimizations with a mathematical trend vector.
        # -------------------------------------------------------------------------
        raw_composite = (
            0.40 * scaled_features[:, 0] +  # Sector_Relative_Strength
            0.20 * scaled_features[:, 1] +  # Sector_Trend_Breadth
            0.20 * scaled_features[:, 2] +  # Sector_Long_Term_Breadth
            0.20 * scaled_features[:, 3]    # Sector_ADX
        )
        
        # Pass through Sigmoid function to cleanly bind values between 0.0 and 1.0 (Median = 0.50)
        sector_matrix["Sector_Score"] = np.round(1 / (1 + np.exp(-raw_composite)), 4)

        # -------------------------------------------------------------------------
        # UPGRADE: Continuous Soft Probability Emulator (RBF Kernel)
        # Reconstructs smooth transitions (e.g. 0.82, 0.16, 0.02) without statistical drift
        # -------------------------------------------------------------------------
        centers = [0.20, 0.50, 0.80]  # Idealized centers for Weak, Neutral, and Strong scores
        prob_matrix = np.zeros((len(sector_matrix), 3))
        
        for i, center in enumerate(centers):
            # Compute radial basis function distance to simulate a stable Gaussian density
            prob_matrix[:, i] = np.exp(-((sector_matrix["Sector_Score"] - center) / 0.22) ** 2)
            
        # Softmax normalization ensures probabilities sum perfectly to 1.0
        prob_matrix /= prob_matrix.sum(axis=1, keepdims=True)

        regime_keys = ["DEEP_BEARISH_CAPITULATION", "NEUTRAL_SIDEWAYS_CONSOLIDATION", "LEADING_MOMENTUM_ACCELERATION"]
        for i, key in enumerate(regime_keys):
            sector_matrix[f"Prob_{key}"] = np.round(prob_matrix[:, i], 4)

        # -------------------------------------------------------------------------
        # PRODUCTION LABELING: Symmetric Slicing Around Market Medians
        # -------------------------------------------------------------------------
        def assign_deterministic_regime(row):
            score = row["Sector_Score"]
            if score >= 0.60:
                return SECTOR_REGIMES["STRONG"]
            elif score <= 0.40:
                return SECTOR_REGIMES["WEAK"]
            else:
                return SECTOR_REGIMES["NEUTRAL"]

        sector_matrix["Macro_Regime"] = sector_matrix.apply(assign_deterministic_regime, axis=1)

        # Generate Descriptive Audit Strings
        reasons = []
        for idx, row in sector_matrix.iterrows():
            rs = row["Sector_Relative_Strength"]
            breadth = row["Sector_Trend_Breadth"] * 100
            lt_breadth = row["Sector_Long_Term_Breadth"] * 100
            adx = row["Sector_ADX"]
            regime = row["Macro_Regime"]
            score = row["Sector_Score"]
            
            reason_str = (
                f"Assigned category '{regime}' (Continuous Score: {score:.4f}). "
                f"Relative Alpha: {rs:+.2f}%. Sector ADX: {adx:.2f} | "
                f"Trend Breadth (EMA50): {breadth:.1f}% | Structural Breadth (EMA200): {lt_breadth:.1f}%."
            )
            reasons.append(reason_str)
            
        sector_matrix["Decision_Reason"] = reasons
        sector_matrix = sector_matrix.sort_values(by="Sector_Score", ascending=False).reset_index(drop=True)
        
        self.save_sectors_to_cache(sector_matrix)
        return sector_matrix