import datetime
import os
import warnings
import numpy as np
import pandas as pd
from nselib import capital_market
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import json
from constants import (
    CACHE_FILE, TODAY, NSE_DATASET_PATH, fetch_data_from_nse
)

warnings.filterwarnings("ignore")


class SectorClusterEngine:

    def __init__(self, lookback_years: float = 1.2):
        self.csv_filename = NSE_DATASET_PATH
        self.lookback_years = lookback_years
        self.scaler = StandardScaler()
        # Set to 5 clusters to capture all sub-trends without over-segmentation
        self.model = KMeans(n_clusters=5, random_state=42, n_init=10)
        self.sector_features = ["Sector_Rolling_Return", "Sector_Delivery_Avg", "Sector_Volume_Expansion"]
        self.sector_mapping = {}
        self.cache_filename =  CACHE_FILE # Hidden disk cache file

    def load_cached_sectors(self) -> pd.DataFrame:
        """Checks the local hidden JSON cache file for valid, same-day sector tracking frames."""
        if not os.path.exists(self.cache_filename):
            return pd.DataFrame()

        try:
            with open(self.cache_filename, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            if cache_data.get("execution_date") == TODAY:
                print(f"✅ [CACHE HIT] Same-day cluster record discovered ({TODAY}). Restoring sector matrix from disk cache.")
                # Hydrate the internal dictionary mapping needed for downstream tasks
                self.sector_mapping = cache_data["sector_mapping"]
                # Reconstruct the tracking DataFrame from serialized storage records
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
                "records": sector_matrix.to_dict(orient="records")  # Converts DataFrame rows into a clean dictionary list
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
        symbols = sample_df.groupby("Sector").head(15)["Symbol"].tolist()
        return fetch_data_from_nse(symbols, self.sector_mapping)
        

    def discover_sectors(self) -> pd.DataFrame:
        """Gold Layer: Runs K-Means cluster pass, logs details to file, and returns the DataFrame."""
        # Check cache first before executing expensive operations
        self.load_mappings_from_csv()
        
        cached_df = self.load_cached_sectors()
        if not cached_df.empty:
            return cached_df

        raw_df = self.fetch_universe_market_data()
        if raw_df.empty:
            print("[ERR] Combined market matrix generated empty records.")
            return pd.DataFrame()

        close_col = "ClosePrice" if "ClosePrice" in raw_df.columns else "Close"
        qty_col = "TotalTradedQty" if "TotalTradedQty" in raw_df.columns else "TotalTradedQuantity"
        dly_col = "DeliverableQtyPct" if "DeliverableQtyPct" in raw_df.columns else "%DlyQttoTradedQty"

        raw_df["Close"] = pd.to_numeric(raw_df[close_col].astype(str).str.replace(",", ""), errors='coerce')
        raw_df["Volume"] = pd.to_numeric(raw_df[qty_col].astype(str).str.replace(",", ""), errors='coerce')
        raw_df["DeliveryPct"] = pd.to_numeric(raw_df[dly_col].astype(str).str.replace(",", ""), errors='coerce')
        raw_df['Date'] = pd.to_datetime(raw_df['Date'], errors='coerce')

        processed_list = []
        for symbol, group in raw_df.groupby("Symbol"):
            group = group.sort_values("Date").copy()
            group["Return_3M"] = group["Close"].pct_change(60) * 100
            group["Vol_Ratio"] = group["Volume"] / (group["Volume"].rolling(20).mean() + 1e-9)
            processed_list.append(group)
            
        master_df = pd.concat(processed_list, ignore_index=True)
        latest_snapshot = master_df.groupby("Symbol").last().reset_index()
        
        sector_matrix = latest_snapshot.groupby("Sector").agg(
            Sector_Rolling_Return=("Return_3M", "mean"),
            Sector_Delivery_Avg=("DeliveryPct", "mean"),
            Sector_Volume_Expansion=("Vol_Ratio", "mean")
        ).reset_index().dropna()

        # Fit 5-Cluster Machine Learning Pass
        scaled_features = self.scaler.fit_transform(sector_matrix[self.sector_features])
        sector_matrix["Cluster_ID"] = self.model.fit_predict(scaled_features)

        # Extract Unscaled Cluster Centroids Metadata
        centers = self.scaler.inverse_transform(self.model.cluster_centers_)
        sorted_indices = np.argsort(centers[:, 0])
        
        regime_labels = {
            sorted_indices[4]: "🔥 ULTRA_MOMENTUM_LEADERS",
            sorted_indices[3]: "🚀 ACTIVE_BREAKOUT_FIELDS",
            sorted_indices[2]: "📈 STABLE_UPWARD_ACCUMULATION",
            sorted_indices[1]: "⏳ NEUTRAL_SIDEWAYS_CONSOLIDATION",
            sorted_indices[0]: "❄️ DEEP_BEARISH_CAPITULATION"
        }
        
        sector_matrix["Macro_Regime"] = sector_matrix["Cluster_ID"].map(regime_labels)

        unique_clusters = len(sector_matrix["Cluster_ID"].unique())
        if unique_clusters > 1:
            score = silhouette_score(scaled_features, sector_matrix["Cluster_ID"])
            print(f"[MLOPS AUDIT] Global Silhouette Score: {score:.4f}")
            
            if score < 0.25:
                print("[WARN] Weak mathematical clustering. Sectors are overlapping today.")
        else:
            print("[MLOPS AUDIT] Silhouette Score skipped: Single cluster detected or insufficient variance.")

        # Generate custom human-readable reasons for every single sector row
        reasons = []
        for _, row in sector_matrix.iterrows():
            ret = row["Sector_Rolling_Return"]
            dly = row["Sector_Delivery_Avg"]
            vol = row["Sector_Volume_Expansion"]
            regime = row["Macro_Regime"]
            
            reason_str = (
                f"Assigned to {regime} because the sector features a 3-Month return profile of {ret:.1f}%, "
                f"sustained overnight institutional delivery weight of {dly:.1f}%, and an active volume "
                f"expansion multiplier of {vol:.2f}x relative to its historical baseline values."
            )
            reasons.append(reason_str)
            
        sector_matrix["Decision_Reason"] = reasons
        sector_matrix = sector_matrix.sort_values(by="Sector_Rolling_Return", ascending=False)                
        return sector_matrix