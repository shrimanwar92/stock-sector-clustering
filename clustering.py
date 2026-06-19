import os
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import json
from constants import (
    CACHE_FILE, TODAY, NSE_DATASET_PATH, fetch_data_from_nse,
    ORDERED_SECTOR_REGIMES
)

warnings.filterwarnings("ignore")


class SectorClusterEngine:

    def __init__(self, lookback_years: float = 1.2):
        self.csv_filename = NSE_DATASET_PATH
        self.lookback_years = lookback_years
        self.scaler = StandardScaler()
        self.model = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=10)
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
        """Gold Layer: Runs GMM clustering pass, calculates soft assignments, and returns the DataFrame."""
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

        scaled_features = self.scaler.fit_transform(sector_matrix[self.sector_features])
        
        # Hyperparameter Tuning Via BIC (Model Complexity Balance Selection)
        max_candidate_k = min(6, len(sector_matrix) - 1)
        candidate_k_values = list(range(2, max_candidate_k + 1))
        
        best_bic = float("inf")
        optimal_k = 5
        best_model = None
        
        for k in candidate_k_values:
            test_model = GaussianMixture(n_components=k, covariance_type='full', random_state=42, n_init=5)
            test_model.fit(scaled_features)
            bic_score = test_model.bic(scaled_features)
            
            if bic_score < best_bic:
                best_bic = bic_score
                optimal_k = k
                best_model = test_model

        print(f"🤖 [BIC OPTIMIZATION] Selected optimal regime count K={optimal_k} based on lowest BIC score of {best_bic:.2f}")
        self.model = best_model
        
        # MLOps Density Audit
        log_likelihood = self.model.score(scaled_features)
        print(f"[MLOPS AUDIT] GMM Average Log-Likelihood: {log_likelihood:.4f}")
        if log_likelihood < -10.0:
            print("[WARN] Anomalous distribution matrix. Sector metrics are highly scattered today.")

        # Extract soft probabilities and hard assignments from optimized GMM density matrix
        hard_clusters = self.model.predict(scaled_features)
        soft_probabilities = self.model.predict_proba(scaled_features)

        # Align clusters from worst absolute return performance to highest
        centers = self.scaler.inverse_transform(self.model.means_)
        sorted_indices = np.argsort(centers[:, 0])
        
        sector_matrix["Cluster_ID"] = hard_clusters
        
        # Map a standardized momentum reward ranking factor based on performance index orientation
        rank_score_lookup = {sorted_indices[rank]: np.round(rank / (optimal_k - 1), 2) for rank in range(optimal_k)}
        sector_matrix["Sector_Score"] = sector_matrix["Cluster_ID"].map(rank_score_lookup)

        # Inject clean probability values as independent data frame features (e.g., Prob_ULTRA_MOMENTUM_LEADERS)
        labels_ordered = list(ORDERED_SECTOR_REGIMES.keys())
        for i, idx in enumerate(sorted_indices):
            col_name = f"Prob_{labels_ordered[i]}"
            sector_matrix[col_name] = np.round(soft_probabilities[:, idx], 4)

        # -------------------------------------------------------------------------
        # UPDATE: Map Percentile Directly to Downstream Labeled Regimes with Icons
        # -------------------------------------------------------------------------
        def assign_dynamic_regime_label(row):
            cluster_id = row["Cluster_ID"]
            # Locate where this cluster sits within the ordered performance rank array
            rank_pos = np.where(sorted_indices == cluster_id)[0][0]
            percentile = rank_pos / (optimal_k - 1)
            
            if percentile == 1.0: 
                key = "ULTRA_MOMENTUM_LEADERS"
            elif percentile >= 0.66: 
                key = "ACTIVE_BREAKOUT_FIELDS"
            elif percentile >= 0.33: 
                key = "STABLE_UPWARD_ACCUMULATION"
            elif percentile > 0.0: 
                key = "NEUTRAL_SIDEWAYS_CONSOLIDATION"
            else: 
                key = "DEEP_BEARISH_CAPITULATION"
                
            # Safely fetch the exact formatted string containing the downstream icon
            return ORDERED_SECTOR_REGIMES[key]

        # Apply the layout mapping to populate the pipeline framework
        sector_matrix["Macro_Regime"] = sector_matrix.apply(assign_dynamic_regime_label, axis=1)

        # Generate readable reasoning using the icon-appended labels
        best_cluster_idx = sorted_indices[-1]
        reasons = []
        for idx, row in sector_matrix.iterrows():
            ret = row["Sector_Rolling_Return"]
            regime = row["Macro_Regime"] # Pulls icon directly (e.g. "🔥 ULTRA_MOMENTUM_LEADERS")
            score = row["Sector_Score"]
            p_top = soft_probabilities[idx, best_cluster_idx] * 100
            
            reason_str = (
                f"Assigned hard category '{regime}' (Factor Score: {score}). "
                f"Features a 3-Month return profile of {ret:.1f}%. The Gaussian distribution maps a "
                f"{p_top:.1f}% explicit membership convergence with the leading macro vector."
            )
            reasons.append(reason_str)
            
        sector_matrix["Decision_Reason"] = reasons
        sector_matrix = sector_matrix.sort_values(by="Sector_Score", ascending=False).reset_index(drop=True)
        
        # Cache outputs before exporting
        self.save_sectors_to_cache(sector_matrix)
        return sector_matrix