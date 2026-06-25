import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from xgboost import XGBClassifier
from constants import (
    MODEL_PATH, 
    FEATURE_COLUMNS,
    REPORTS_DIR,
    fetch_data_from_nse,
    CALIBRATOR_MODEL,
    TODAY,
    MODEL_HEALTH_METADATA
)

warnings.filterwarnings("ignore")


class StocksRuleEngine:
    """
    Engine responsible for high-performance feature engineering, data sanitation,
    label definition, and executing/monitoring production ML trade signals.
    """

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
        self.metadata_path = MODEL_HEALTH_METADATA
        self.model = None

    def fetch_universe_data(self) -> pd.DataFrame:
        """Delegates directly to unified exchange engine which automatically handles disk-cache hits."""
        filtered_symbols = list(set(self.symbols + ["NIFTY 500"]))
        return fetch_data_from_nse(filtered_symbols, self.symbol_to_sector_map)

    def _parse_and_sanitize_columns(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()
        df.columns = [str(col).replace('ï»¿', '').strip() for col in df.columns]
        
        possible_closes = ['ClosePrice', 'Close', 'close', 'CLOSE', 'ClosePriceParticulars']
        close_col = next((c for c in possible_closes if c in df.columns), None)
        if not close_col: 
            return pd.DataFrame()
            
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
        if index_df.empty: 
            return {}, {}, {}, {}
            
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

    def _engineer_single_asset_features(
        self, 
        group: pd.DataFrame, 
        index_roc_20_map: dict, 
        index_roc_252_map: dict, 
        regime_risk_map: dict, 
        sector_trends: dict, 
        dynamic_alpha_map: dict
    ) -> pd.DataFrame:
        group = group.sort_values(by="Date").copy()
        if len(group) < 252: 
            return pd.DataFrame()

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
        if df.empty: 
            return pd.DataFrame()
            
        index_roc_20, index_roc_252, regime_risk, dynamic_alpha = self._generate_benchmark_regime_maps(df)
        stock_pool_df = df[df["Symbol"] != "NIFTY 500"].sort_values(by="Date").copy()
        if stock_pool_df.empty: 
            return pd.DataFrame()
            
        sector_trends = self._generate_sector_trend_maps(stock_pool_df)
        
        processed_stocks = []
        for _, group in stock_pool_df.groupby("Symbol"):
            feat_df = self._engineer_single_asset_features(group, index_roc_20, index_roc_252, regime_risk, sector_trends, dynamic_alpha)
            if not feat_df.empty: 
                processed_stocks.append(feat_df)
                
        return pd.concat(processed_stocks, ignore_index=True) if processed_stocks else pd.DataFrame()
    
    def engineer_training_labels(
        self, 
        gold_df: pd.DataFrame, 
        pt_horizon: int = 20, 
        pt_mult: float = 2.5, 
        sl_mult: float = 1.5
    ) -> pd.DataFrame:
        print(f"\n🏷️ [LABELING ENGINE] Generating Triple Barrier target matrix (Horizon={pt_horizon} steps)...")
        if gold_df.empty:
            return pd.DataFrame()
            
        df = gold_df.copy().sort_values(by=["Symbol", "Date"]).reset_index(drop=True)
        processed_groups = []
        
        for symbol, group in df.groupby("Symbol"):
            group = group.copy().reset_index(drop=True)
            n_steps = len(group)
            target_labels = np.full(n_steps, np.nan)
            
            close_arr = group["Close"].values
            atr_arr = group["ATR_14"].fillna(group["Close"] * 0.03).values
            
            for i in range(n_steps):
                if i + pt_horizon >= n_steps:
                    continue
                
                entry_price = close_arr[i]
                current_atr = atr_arr[i]
                
                upper_barrier = entry_price + (pt_mult * current_atr)
                lower_barrier = entry_price - (sl_mult * current_atr)
                
                forward_window = close_arr[i + 1 : i + pt_horizon + 1]
                assigned_class = 1
                
                for price in forward_window:
                    if price <= lower_barrier:
                        assigned_class = 0
                        break
                    elif price >= upper_barrier:
                        assigned_class = 2
                        break
                        
                target_labels[i] = assigned_class
                
            group["Strategic_Label"] = target_labels
            processed_groups.append(group)
            
        labeled_master_df = pd.concat(processed_groups, ignore_index=True)
        labeled_master_df = labeled_master_df.dropna(subset=["Strategic_Label"])
        labeled_master_df["Strategic_Label"] = labeled_master_df["Strategic_Label"].astype(int)
        
        print(f"✅ Target allocation matrix built. Retained {len(labeled_master_df)} training entries.")
        return labeled_master_df

    def compile_and_save_health_contract(self, training_pool: pd.DataFrame, master_model, avg_auc, avg_p10, avg_edge, wf_records):
        print("\n⚙️ [METADATA CONTRACT] Generating Model Health Metadata Contract...")
        X_master = training_pool[FEATURE_COLUMNS]
        y_master = training_pool["Strategic_Label"]

        feature_statistics = {}
        for col in FEATURE_COLUMNS:
            feature_statistics[col] = {
                "mean": float(X_master[col].mean()),
                "std": float(X_master[col].std()),
                "min": float(X_master[col].min()),
                "max": float(X_master[col].max())
            }

        label_dist_raw = y_master.value_counts(normalize=True).sort_index().to_dict()
        label_distribution = {str(k): float(v) for k, v in label_dist_raw.items()}

        master_probs = master_model.predict_proba(X_master.values)
        probability_statistics = {
            "failure_mean": float(master_probs[:, 0].mean()),
            "failure_std": float(master_probs[:, 0].std()),
            "stagnation_mean": float(master_probs[:, 1].mean()),
            "stagnation_std": float(master_probs[:, 1].std()),
            "success_mean": float(master_probs[:, 2].mean()),
            "success_std": float(master_probs[:, 2].std())
        }

        if "Sector_Regime_Label" in training_pool.columns:
            regime_distribution = {str(k): float(v) for k, v in training_pool["Sector_Regime_Label"].value_counts(normalize=True).to_dict().items()}
        else:
            regime_distribution = {}

        feature_importance = dict(
            sorted(
                zip(FEATURE_COLUMNS, [float(x) for x in master_model.feature_importances_]),
                key=lambda x: x[1],
                reverse=True
            )
        )

        metadata = {
            "compiled_on": TODAY,
            "features_schema": list(FEATURE_COLUMNS),
            "global_samples_count": len(training_pool),
            "average_walk_forward_auc": float(avg_auc),
            "average_precision_top_10": float(avg_p10),
            "average_pure_alpha_edge": float(avg_edge),
            "label_distribution": label_distribution,
            "probability_statistics": probability_statistics,
            "feature_statistics": feature_statistics,
            "regime_distribution": regime_distribution,
            "feature_importance": feature_importance,
            "walk_forward_folds": wf_records
        }

        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"✅ [METADATA CONTRACT] Stored health profile contract at: '{self.metadata_path}'")
        return metadata

    def check_live_health_degradation(self, live_gold_df: pd.DataFrame) -> bool:
        if not os.path.exists(self.metadata_path):
            print("⚠️ [HEALTH CHECK] Metadata contract missing from file system. Initial training forced.")
            return True

        if live_gold_df.empty:
            return False

        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)

        print("\n" + "="*60)
        print(" 🔍 RUNNING AUTONOMOUS MODEL HEALTH VALIDATION MATRIX")
        print("="*60)

        live_snapshot = live_gold_df.groupby("Symbol").tail(1).reset_index(drop=True)
        drift_signals = 0
        max_drift_threshold_sigmas = 2.5
        
        for col in FEATURE_COLUMNS:
            if col in live_snapshot.columns:
                live_mean = float(live_snapshot[col].mean())
                train_meta = metadata["feature_statistics"].get(col, {})
                train_mean = train_meta.get("mean", live_mean)
                train_std = train_meta.get("std", 1.0)
                
                z_score_distance = abs(live_mean - train_mean) / (train_std + 1e-9)
                if z_score_distance > max_drift_threshold_sigmas:
                    print(f" ❌ [DRIFT DETECTED] Feature '{col}' shifted by {z_score_distance:.2f} sigmas from baseline.")
                    drift_signals += 1

        if drift_signals > (len(FEATURE_COLUMNS) * 0.20):
            print(f"🚨 [RETRAIN TRIGGER] Broad structural feature drift detected across {drift_signals} elements.")
            return True

        try:
            model = XGBClassifier()
            model.load_model(MODEL_PATH)
            
            domain_mask = (
                (live_snapshot["is_tradable"] == 1) &
                (live_snapshot["Feature_Sector_Aligned"] == 1) &
                (live_snapshot["Market_Regime_Risk_Off"] == 0) &
                (live_snapshot["Feature_RSI"] < 82.0) &
                (live_snapshot["Close"] >= 15.0)
            )
            valid_universe = live_snapshot[domain_mask].copy()

            if not valid_universe.empty:
                X_live = valid_universe[FEATURE_COLUMNS].values
                calibrator = joblib.load(CALIBRATOR_MODEL)
                current_probs = calibrator.predict_proba(X_live)
                
                live_success_std = float(current_probs[:, 2].astype(float).std())
                print(f" 📊 Live Class-2 (Success) Probability Dispersion Std: {live_success_std:.4f}")
                
                if live_success_std < 0.03:
                    print("🚨 [RETRAIN TRIGGER] Model distribution collapse identified. Signal variance too low.")
                    return True
        except Exception as ex:
            print(f" [WARN] Skinned predictive model health validation bypass: {str(ex)}")

        perf_ledger_path = os.path.join(REPORTS_DIR, "performance_ledger.csv")
        if os.path.exists(perf_ledger_path):
            try:
                perf_df = pd.read_csv(perf_ledger_path)
                perf_df['Date'] = pd.to_datetime(perf_df['Date'])
                trailing_30_days = perf_df[perf_df['Date'] >= (datetime.now() - pd.Timedelta(days=30))]
                
                if len(trailing_30_days) >= 5:
                    rolling_alpha_edge_30d = trailing_30_days['realized_alpha_edge'].mean()
                    contract_edge_baseline = metadata.get("average_pure_alpha_edge", 0.04)
                    
                    print(f" 📉 Realized 30-Day Pure Alpha Edge: {rolling_alpha_edge_30d:.4f} (Baseline: {contract_edge_baseline:.4f})")
                    if rolling_alpha_edge_30d < (contract_edge_baseline * 0.5):
                        print("🚨 [RETRAIN TRIGGER] Realized trading edge deteriorated past 50% safety hurdle.")
                        return True
            except Exception as pe:
                print(f" [WARN] Performance ledger processing skipped: {str(pe)}")

        print("✅ [MODEL HEALTH HEALTHY] Operational parameters verified. Skipping retrain phase.")
        print("="*60 + "\n")
        return False

    def execute_ml_signals(self, gold_df: pd.DataFrame = None) -> pd.DataFrame:
        if gold_df is None or gold_df.empty:
            print("[WARN] Empty data frame passed to execution engine. Aborting.")
            return pd.DataFrame()

        model = XGBClassifier()
        model.load_model(MODEL_PATH)
        self.model = model

        working_df = gold_df.copy().sort_values(by="Date")
        latest_snapshot = working_df.groupby("Symbol").tail(1).reset_index(drop=True)

        sector_regime_map = getattr(self, "sector_regime_map", {})
        sector_score_map = getattr(self, "sector_score_map", {})

        latest_snapshot["Sector_Regime_Label"] = (
            latest_snapshot["Sector"].map(sector_regime_map).fillna("📈 NEUTRAL_CONSOLIDATION")
        )
        latest_snapshot["Sector_GMM_Factor"] = (
            latest_snapshot["Sector"].map(sector_score_map).fillna(0.0)
        )

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
            print(f"[WARN] Zero portfolio elements passed macro threshold ({self.macro_score_threshold})")
            return pd.DataFrame()

        X_live = valid_universe[FEATURE_COLUMNS]
        calibrator = joblib.load(CALIBRATOR_MODEL)
        probabilities_matrix = calibrator.predict_proba(X_live)

        if probabilities_matrix.shape[1] < 3:
            raise ValueError("❌ Model must be trained as a 3-class classifier.")

        valid_universe["Prob_Failure_SL"] = probabilities_matrix[:, 0]
        valid_universe["Prob_Stagnation"] = probabilities_matrix[:, 1]
        valid_universe["Alpha_ML_Score"] = probabilities_matrix[:, 2]

        print("\n========== PROBABILITY DIAGNOSTICS ==========")
        print(f"Mean P(Failure):  {probabilities_matrix[:,0].mean():.3f}")
        print(f"Mean P(Stagnate): {probabilities_matrix[:,1].mean():.3f}")
        print(f"Mean P(Success):  {probabilities_matrix[:,2].mean():.3f}")

        close_prices = valid_universe["Close"].values
        atr_14 = valid_universe["ATR_14"].fillna(valid_universe["Close"] * 0.03).values if "ATR_14" in valid_universe.columns else close_prices * 0.03
        piv_lows = valid_universe["Pivot_Low_30"].fillna(valid_universe["Close"] * 0.95).values if "Pivot_Low_30" in valid_universe.columns else close_prices * 0.95
        sector_aligned_vals = valid_universe["Feature_Sector_Aligned"].values

        pt_multipliers = np.where(sector_aligned_vals == 1, 2.5 * 1.2, 2.5 * 0.8)
        sl_multipliers = np.where(sector_aligned_vals == 1, 1.5, 1.5 * 0.66)

        rupee_rewards = pt_multipliers * atr_14
        rupee_risks = sl_multipliers * atr_14
        profit_targets = close_prices + rupee_rewards
        volatility_sl = close_prices - rupee_risks
        stop_losses = np.maximum(piv_lows, volatility_sl)
        actual_rupee_risks = np.maximum(close_prices - stop_losses, 1e-9)

        valid_universe["Stop_Loss"] = np.round(stop_losses, 2)
        valid_universe["Profit_Target"] = np.round(profit_targets, 2)

        reward_risk_ratio = rupee_rewards / (actual_rupee_risks + 1e-9)
        valid_universe["Reward_Risk"] = reward_risk_ratio

        print("\n========== REWARD/RISK DIAGNOSTICS ==========")
        print(valid_universe["Reward_Risk"].describe())

        reward_pct = rupee_rewards / close_prices
        risk_pct = actual_rupee_risks / close_prices

        valid_universe["Expected_Value"] = (
            valid_universe["Alpha_ML_Score"] * reward_pct +
            valid_universe["Prob_Stagnation"] * 0.002 -
            valid_universe["Prob_Failure_SL"] * risk_pct
        )

        print("\n========== EV DIAGNOSTICS ==========")
        print(valid_universe["Expected_Value"].describe())
        print(f"Positive EV Stocks: {(valid_universe['Expected_Value'] > 0).sum()}/{len(valid_universe)}")

        valid_universe = valid_universe.sort_values(by="Expected_Value", ascending=False).reset_index(drop=True)
        top_20_signals = valid_universe.head(20).copy()
        top_20_signals["Alpha_Rank"] = top_20_signals.index + 1
        top_20_signals["Confidence_Score"] = np.round((0.6 * top_20_signals["Alpha_ML_Score"] + 0.4 * top_20_signals["Sector_GMM_Factor"]), 2)

        success_baseline = valid_universe["Alpha_ML_Score"].mean()
        stagnation_baseline = valid_universe["Prob_Stagnation"].mean()

        print("\n========== BASELINE CALIBRATION ==========")
        print(f"Mean Success Probability   : {success_baseline:.3f}")
        print(f"Mean Stagnation Probability: {stagnation_baseline:.3f}")

        def assign_presentation_tags(row):
            p_success = row["Alpha_ML_Score"]
            p_fail = row["Prob_Failure_SL"]
            p_stagnate = row["Prob_Stagnation"]
            ev_val = row["Expected_Value"]
            deliv_ratio = row.get("Feature_Delivery_Ratio", 1.0)
            close_strength = row.get("Feature_Close_Strength", 0.5)

            base_reason = f"EV={ev_val:.4f} | P🚀={p_success*100:.1f}% | P🛑={p_fail*100:.1f}% | P🏢={p_stagnate*100:.1f}%"

            if ev_val <= 0:
                return pd.Series(["⚠️ HIGH ASYMMETRY HAZARD", f"Negative EV profile | {base_reason}"])

            if p_success >= (success_baseline + 0.05):
                if deliv_ratio >= 1.15 or close_strength >= 0.65:
                    return pd.Series(["🚀 INSIDER BREAKOUT", f"Institutional accumulation confirmed | {base_reason}"])
                return pd.Series(["🚀 ACTIVE BREAKOUT", f"Momentum expansion profile | {base_reason}"])

            if p_stagnate >= (stagnation_baseline * 2.0):
                return pd.Series(["🏢 INSTITUTIONAL LAUNCHPAD", f"Compression structure detected | {base_reason}"])

            return pd.Series(["🏢 LAUNCHPAD", f"Positive EV accumulation profile | {base_reason}"])

        top_20_signals[["Strategic_Label", "Decision_Reason"]] = top_20_signals.apply(assign_presentation_tags, axis=1)

        print("\n========== FINAL LABEL COUNTS ==========")
        print(top_20_signals["Strategic_Label"].value_counts())
        print(f"\n🎯 Portfolio Allocation Engine dispatched {len(top_20_signals)} positions.")

        required_ui_cols = [
            "Symbol", "Sector", "Close", "Expected_Value", "Reward_Risk", "Alpha_ML_Score",
            "Prob_Failure_SL", "Prob_Stagnation", "Confidence_Score", "Alpha_Rank",
            "Stop_Loss", "Profit_Target", "Strategic_Label", "Decision_Reason"
        ]

        final_return_columns = list(dict.fromkeys(required_ui_cols + list(FEATURE_COLUMNS)))
        existing_return_columns = [col for col in final_return_columns if col in top_20_signals.columns]

        return top_20_signals[existing_return_columns]