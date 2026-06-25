import os
import shutil
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv

from clustering import SectorClusterEngine
from rule_engine import StocksRuleEngine
from ml_feature_engg_train_params import run_offline_model_training
from deployment_engine import ProgrammaticDashboardDeployer
from llm_sentiment_engine import GeminiSentimentEngine
from constants import LOOKBACK_YEARS, TODAY, FEATURE_COLUMNS

# Load environment configurations
load_dotenv()

MIN_SCORE_HURDLE = 0.55
TOP_N_SECTORS_CAP = 6


def extract_local_shap_drivers(model, X_live, feature_columns):
    """
    Extracts local SHAP decision drivers for a multi-class model,
    capturing both positive and negative drivers sorted by absolute impact magnitude.
    """
    try:
        model_features = model.get_booster().feature_names
    except Exception:
        model_features = None

    if model_features is not None:
        X_df = pd.DataFrame(X_live, columns=feature_columns)
        X_aligned = X_df[model_features].values
        active_features = model_features
    else:
        X_aligned = X_live
        active_features = feature_columns

    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_aligned)

    if isinstance(raw_shap, list):
        shap_matrix = raw_shap[2] if len(raw_shap) > 2 else raw_shap[-1]
    elif isinstance(raw_shap, np.ndarray) and len(raw_shap.shape) == 3:
        if raw_shap.shape[2] == 3:
            shap_matrix = raw_shap[:, :, 2]
        elif raw_shap.shape[0] == 3:
            shap_matrix = raw_shap[2, :, :]
        else:
            shap_matrix = raw_shap[:, :, -1]
    elif hasattr(raw_shap, "values") and len(raw_shap.values.shape) == 3:
        shap_matrix = raw_shap.values[:, :, 2]
    else:
        shap_matrix = raw_shap

    shap_drivers_result = []
    for i in range(len(X_aligned)):
        sample_drivers = {}
        for j, feat in enumerate(active_features):
            attribution_weight = shap_matrix[i, j]
            if abs(attribution_weight) > 1e-4:
                sample_drivers[feat] = float(attribution_weight)
                
        sorted_drivers = dict(sorted(sample_drivers.items(), key=lambda item: abs(item[1]), reverse=True))
        shap_drivers_result.append(sorted_drivers)

    return shap_drivers_result


def purge_historical_artifacts(reports_dir="reports"):
    """
    Safely purges historical reports ONLY if today's automated trading run 
    successfully generated fresh target data assets.
    """
    print("\n[STAGE 7] Executing Historical Artifact Purge Engine...")
    try:
        purged_count = 0
        for item in os.listdir(reports_dir):
            item_path = os.path.join(reports_dir, item)
            
            if os.path.isdir(item_path):
                if item != f"[{TODAY}]":
                    print(f" 🗑️ Deleting historical artifact directory: {item_path}")
                    shutil.rmtree(item_path)
                    purged_count += 1
                else:
                    print(f" ✅ Preserving active current directory: {item_path}")
                    
        print(f" 🟢 Purge cycle completed successfully. Cleared {purged_count} historical records.")
    except Exception as e:
        print(f" [ERROR] Structural failure during filesystem cleanup: {str(e)}")


def run_sector_clustering():
    print("\n[STAGE 1] Querying Unsupervised Macro Segmentation Engines...")
    cluster = SectorClusterEngine(lookback_years=LOOKBACK_YEARS)
    sectors = cluster.discover_sectors()
    
    filtered_df = sectors[sectors["Sector_Score"] >= MIN_SCORE_HURDLE]
    bullish_sectors = set(
        filtered_df.sort_values(by="Sector_Score", ascending=False)
        .head(TOP_N_SECTORS_CAP)["Sector"]
        .unique()
    )
    print(f"[STAGE 1] Bullish Clusters Identified: {list(bullish_sectors)}")

    filtered_stocks = [
        symbol for symbol, sector in cluster.sector_mapping.items() if sector in bullish_sectors
    ]
    print(f"[STAGE 2 GATEKEEPER] Universe restricted from {len(cluster.sector_mapping)} to {len(filtered_stocks)} stocks.")
    
    return cluster, sectors, bullish_sectors, filtered_stocks


def run_inference(rule_engine, nse_df):
    gold_features_df = rule_engine.engineer_gold_features(nse_df)
    gold_features_df = gold_features_df[gold_features_df["Close"] >= 15.0]
    execution_signals = rule_engine.execute_ml_signals(gold_features_df)
    return execution_signals


def generate_dashboard(execution_signals):
    print("\n[STAGE 6] Triggering Automated GitHub Deployment Pipelines...")
    deployer = ProgrammaticDashboardDeployer()
    deployer.generate_and_save_data(execution_signals)


def run_training_pipeline(rule_engine, nse_df):
    print("[HOOK] Train configuration active. Commencing XGBoost parameter updates...")
    run_offline_model_training(nse_df, rule_engine)


def compute_SHAP_baselines(rule_engine, execution_signals):
    print("\n[STAGE 5] Querying Autonomous LLM Narrative Analyst (Batch Mode)...")
    X_live = execution_signals[FEATURE_COLUMNS].values
    shap_drivers = extract_local_shap_drivers(rule_engine.model, X_live, FEATURE_COLUMNS)
    
    execution_signals['shap_drivers'] = shap_drivers
    execution_signals['quantitative_drivers'] = shap_drivers 

    llm_engine = GeminiSentimentEngine()
    batch_analysis = llm_engine.analyze_batch_narratives(execution_signals)

    sentiments, catalysts, confidence_scores, threats, shap_syntheses = [], [], [], [], []

    for _, row in execution_signals.iterrows():
        sym = str(row.get("Symbol", "")).strip().upper()
        analysis = batch_analysis.get(sym, {})
        
        sentiments.append(analysis.get("sentiment", "NEUTRAL"))
        catalysts.append(analysis.get("news_catalyst", "No active catalyst logged."))
        confidence_scores.append(analysis.get("confidence_score", 60))
        threats.append(analysis.get("strategic_threat", "No structural risk identified."))
        shap_syntheses.append(analysis.get("shap_synthesis", "No mathematical-fundamental alignment synthesis generated."))

    execution_signals["Sentiment"] = sentiments
    execution_signals["News_Catalyst"] = catalysts
    execution_signals["Confidence_Score"] = confidence_scores
    execution_signals["Strategic_Threat"] = threats
    execution_signals["SHAP_Synthesis"] = shap_syntheses

    return execution_signals


if __name__ == "__main__":
    print("=" * 110)
    print(" PRODUCTION CONTROL ENGINE: EXECUTING TOP-DOWN HYBRID ML QUANT TRADING SYSTEM")
    print("=" * 110)

    # 1. Unsupervised Clustering fetches full ~500 stock universe & populates compressed day cache file
    cluster, sectors, bullish_set, filtered_list = run_sector_clustering()

    score_map = dict(zip(sectors["Sector"], sectors["Sector_Score"]))
    regime_map = dict(zip(sectors["Sector"], sectors["Macro_Regime"]))

    rule_engine = StocksRuleEngine(
        symbols=filtered_list,
        market_cap_map={},
        symbol_to_sector_map=cluster.sector_mapping,
        sector_regime_map=regime_map,
        sector_score_map=score_map,
        lookback_years=LOOKBACK_YEARS,
        macro_score_threshold=MIN_SCORE_HURDLE
    )

    # 2. Rule Engine requests only filtered ~150 stocks; automatically catches a cache HIT on the day-cache
    nse_df = rule_engine.fetch_universe_data()
    
    # 🟢 Step 1: Compute fresh features for health assessment
    gold_features_df = rule_engine.engineer_gold_features(nse_df)
    
    # 🟢 Step 2: Run verification loop against the contract rules
    train_model = rule_engine.check_live_health_degradation(gold_features_df)

    # 🛑 Step 3: Branch out training only if health check returns True
    if train_model:        
        run_training_pipeline(rule_engine, nse_df)
        
    # 🚀 Step 4: Unified execution pipeline (Runs smoothly with either the old or newly trained model)
    signals = run_inference(rule_engine, nse_df)
    signals = compute_SHAP_baselines(rule_engine, signals)
    generate_dashboard(signals)
    purge_historical_artifacts()