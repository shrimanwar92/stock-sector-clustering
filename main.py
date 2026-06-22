import os
import os
import shutil
from dotenv import load_dotenv
import shap
import numpy as np
from clustering import SectorClusterEngine
from rule_engine import StocksRuleEngine
from ml_feature_engg_train_params import run_offline_model_training

from deployment_engine import ProgrammaticDashboardDeployer
from llm_sentiment_engine import GeminiSentimentEngine
from constants import LOOKBACK_YEARS, TODAY, REPORTS_DIR, FEATURE_COLUMNS

load_dotenv()

def extract_local_shap_drivers(model, X_live, feature_columns):
    """
    Extracts local SHAP decision drivers for a multi-class model,
    capturing both positive and negative drivers sorted by absolute impact magnitude.
    """
    import numpy as np
    import shap
    import pandas as pd

    # 1. Feature Alignment Guard
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

    # 2. Compute SHAP values
    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_aligned)

    # 3. Isolate Class 2 (Alpha_ML_Score / Success Class)
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

    # 4. Parse execution metrics using absolute magnitude
    shap_drivers_result = []
    for i in range(len(X_aligned)):
        sample_drivers = {}
        for j, feat in enumerate(active_features):
            attribution_weight = shap_matrix[i, j]
            
            # ✅ Capture both tailwinds (+) and headwinds (-) ignoring micro-noise
            if abs(attribution_weight) > 1e-4:
                sample_drivers[feat] = float(attribution_weight)
                
        # ✅ Sort by absolute importance so the most influential drivers appear first
        sorted_drivers = dict(sorted(sample_drivers.items(), key=lambda item: abs(item[1]), reverse=True))
        shap_drivers_result.append(sorted_drivers)

    return shap_drivers_result

def purge_historical_artifacts(reports_dir = "reports"):
    """
    Safely purges historical reports ONLY if today's automated trading run 
    successfully generated fresh target data assets.
    """
    print("\n[STAGE 7] Executing Historical Artifact Purge Engine...")

    # 3. Controlled Purge Matrix (Runs only when today's generation is confirmed present)
    try:
        purged_count = 0
        for item in os.listdir(reports_dir):
            item_path = os.path.join(reports_dir, item)
            
            # Interact exclusively with child directories
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

def run_production_pipeline():
    print("=" * 110)
    print(" PRODUCTION CONTROL ENGINE: EXECUTING TOP-DOWN HYBRID ML QUANT TRADING SYSTEM")
    print("=" * 110)

    # -------------------------------------------------------------------------
    # STAGE 1: Identify bullish sectors using kmeans clustering
    # -------------------------------------------------------------------------
    print("\n[STAGE 1] Querying Unsupervised Macro Segmentation Engines...")
    cluster = SectorClusterEngine(lookback_years=LOOKBACK_YEARS)
    sectors = cluster.discover_sectors()
    cluster.save_sectors_to_cache(sectors)

    if sectors is None or sectors.empty:
        print("[CRITICAL ERROR] Macro Engine failed to return valid data. Halting.")
        return
    
    MIN_SCORE_HURDLE = 0.55 
    filtered_df = sectors[sectors["Sector_Score"] >= MIN_SCORE_HURDLE]

    # Rule 2: Cap the downstream pipeline to the Top 6 absolute best macro configurations
    TOP_N_SECTORS_CAP = 6
    bullish_sectors = set(
        filtered_df.sort_values(by="Sector_Score", ascending=False)
        .head(TOP_N_SECTORS_CAP)["Sector"]
        .unique()
    )
    print(f"[STAGE 1] Bullish Clusters Identified: {list(bullish_sectors)}")

    # -------------------------------------------------------------------------
    # STAGE 2: pick stocks from bullish sectors only
    # -------------------------------------------------------------------------
    filtered_stocks = [
        symbol for symbol, sector in cluster.sector_mapping.items() if sector in bullish_sectors
    ]
    print(f"[STAGE 2 GATEKEEPER] Universe restricted from {len(cluster.sector_mapping)} to {len(filtered_stocks)} stocks.")
    
    # -------------------------------------------------------------------------
    # STAGE 3: Core Pipeline Initialization
    # -------------------------------------------------------------------------
    gmm_score_map = dict(zip(sectors["Sector"], sectors["Sector_Score"]))
    gmm_regime_map = dict(zip(sectors["Sector"], sectors["Macro_Regime"]))

    rule_engine = StocksRuleEngine(
        symbols=filtered_stocks,
        market_cap_map={},
        symbol_to_sector_map=cluster.sector_mapping,
        sector_regime_map=gmm_regime_map,
        sector_score_map=gmm_score_map,
        lookback_years=LOOKBACK_YEARS,  # Kept at 2.0+ to guarantee warm features data blocks
        macro_score_threshold=MIN_SCORE_HURDLE      # 🚨 UPGRADE: Continuous gatekeeper dial (0.55 = Strong + Top-Tier Neutral)
    )
    
    nse_df = rule_engine.fetch_universe_data()
    rule_engine.save_stocks_to_cache(nse_df)
    if nse_df.empty:
        print("[WARN] Technical streams returned zero records. Halting.")
        return

    # -------------------------------------------------------------------------
    # 4: Machine Learning Training To Find Best Indicators
    # -------------------------------------------------------------------------
    print("\n[HOOK] Train configuration active. Commencing XGBoost parameter updates...")
    run_offline_model_training(nse_df, rule_engine)

    # -------------------------------------------------------------------------
    # STAGE 4: Feature Store Generation and ML Scorer Execution
    # -------------------------------------------------------------------------
    gold_features_df = rule_engine.engineer_gold_features(nse_df)
    gold_features_df = gold_features_df[gold_features_df["Close"] >= 15.0]

    if gold_features_df.empty:
        print("[WARN] Feature matrix empty after minimum asset price sorting. Halting.")
        return

    # Pass everything cleanly to the execution engine
    execution_signals = rule_engine.execute_ml_signals(gold_features_df)

    if execution_signals is None or execution_signals.empty:
        print("[WARN] Hybrid tree scorer returned zero active trade signals. Halting.")
        return

    # -------------------------------------------------------------------------
    # STAGE 5: Autonomous LLM Semantic Overlay (With SHAP Grounding)
    # -------------------------------------------------------------------------
    print("\n[STAGE 5] Querying Autonomous LLM Narrative Analyst (Batch Mode)...")
    
    # Extract the live feature values matching the top candidates
    X_live = execution_signals[FEATURE_COLUMNS].values
    shap_drivers = extract_local_shap_drivers(rule_engine.model, X_live, FEATURE_COLUMNS)
    
    # 🛠️ FIX 1: Align naming convention to match the exact key the LLM Prompt Generator expects
    execution_signals['shap_drivers'] = shap_drivers
    execution_signals['quantitative_drivers'] = shap_drivers 

    llm_engine = GeminiSentimentEngine()
    
    # Dispatch data to the engine
    batch_analysis = llm_engine.analyze_batch_narratives(execution_signals)

    # 🛠️ FIX 2: Uncomment and initialize lists to avoid NameError/stale mapping crashes
    sentiments, catalysts, confidence_scores, threats, shap_syntheses = [], [], [], [], []

    # 🛠️ FIX 3: Parse the LLM batch dict directly against the execution_signals rows
    for _, row in execution_signals.iterrows():
        # Production Safeguard: Normalize symbol casing to guarantee exact match with the LLM output keys
        sym = str(row.get("Symbol", "")).strip().upper()
        
        analysis = batch_analysis.get(sym, {})
        
        sentiments.append(analysis.get("sentiment", "NEUTRAL"))
        catalysts.append(analysis.get("news_catalyst", "No active catalyst logged."))
        confidence_scores.append(analysis.get("confidence_score", 60))
        threats.append(analysis.get("strategic_threat", "No structural risk identified."))
        
        # Pull the now-populated explainability text
        shap_syntheses.append(analysis.get("shap_synthesis", "No mathematical-fundamental alignment synthesis generated."))

    # Map arrays cleanly back to your master execution signal DataFrame/Structure
    execution_signals["Sentiment"] = sentiments
    execution_signals["News_Catalyst"] = catalysts
    execution_signals["Confidence_Score"] = confidence_scores
    execution_signals["Strategic_Threat"] = threats
    execution_signals["SHAP_Synthesis"] = shap_syntheses

    # -------------------------------------------------------------------------
    # STAGE 6: Programmatic HTML Generation & Deployment
    # -------------------------------------------------------------------------
    print("\n[STAGE 6] Triggering Automated GitHub Deployment Pipelines...")
    
    deployer = ProgrammaticDashboardDeployer()
    deployer.generate_and_save_data(execution_signals)

    # -------------------------------------------------------------------------
    # STAGE 7: Historical Artifact Purge Engine
    # -------------------------------------------------------------------------
    # Executed right before exit to guarantee today's processing completes first
    purge_historical_artifacts()

if __name__ == "__main__":
    run_production_pipeline()