import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Import updated modular entities
from clustering import SectorClusterEngine
from rule_engine import StocksRuleEngine
from ml_feature_engg_train_params import run_offline_model_training

from deployment_engine import ProgrammaticDashboardDeployer
from llm_sentiment_engine import GeminiSentimentEngine
from constants import LOOKBACK_YEARS, APPROVED_REGIMES

load_dotenv()

# Operational Configuration Flag
TRAIN_MODE = True  # Set to True when updating XGBoost patterns historically

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
    # STAGE 5: Autonomous LLM Semantic Overlay
    # -------------------------------------------------------------------------
    print("\n[STAGE 5] Querying Autonomous LLM Narrative Analyst (Batch Mode)...")
    llm_engine = GeminiSentimentEngine()
    
    tickers_payload = [
        {"symbol": row["Symbol"], "sector": row["Sector"], "close": row["Close"], "label": row["Strategic_Label"]}
        for idx, row in execution_signals.iterrows()
    ]

    batch_analysis = llm_engine.analyze_batch_narratives(tickers_payload)
    
    sentiments, catalysts, confidence_scores, threats = [], [], [], []
    for row in tickers_payload:
        sym = row["symbol"]
        analysis = batch_analysis.get(sym, {})
        sentiments.append(analysis.get("sentiment", "NEUTRAL"))
        catalysts.append(analysis.get("news_catalyst", "No active catalyst logged."))
        confidence_scores.append(analysis.get("confidence_score", 60))
        threats.append(analysis.get("strategic_threat", "No structural risk identified."))

    execution_signals["Sentiment"] = sentiments
    execution_signals["News_Catalyst"] = catalysts
    execution_signals["Confidence_Score"] = confidence_scores
    execution_signals["Strategic_Threat"] = threats

    # -------------------------------------------------------------------------
    # STAGE 6: Programmatic HTML Generation & Deployment
    # -------------------------------------------------------------------------
    print("\n[STAGE 6] Triggering Automated GitHub Deployment Pipelines...")
    
    deployer_engine = ProgrammaticDashboardDeployer(
        github_token=os.getenv("GITHUB_PAT"),
        repo_owner=os.getenv("REPO_OWNER"),
        repo_name=os.getenv("REPO_NAME"),
        branch=os.getenv("BRANCH")
    )

    # Sort final candidates by raw statistical probability before dashboard generation
    execution_signals = execution_signals.sort_values(by="Alpha_ML_Score", ascending=False)
    html = deployer_engine.generate_html_string(execution_signals)

    # save local
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(" 🟢 Saved dashboard layout locally to 'index.html'.")

    deployer_engine.deploy_to_github(file_content=html, destination_path="index.html")

if __name__ == "__main__":
    run_production_pipeline()