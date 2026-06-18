import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Import updated modular entities
from clustering import AuditedSectorClusterEngine
from rule_engine import AuditableMomentumPipeline
from ml_feature_engg_train_params import run_offline_model_training

from deployment_engine import ProgrammaticDashboardDeployer
from llm_sentiment_engine import GeminiSentimentEngine
from constants import SECTOR_REPORTS_PATH, STOCK_ANALYSIS, FEATURE_COLUMNS
from xgboost import XGBClassifier
from constants import MODEL_PATH

load_dotenv()

# Operational Configuration Flag
TRAIN_MODE = True  # Set to True when updating XGBoost patterns historically

def run_production_pipeline():
    print("=" * 110)
    print(" PRODUCTION CONTROL ENGINE: EXECUTING TOP-DOWN HYBRID ML QUANT TRADING SYSTEM")
    print("=" * 110)

    csv_file = "dataset/nse_companies.csv"
    if not os.path.exists(csv_file):
        print(f"[CRITICAL ERROR] File '{csv_file}' not found. Halting pipeline.")
        return

    # -------------------------------------------------------------------------
    # STAGE 1: Execute Macro Sector Rotation Pass
    # -------------------------------------------------------------------------
    print("\n[STAGE 1] Querying Unsupervised Macro Segmentation Engines...")
    macro_engine = AuditedSectorClusterEngine(csv_filename=csv_file, lookback_years=2.0)
    macro_engine.load_mappings_from_csv()
    sector_report = macro_engine.discover_and_export_sectors(output_filename=SECTOR_REPORTS_PATH)

    if sector_report is None or sector_report.empty:
        print("[CRITICAL ERROR] Macro Engine failed to return valid data. Halting.")
        return

    approved_regimes = [
        "🔥 ULTRA_MOMENTUM_LEADERS", 
        "🚀 ACTIVE_BREAKOUT_FIELDS", 
        "📈 STABLE_UPWARD_ACCUMULATION"
    ]

    bullish_sectors = set(
        sector_report[sector_report["Macro_Regime"].isin(approved_regimes)]["Sector"].unique()
    )
    print(f"[STAGE 1] Bullish Clusters Identified: {list(bullish_sectors)}")

    # -------------------------------------------------------------------------
    # STAGE 2: Restrict Universe Stream
    # -------------------------------------------------------------------------
    master_universe_map = macro_engine.sector_mapping 
    mcap_map = macro_engine.company_caps if hasattr(macro_engine, 'company_caps') else {}
    sector_regime_map = dict(zip(sector_report["Sector"], sector_report["Macro_Regime"]))

    actionable_symbols = [
        symbol for symbol, sector in master_universe_map.items() if sector in bullish_sectors
    ]
    print(f"[STAGE 2 GATEKEEPER] Universe restricted from {len(master_universe_map)} to {len(actionable_symbols)} stocks.")

    # -------------------------------------------------------------------------
    # STAGE 3: Core Pipeline Initialization
    # -------------------------------------------------------------------------
    micro_pipeline = AuditableMomentumPipeline(
        symbols=actionable_symbols,
        market_cap_map=mcap_map,
        symbol_to_sector_map=master_universe_map,
        sector_regime_map=sector_regime_map,
        lookback_years=2.0  # Kept at 2.0+ to guarantee warm features data blocks
    )
    
    raw_micro_df = micro_pipeline.fetch_universe_data()
    if raw_micro_df.empty:
        print("[WARN] Technical streams returned zero records. Halting.")
        return

    # -------------------------------------------------------------------------
    # OPTIONAL STAGE: Machine Learning Training Router Hook
    # -------------------------------------------------------------------------
    if TRAIN_MODE:
        print("\n[HOOK] Train configuration active. Commencing XGBoost parameter updates...")
        run_offline_model_training(raw_micro_df, micro_pipeline)

    # -------------------------------------------------------------------------
    # STAGE 4: Feature Store Generation and ML Scorer Execution
    # -------------------------------------------------------------------------
    gold_features_df = micro_pipeline.engineer_gold_features(raw_micro_df)
    gold_features_df = gold_features_df[gold_features_df["Close"] >= 15.0]

    if gold_features_df.empty:
        print("[WARN] Feature matrix empty after minimum asset price sorting. Halting.")
        return

    live_model = XGBClassifier()
    live_model.load_model(MODEL_PATH)

    # Pass everything cleanly to the execution engine
    execution_signals = micro_pipeline.export_execution_signals(
        gold_df=gold_features_df,
        model_classifier=live_model,
        feature_columns=FEATURE_COLUMNS
    )

    if execution_signals is None or execution_signals.empty:
        print("[WARN] Hybrid tree scorer returned zero active trade signals. Halting.")
        return

    # Generate Audit Logging trace
    with open(STOCK_ANALYSIS, "w", encoding="utf-8") as f:
        f.write("=" * 120 + "\n")
        f.write(" PRODUCTION QUANT DATA LOGS: STAGE 2 HYBRID TRACE\n")
        f.write(f" Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 120 + "\n\n")
        for _, row in sector_report.iterrows():
            f.write(f"📍 SECTOR: {row['Sector']:<35} | REGIME: {row['Macro_Regime']}\n")

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
    GITHUB_PAT = os.getenv("GITHUB_PAT")
    REPO_OWNER = os.getenv("REPO_OWNER")
    REPO_NAME = "stock-sector-clustering"
    
    deployer_engine = ProgrammaticDashboardDeployer(
        github_token=GITHUB_PAT if GITHUB_PAT else "",
        repo_owner=REPO_OWNER if REPO_OWNER else "your_github_username",
        repo_name=REPO_NAME,
        branch="main"
    )

    # Sort final candidates by raw statistical probability before dashboard generation
    execution_signals = execution_signals.sort_values(by="Alpha_ML_Score", ascending=False)
    dashboard_html = deployer_engine.generate_html_string(execution_signals)

    if not GITHUB_PAT:
        with open("index.html", "w", encoding="utf-8") as f:
            f.write(dashboard_html)
        print(" 🟢 Saved dashboard layout locally to 'index.html'.")
        return

    deployer_engine.deploy_to_github(file_content=dashboard_html, destination_path="index.html")

if __name__ == "__main__":
    run_production_pipeline()