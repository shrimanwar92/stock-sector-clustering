import os
import pandas as pd
from clustering import AuditedSectorClusterEngine
from handler import AuditableMomentumPipeline
from datetime import datetime
from deployment_engine import ProgrammaticDashboardDeployer
from dotenv import load_dotenv
from llm_sentiment_engine import GeminiSentimentEngine
from constants import (
    SECTOR_REPORTS_PATH, STOCK_ANALYSIS, EXECUTION_SIGNALS
)

load_dotenv()

def run_production_pipeline():
    print("=" * 110)
    print(" PRODUCTION CONTROL ENGINE: EXECUTING TOP-DOWN QUANT TRADING LAYERS")
    print("=" * 110)

    csv_file = "dataset/nse_companies.csv"
    if not os.path.exists(csv_file):
        print(f"[CRITICAL ERROR] File '{csv_file}' not found. Halting pipeline.")
        return

   # -------------------------------------------------------------------------
    # STAGE 1: Execute Macro Sector Rotation Pass
    # -------------------------------------------------------------------------
    print("\n[STAGE 1] Querying Unsupervised Macro Segmentation Engines...")
    macro_engine = AuditedSectorClusterEngine(csv_filename=csv_file, lookback_years=1.2)
    macro_engine.load_mappings_from_csv()
    sector_report = macro_engine.discover_and_export_sectors(output_filename=SECTOR_REPORTS_PATH)

    if sector_report is None or sector_report.empty:
        print("[CRITICAL ERROR] Macro Engine failed to return valid data. Halting.")
        return

    # Define our approved high-performing cluster regimes
    approved_regimes = [
        "🔥 ULTRA_MOMENTUM_LEADERS", 
        "🚀 ACTIVE_BREAKOUT_FIELDS", 
        "📈 STABLE_UPWARD_ACCUMULATION"
    ]

    # Filter the sector report to isolate approved sectors only
    bullish_sectors = set(
        sector_report[sector_report["Macro_Regime"].isin(approved_regimes)]["Sector"].unique()
    )
    print(f"[STAGE 2] Bullish Clusters Identified: {list(bullish_sectors)}")

    # -------------------------------------------------------------------------
    # STAGE 2: Restrict Universe BEFORE Technical Download Stream
    # -------------------------------------------------------------------------
    master_universe_map = macro_engine.sector_mapping  # Map of Ticker -> Sector
    
    # 🟢 CRITICAL CHANGE: Only keep tickers if their mapped sector is in our bullish whitelist
    actionable_symbols = [
        symbol for symbol, sector in master_universe_map.items()
        if sector in bullish_sectors
    ]
    
    print(f"[STAGE 2 GATEKEEPER] Universe restricted from {len(master_universe_map)} to {len(actionable_symbols)} stocks.")

    # -------------------------------------------------------------------------
    # STAGE 3: Run Micro Technical Rules Firewall Pass
    # -------------------------------------------------------------------------
    print(f"\n[STAGE 3] Loading {len(actionable_symbols)} Actionable Assets into Technical Firewall...")
    
    mcap_map = macro_engine.company_caps if hasattr(macro_engine, 'company_caps') else {}
    
    # Create sector regime lookup map for downstream verification checks
    sector_regime_map = dict(zip(sector_report["Sector"], sector_report["Macro_Regime"]))

    # Pass ONLY the actionable_symbols list instead of all_symbols
    micro_pipeline = AuditableMomentumPipeline(
        symbols=actionable_symbols,  # 🟢 Only download what we care about!
        market_cap_map=mcap_map,
        symbol_to_sector_map=master_universe_map,
        sector_regime_map=sector_regime_map,
        lookback_years=1.2
    )
    
    raw_micro_df = micro_pipeline.fetch_universe_data()
    
    if raw_micro_df.empty:
        print("[WARN] Technical streams returned zero records. Halting.")
        return

    gold_features_df = micro_pipeline.engineer_gold_features(raw_micro_df)
    gold_features_df = gold_features_df[gold_features_df["Close"] >= 15.0]

    print(gold_features_df.head(5))
    #import sys
    #sys.exit(0)

    # Process firewall logic and write output to disk[cite: 11]
    execution_signals = micro_pipeline.export_execution_signals(
        gold_df=gold_features_df, output_filename=EXECUTION_SIGNALS
    )

    if execution_signals is None or execution_signals.empty:
        print("[WARN execution_signals] Micro technical rules engine returned zero active signals. Halting.")
        return

    # 🟢 NEW: Create a clean routing trail tracking document for Stage 2 auditing[cite: 11]
    with open(STOCK_ANALYSIS, "w", encoding="utf-8") as f:
        f.write("=" * 120 + "\n")
        f.write(" PRODUCTION QUANT DATA LOGS: STAGE 2 CLUSTER FILTER TRACE\n")
        f.write(f" Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 120 + "\n\n")
        for _, row in sector_report.iterrows():
            f.write(f"📍 SECTOR: {row['Sector']:<35} | REGIME: {row['Macro_Regime']}\n")

    # -------------------------------------------------------------------------
    # STAGE 5: Autonomous LLM Semantic Overlay (Batch Optimized)[cite: 11]
    # -------------------------------------------------------------------------
    print("\n[STAGE 5] Querying Autonomous LLM Narrative Analyst (Batch Mode)...")
    llm_engine = GeminiSentimentEngine()
    
    tickers_payload = []
    for idx, row in execution_signals.iterrows():
        tickers_payload.append({
            "symbol": row["Symbol"],
            "sector": row["Sector"],
            "close": row["Close"],
            "label": row["Strategic_Label"]
        })

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
    # STAGE 6: Programmatic HTML Generation & Live GitHub Actions Deployment[cite: 11]
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

    dashboard_html = deployer_engine.generate_html_string(execution_signals)

    if not GITHUB_PAT:
        with open("index.html", "w", encoding="utf-8") as f:
            f.write(dashboard_html)
        print(" 🟢 Saved dashboard layout locally to 'index.html'.")
        return

    deployer_engine.deploy_to_github(file_content=dashboard_html, destination_path="index.html")

if __name__ == "__main__":
    run_production_pipeline()