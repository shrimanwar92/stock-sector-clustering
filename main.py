import os
import pandas as pd
from clustering import AuditedSectorClusterEngine
from handler import AuditableMomentumPipeline
from datetime import date, datetime
from deployment_engine import ProgrammaticDashboardDeployer
from dotenv import load_dotenv
from llm_sentiment_engine import GeminiSentimentEngine

load_dotenv()

TODAY = date.today().strftime("%d-%m-%Y")
os.makedirs(f"reports/[{TODAY}]", exist_ok=True)
SECTOR_REPORTS_PATH = f"reports/[{TODAY}]/sector_audit_report.txt"
STOCK_ANALYSIS = f"reports/[{TODAY}]/stock_analysis_report.txt"
EXECUTION_SIGNALS = f"reports/[{TODAY}]/execution_signals_report.txt"


def run_production_pipeline():
    print("=" * 110)
    print(" PRODUCTION CONTROL ENGINE: EXECUTING TOP-DOWN QUANT TRADING LAYERS")
    print("=" * 110)

    csv_file = "dataset/nse_companies.csv"
    if not os.path.exists(csv_file):
        print(f"[CRITICAL ERROR] File '{csv_file}' not found. Please place your file in this folder.")
        return

    # -------------------------------------------------------------------------
    # STAGE 1: Execute 5-Cluster Macro Sector Rotation Pass (With Cache-Lookup Gateways)
    # -------------------------------------------------------------------------
    print("\n[STAGE 1] Querying Unsupervised Macro Segmentation Engines...")
    macro_engine = AuditedSectorClusterEngine(csv_filename=csv_file, lookback_years=1.2)
    
    # Pre-hydrate company configurations so ticker mappings are available
    macro_engine.load_mappings_from_csv()
    
    sector_report = macro_engine.discover_and_export_sectors(output_filename=SECTOR_REPORTS_PATH)

    if sector_report is None or not isinstance(sector_report, pd.DataFrame) or sector_report.empty:
        print("[CRITICAL ERROR] Macro Engine failed to return valid data matrices. Halting pipeline.")
        return

    # -------------------------------------------------------------------------
    # STAGE 2: Algorithmic Gateway Filtering Layer & Audit Export
    # -------------------------------------------------------------------------
    print("\n[STAGE 2] Running Algorithmic Gateway Routing Filters...")
    
    momentum_regimes = ["🔥 ULTRA_MOMENTUM_LEADERS", "🚀 ACTIVE_BREAKOUT_FIELDS"]
    high_velocity_sectors = sector_report[sector_report["Macro_Regime"].isin(momentum_regimes)]["Sector"].tolist()
    
    # The Safety Gateway Filter - Traps high-performing outliers stuck in lower clusters (like SERVICES)
    escaping_sectors = sector_report[
        (~sector_report["Macro_Regime"].isin(momentum_regimes)) & 
        (sector_report["Sector_Rolling_Return"] > 15.0)
    ]["Sector"].tolist()
    
    targeted_sectors = list(set(high_velocity_sectors + escaping_sectors))

    # Compile the Stage 2 Metadata Reason Text File Log
    with open(STOCK_ANALYSIS, "w", encoding="utf-8") as f:
        f.write("=" * 120 + "\n")
        f.write(" PRODUCTION QUANT DATA LOGS: STAGE 2 GATEWAY INDUSTRY ROUTING TRAIL\n")
        f.write(f" Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Target Universe: Nifty 500\n")
        f.write("=" * 120 + "\n\n")
        f.write(f"📈 SELECTED ACTIVE SECTOR ROUTING TRACKS:\n{targeted_sectors}\n\n")
        f.write("## ROUTING LOG METADATA DETAILED JUSTIFICATION:\n")
        f.write("-" * 120 + "\n")
        
        for _, row in sector_report.iterrows():
            sec = row["Sector"]
            if sec in targeted_sectors:
                status = "✅ PASSED FOR ANALYSIS"
                reason = "Target Momentum parameters achieved."
                if sec in escaping_sectors:
                    reason = "ALERT: High Return Alpha Outlier matched via Safety Filter Exception Rule."
            else:
                status = "❌ FILTERED / EXCLUDED"
                reason = "Underperforming performance metrics; velocity remains locked below minimum trend line."
                
            f.write(f"📍 SECTOR: {sec:<35} | STATUS: {status:<22}\n")
            f.write(f"   REASON: {reason}\n")
            f.write("-" * 80 + "\n")

    print(f" 🟢 Success: Gateway routing log saved locally to '{STOCK_ANALYSIS}'.")

    # Map industries back to tickers using your local CSV directory
    master_universe_map = macro_engine.sector_mapping
    actionable_tickers = [
        ticker for ticker, sector in master_universe_map.items()
        if sector in targeted_sectors
    ]

    if not actionable_tickers:
        print("[WARN] Zero tickers isolated from targeted sectors. Halting pipeline.")
        return

    # -------------------------------------------------------------------------
    # STAGE 3: Run Micro Technical Rules Firewall Pass over ALL Actionable Tickers
    # -------------------------------------------------------------------------
    print(f"\n[STAGE 3] Loading {len(actionable_tickers)} Targeted Tickers into Downstream Engine...")
    micro_pipeline = AuditableMomentumPipeline(symbols=actionable_tickers, lookback_years=1.2)
    
    raw_micro_df = micro_pipeline.fetch_universe_data()
    if raw_micro_df.empty:
        print("[WARN] Live technical streams returned zero records. Halting pipeline.")
        return

    gold_features_df = micro_pipeline.engineer_gold_features(raw_micro_df)
    
    # 🧼 Production Data-Hygiene Filter: Strip out low-priced assets (< Rs. 15) to prevent calculation distortion
    gold_features_df = gold_features_df[gold_features_df["Close"] >= 15.0]

    # Process rules engine calculations and write everything directly to file
    execution_signals = micro_pipeline.export_execution_signals(
        gold_df=gold_features_df, output_filename=EXECUTION_SIGNALS
    )

    if execution_signals is None or execution_signals.empty:
        print("[WARN] Micro technical rules engine returned zero active signals. Halting pipeline.")
        return

    # -------------------------------------------------------------------------
    # STAGE 4: Final Summary Report Console Printout & Data Mapping Patch
    # -------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print(" SYSTEM PRODUCTION PIPELINE EXECUTION SUCCESSFUL")
    print("=" * 110)
    print(f" 📂 Output 1: {SECTOR_REPORTS_PATH}      -> Stage 1 Unsupervised K-Means Models")
    print(f" 📂 Output 2: {STOCK_ANALYSIS}           -> Stage 2 Automated Selection Trails")
    print(f" 📂 Output 3: {EXECUTION_SIGNALS}        -> Stage 3 Micro Trading Firewall Matrices")
    print("=" * 110 + "\n")

    # 🟢 THE FIX: Dynamically map 'Sector' back to each symbol using the master map
    execution_signals["Sector"] = execution_signals["Symbol"].map(master_universe_map)

    # -------------------------------------------------------------------------
    # STAGE 5: Autonomous LLM Semantic Overlay (Batch Optimized)
    # -------------------------------------------------------------------------
    print("\n[STAGE 5] Querying Autonomous LLM Narrative Analyst (Batch Mode)...")
    llm_engine = GeminiSentimentEngine()
    
    # 1. Package all tickers into a clean, unified payload list for a single API call
    tickers_payload = []
    for idx, row in execution_signals.iterrows():
        tickers_payload.append({
            "symbol": row["Symbol"],
            "sector": row["Sector"],
            "close": row["Close"],
            "label": row["Strategic_Label"]
        })

    print(f" -> Enforcing LLM qualitative batch audit over {len(tickers_payload)} active target assets in a single call...")
    
    # 2. Dispatch the single batch request
    batch_analysis = llm_engine.analyze_batch_narratives(tickers_payload)
    
    # 3. Unpack the mapping back into lists to enrich our master DataFrame
    sentiments = []
    catalysts = []
    confidence_scores = []
    threats = []

    for row in tickers_payload:
        sym = row["symbol"]
        # Fallback parameters if the LLM misses a symbol or experiences parsing issues
        analysis = batch_analysis.get(sym, {})
        
        sentiments.append(analysis.get("sentiment", "NEUTRAL"))
        catalysts.append(analysis.get("news_catalyst", "No active catalyst logged."))
        confidence_scores.append(analysis.get("confidence_score", 60))
        threats.append(analysis.get("strategic_threat", "No structural risk identified."))

    # Enriching dataset with the structural LLM features
    execution_signals["Sentiment"] = sentiments
    execution_signals["News_Catalyst"] = catalysts
    execution_signals["Confidence_Score"] = confidence_scores
    execution_signals["Strategic_Threat"] = threats

    # -------------------------------------------------------------------------
    # STAGE 6: Programmatic HTML Generation & Live GitHub Actions Deployment
    # -------------------------------------------------------------------------
    print("\n[STAGE 5] Triggering Automated GitHub Deployment Pipelines...")
    
    #Retrieve system environmental variables safely
    GITHUB_PAT = os.getenv("GITHUB_PAT")
    REPO_OWNER = os.getenv("REPO_OWNER")
    REPO_NAME = "stock-sector-clustering"
    
    deployer_engine = ProgrammaticDashboardDeployer(
        github_token=GITHUB_PAT if GITHUB_PAT else "",
        repo_owner=REPO_OWNER if REPO_OWNER else "your_github_username",
        repo_name=REPO_NAME,
        branch="main"
    )

    # Compile dynamic DataFrame variables to HTML with the newly added 'Sector' data
    dashboard_html = deployer_engine.generate_html_string(execution_signals)

    if not GITHUB_PAT:
        print("[WARN] GITHUB_PAT environment token is missing. Skipping cloud hosting deployment.")
        # Save dashboard locally so you can still view it offline
        with open("index.html", "w", encoding="utf-8") as f:
            f.write(dashboard_html)
        print(" 🟢 Saved dashboard layout locally to 'index.html'. You can open this file in your browser!")
        return

    # Deploy to GitHub Pages
    deployer_engine.deploy_to_github(file_content=dashboard_html, destination_path="index.html")


if __name__ == "__main__":
    run_production_pipeline()