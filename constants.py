import os
import gzip
from datetime import timezone, timedelta, datetime
import pandas as pd
from tqdm.contrib.concurrent import thread_map
from nselib import capital_market

NSE_DATASET_PATH = "dataset/nse_companies.csv"
LOOKBACK_YEARS = 2.0

IST = timezone(timedelta(hours=5, minutes=30))
TODAY = datetime.now(IST).strftime("%d-%m-%Y")
REPORTS_DIR = f"reports/[{TODAY}]"
os.makedirs(REPORTS_DIR, exist_ok=True)

LLM_SENTIMENT_RESULT = f"reports/[{TODAY}]/llm_sentiment_results.json"
LLM_MODEL_NAME = "gemini-2.5-flash-lite"

# Unified Compressed Raw Data Cache
RAW_MARKET_CACHE = f"reports/[{TODAY}]/.raw_market_cache.json.gz"

# Model training artifacts
MODEL_PATH = "reports/alpha_xgboost_scorer.json"
CALIBRATOR_MODEL = "reports/alpha_calibrator.joblib"
MODEL_TRAINING_METADATA = "reports/model_training.json"
MODEL_HEALTH_METADATA = "reports/model_health.json"

FEATURE_COLUMNS = [
    # Original Technicals
    "Feature_RSI",
    "Feature_EMA_Dist",
    "Feature_Volume_Ratio",
    "Feature_ADX_14",
    "Feature_ATR_Ratio",
    "Feature_MACD_Hist_Accel",
    "Feature_Relative_Strength",
    
    # Raw Metrics (Absolute Context)
    "Feature_ROC_20",
    "Feature_Trend_Age",
    "Feature_Vol_Expansion",
    "Feature_Dist_To_200DMA",
    "Feature_Bollinger_Width",
    "Feature_Rel_ROC_20",
    "Feature_Rel_Trend_Age",
    
    # --- ADD THESE PERCENTILE RANKS (The "Quant-Grade" features) ---
    "Feature_ROC_20_PctRank",
    "Feature_Trend_Age_PctRank",
    "Feature_Bollinger_Width_PctRank"
]

APPROVED_REGIMES = [
    "🔥 LEADING_MOMENTUM_ACCELERATION"
]

SECTOR_REGIMES = {
    "STRONG": "🔥 LEADING_MOMENTUM_ACCELERATION",
    "NEUTRAL": "📈 NEUTRAL_SIDEWAYS_CONSOLIDATION",
    "WEAK": "❄️ DEEP_BEARISH_CAPITULATION"
}


def fetch_data_from_nse(filtered_symbols, symbol_to_sector_map):
    """
    Unified central data ingestion engine. 
    Loads from a single compressed day-cache if available, otherwise executes high-performance exchange pulls.
    """
    end_date = datetime.strptime(TODAY, "%d-%m-%Y")
    start_date = end_date - timedelta(days=int(365 * LOOKBACK_YEARS))

    # Normalize incoming search symbols
    filtered_symbols = list(set([
        str(sym).split(".")[0].strip().upper()
        for sym in filtered_symbols
    ]))

    # 1. Check Unified Day Cache
    if os.path.exists(RAW_MARKET_CACHE):
        try:
            print(f"💾 [CACHE HIT] Hydrating data from unified space cache: '{RAW_MARKET_CACHE}'")
            with gzip.open(RAW_MARKET_CACHE, "rt", encoding="utf-8") as f:
                cached_df = pd.read_json(f, orient="records")
            
            if not cached_df.empty:
                cached_df.columns = [str(col).replace("ï»¿", "").strip() for col in cached_df.columns]
                cached_df["Symbol"] = cached_df["Symbol"].astype(str).str.strip().str.upper()
                
                # Cross-sectional filter for requested symbols
                matched_df = cached_df[cached_df["Symbol"].isin(filtered_symbols)].reset_index(drop=True)
                if not matched_df.empty:
                    print(f"✅ Successfully filtered {matched_df['Symbol'].nunique()} symbols ({len(matched_df)} rows) from cache.")
                    return matched_df
        except Exception as e:
            print(f"[WARN] Cache hydration collision: {e}. Falling back to live network streams...")

    # 2. Live Exchange Fetch Pathway
    def fetch_single_symbol(symbol):
        try:
            df = capital_market.price_volume_and_deliverable_position_data(
                symbol=symbol,
                from_date=start_date.strftime("%d-%m-%Y"),
                to_date=end_date.strftime("%d-%m-%Y")
            )

            if df is None or df.empty:
                return None

            df = df.copy().reset_index(drop=True)
            df.columns = [str(col).replace("ï»¿", "").strip() for col in df.columns]

            if "Series" in df.columns:
                df = df[df["Series"].astype(str).str.strip() == "EQ"]
                if df.empty:
                    return None

            if "Symbol" in df.columns:
                df = df.drop(columns=["Symbol"])

            clean_symbol = str(symbol).strip().upper()
            df["Symbol"] = clean_symbol
            df["Sector"] = symbol_to_sector_map.get(clean_symbol, "UNKNOWN")
            return df
        except Exception:
            return None

    print(f"[START] Requesting historical streams across {len(filtered_symbols)} target tickers from exchange...")
    results = thread_map(
        fetch_single_symbol,
        filtered_symbols,
        max_workers=6,
        desc="Fetching NSE data"
    )

    all_data = [r for r in results if r is not None]
    if not all_data:
        return pd.DataFrame()

    final_df = pd.concat(all_data, ignore_index=True)

    # 3. Commit to Unified Day Cache
    try:
        os.makedirs(os.path.dirname(RAW_MARKET_CACHE), exist_ok=True)
        with gzip.open(RAW_MARKET_CACHE, "wt", encoding="utf-8") as f:
            final_df.to_json(f, orient="records", date_format="iso")
        print(f"💾 [CACHE WRITE] Successfully committed active universe stream array to unified cache file.")
    except Exception as e:
        print(f"[WARN] Cache commit execution failure: {e}")

    return final_df