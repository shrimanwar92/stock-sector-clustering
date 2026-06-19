from datetime import date
import os
import datetime
from nselib import capital_market
import pandas as pd
from tqdm.contrib.concurrent import thread_map

NSE_DATASET_PATH = "dataset/nse_companies.csv"
LOOKBACK_YEARS = 2.0

TODAY = date.today().strftime("%d-%m-%Y")
REPORTS_DIR = f"reports/[{TODAY}]"
os.makedirs(REPORTS_DIR, exist_ok=True)

LLM_SENTIMENT_RESULT = f"reports/[{TODAY}]/llm_sentiment_results.json"
CACHE_FILE = f"reports/[{TODAY}]/.sector_cache.json"
MODEL_PATH = f"reports/[{TODAY}]/alpha_xgboost_scorer.json"
MODEL_SCHEMA_METADATA = f"reports/[{TODAY}]/model_training_metadata.json"

FEATURE_COLUMNS = [
    "Feature_RSI", "Feature_EMA_Dist", "Feature_Volume_Ratio", 
    "Feature_Delivery_Ratio", "Feature_ADX_14", "Feature_ATR_Ratio", 
    "Feature_ROC_20", "Feature_MACD_Hist_Accel", "Feature_Close_Strength",
    "Feature_Relative_Strength"
]

APPROVED_REGIMES = [
    "🔥 ULTRA_MOMENTUM_LEADERS", 
    "🚀 ACTIVE_BREAKOUT_FIELDS", 
    "📈 STABLE_UPWARD_ACCUMULATION"
]

def fetch_data_from_nse(filtered_symbols, symbol_to_sector_map):
    end_date = datetime.datetime.strptime(TODAY, "%d-%m-%Y").date()
    start_date = end_date - datetime.timedelta(days=int(365 * LOOKBACK_YEARS))

    filtered_symbols = [str(sym).split('.')[0].strip().upper() for sym in filtered_symbols]

    def fetch_single_symbol(symbol):
        df = capital_market.price_volume_and_deliverable_position_data(
            symbol=symbol, 
            from_date=start_date.strftime("%d-%m-%Y"), 
            to_date=end_date.strftime("%d-%m-%Y")
        )
        if df is not None and not df.empty:
            df = df.copy().reset_index(drop=True)
            df.columns = [str(col).replace('ï»¿', '').strip() for col in df.columns]
            if 'Series' in df.columns:
                df = df[df['Series'].str.strip() == 'EQ']
                if df.empty:
                    return None  # Drop non-equity tracking tickers early
                
            if "Symbol" in df.columns:
                df = df.drop(columns=["Symbol"])
            
            clean_symbol = str(symbol).strip().upper()
            df["Symbol"] = clean_symbol
            df["Sector"] = symbol_to_sector_map[clean_symbol]
        return df

    print(f"[START] Requesting historical streams across {len(filtered_symbols)} target tickers...")
    results = thread_map(
        fetch_single_symbol,
        filtered_symbols,
        max_workers=6,
        desc="Fetching NSE data"
    )

    all_data = [r for r in results if r is not None]

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()