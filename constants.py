import os
import datetime
from datetime import timezone, timedelta, datetime
from nselib import capital_market
import pandas as pd
from tqdm.contrib.concurrent import thread_map

NSE_DATASET_PATH = "dataset/nse_companies.csv"
LOOKBACK_YEARS = 2.0

IST = timezone(timedelta(hours=5, minutes=30))
TODAY = datetime.now(IST).strftime("%d-%m-%Y")
REPORTS_DIR = f"reports/[{TODAY}]"
os.makedirs(REPORTS_DIR, exist_ok=True)

LLM_SENTIMENT_RESULT = f"reports/[{TODAY}]/llm_sentiment_results.json"
LLM_MODEL_NAME = "gemini-2.5-flash-lite"
CACHE_FILE = f"reports/[{TODAY}]/.sector_cache.json"
MODEL_PATH = f"reports/[{TODAY}]/alpha_xgboost_scorer.json"
MODEL_SCHEMA_METADATA = f"reports/[{TODAY}]/model_training_metadata.json"

FEATURE_COLUMNS = [
    "Feature_RSI",
    "Feature_EMA_Dist",
    "Feature_Volume_Ratio",
    "Feature_ADX_14",
    "Feature_ATR_Ratio",
    "Feature_ROC_20",
    "Feature_MACD_Hist_Accel",
    "Feature_Relative_Strength"
]

APPROVED_REGIMES = [
    "🔥 LEADING_MOMENTUM_ACCELERATION"
    # "📈 NEUTRAL_SIDEWAYS_CONSOLIDATION"  # <-- Include this if you want to allow high-scoring sideways assets
]

SECTOR_REGIMES = {
    "STRONG": "🔥 LEADING_MOMENTUM_ACCELERATION",
    "NEUTRAL": "📈 NEUTRAL_SIDEWAYS_CONSOLIDATION",
    "WEAK": "❄️ DEEP_BEARISH_CAPITULATION"
}

import os
import gzip
import datetime
import pandas as pd
from tqdm.contrib.concurrent import thread_map
from nselib import capital_market


def fetch_data_from_nse(filtered_symbols, symbol_to_sector_map):
    end_date = datetime.datetime.strptime(TODAY, "%d-%m-%Y")
    start_date = end_date - datetime.timedelta(days=int(365 * LOOKBACK_YEARS))

    # ------------------------------------------------------------------
    # NORMALIZE SYMBOLS
    # ------------------------------------------------------------------
    filtered_symbols = [
        str(sym).split(".")[0].strip().upper()
        for sym in filtered_symbols
    ]
    
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

            df.columns = [
                str(col).replace("ï»¿", "").strip()
                for col in df.columns
            ]

            # keep only EQ series
            if "Series" in df.columns:
                df = df[df["Series"].astype(str).str.strip() == "EQ"]

                if df.empty:
                    return None

            if "Symbol" in df.columns:
                df = df.drop(columns=["Symbol"])

            clean_symbol = str(symbol).strip().upper()

            df["Symbol"] = clean_symbol
            df["Sector"] = symbol_to_sector_map.get(
                clean_symbol,
                "UNKNOWN"
            )

            return df

        except Exception:
            return None

    print(
        f"[START] Requesting historical streams across "
        f"{len(filtered_symbols)} target tickers..."
    )

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

    try:
        with gzip.open(cache_file, "wt", encoding="utf-8") as f:
            final_df.to_json(
                f,
                orient="records",
                date_format="iso"
            )

        print(
            f"💾 [CACHE WRITE] Successfully stored today's "
            f"raw sector universe data."
        )

    except Exception as e:
        print(f"[WARN] Cache write failed: {e}")

    return final_df