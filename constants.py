from datetime import date
import os

TODAY = date.today().strftime("%d-%m-%Y")
os.makedirs(f"reports/[{TODAY}]", exist_ok=True)
SECTOR_REPORTS_PATH = f"reports/[{TODAY}]/sector_audit_report.txt"
STOCK_ANALYSIS = f"reports/[{TODAY}]/stock_analysis_report.txt"
EXECUTION_SIGNALS = f"reports/[{TODAY}]/execution_signals_report.txt"
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