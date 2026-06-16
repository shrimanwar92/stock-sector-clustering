from datetime import date
import os

TODAY = date.today().strftime("%d-%m-%Y")
os.makedirs(f"reports/[{TODAY}]", exist_ok=True)
SECTOR_REPORTS_PATH = f"reports/[{TODAY}]/sector_audit_report.txt"
STOCK_ANALYSIS = f"reports/[{TODAY}]/stock_analysis_report.txt"
EXECUTION_SIGNALS = f"reports/[{TODAY}]/execution_signals_report.txt"
LLM_SENTIMENT_RESULT = f"reports/[{TODAY}]/llm_sentiment_results.json"
CACHE_FILE = f"reports/[{TODAY}]/.sector_cache.json"