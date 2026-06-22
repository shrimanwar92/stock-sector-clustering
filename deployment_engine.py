import json
import datetime
import pandas as pd

class ProgrammaticDashboardDeployer:
    """
    Acts as the data engine backend. Processes trading configurations into clean,
    structured JSON data payloads and saves them directly to a local file.
    """
    def __init__(self):
        # Removed GitHub configuration parameters since we are writing locally
        pass

    def generate_and_save_data(self, execution_df: pd.DataFrame, target_path: str = "dataset/data.json"):
        """
        Extracts mathematical parameters from the DataFrame, formats a lightweight JSON structure,
        and writes it directly to the local filesystem.
        """
        if execution_df.empty:
            qualified_df = pd.DataFrame()
        else:
            # Clear out vetoed, overextended, and unconfirmed setups
            invalid_states = [
                "⏳ FILTERED_OUT", "❌ REGIME_VETO", "⏳ SECTOR_MISALIGNED", 
                "⏳ SECTOR_VETO", "🛑 OVEREXTENDED", "📉 DEEP FLUSH", 
                "⏳ COUNTER_TREND_MOMENTUM", "⏳ UNCONFIRMED CHOP"
            ]
            qualified_df = execution_df[~execution_df["Strategic_Label"].isin(invalid_states)].copy()
        
        records = []
        for _, row in qualified_df.iterrows():
            records.append({
                "symbol": str(row["Symbol"]),
                "label": str(row["Strategic_Label"]),
                "close": float(row["Close"]),
                "stopLoss": float(row.get("Stop_Loss", 0.0)),
                "profitTarget": float(row.get("Profit_Target", 0.0)),
                "rsi": float(row.get("Feature_RSI", 50.0)),
                "emaDist": float(row.get("Feature_EMA_Dist", 0.0)),
                "volRatio": float(row.get("Feature_Volume_Ratio", 1.0)),
                "macdAccel": float(row.get("Feature_MACD_Hist_Accel", 0.0)),
                "closeStrength": float(row.get("Feature_Close_Strength", 0.5)),
                "sector": str(row.get("Sector", "UNKNOWN")),
                "trendAligned": int(row.get("Feature_Trend_Aligned", 0)),
                "relativeStrength": float(row.get("Feature_Relative_Strength", 0.0)),
                "deliveryRatio": float(row.get("Feature_Delivery_Ratio", 1.0)),
                "atrRatio": float(row.get("Feature_ATR_Ratio", 1.0)),
                "sectorAligned": int(row.get("Feature_Sector_Aligned", 0)),
                "sentiment": str(row.get("Sentiment", "NEUTRAL")),
                "news_catalyst": str(row.get("News_Catalyst", "No active fundamental catalyst logged.")),
                "confidence": int(row.get("Confidence_Score", 50)),
                "threat": str(row.get("Strategic_Threat", "No operational risk threats identified.")),
                "shapSynthesis": str(row.get("SHAP_Synthesis") or row.get("shap_synthesis") or "No mathematical alignment synthesis generated.")
            })

        # Assemble unified dashboard data ecosystem object
        dashboard_payload = {
            "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tickers": records
        }
        
        # Write cleanly to the local file path
        try:
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(dashboard_payload, f, indent=4)
            print(f"✅ Telemetry data successfully compiled locally to: {target_path}")
        except Exception as e:
            print(f"❌ Failed to write file to disk: {e}")