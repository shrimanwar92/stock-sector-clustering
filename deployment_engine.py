import json
import datetime
import pandas as pd
import zoneinfo, datetime

class ProgrammaticDashboardDeployer:
    """
    Acts as the quantitative data serialization engine. Compiles model matrix outputs, 
    risk-reward analytics, and 3-class triple barrier metrics directly to a local JSON payload.
    """
    def __init__(self):
        pass

    def generate_and_save_data(self, execution_df: pd.DataFrame, target_path: str = "dataset/data.json"):
        """
        Extracts structural execution features from the incoming ML DataFrame 
        and updates mapping bindings to precisely track the revised presentations layer.
        """
        if execution_df.empty:
            qualified_df = pd.DataFrame()
        else:
            # Drop structural configuration failures that should not hit downstream UI components
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
                "expectedValue": float(row.get("Expected_Value", 0.0)),
                "rewardRisk": float(row.get("Reward_Risk", 0.0)),
                "pSuccess": float(row.get("Alpha_ML_Score", 0.0)),
                "pFailure": float(row.get("Prob_Failure_SL", 0.0)),
                "pStagnate": float(row.get("Prob_Stagnation", 0.0)),
                "confidence": float(row.get("Confidence_Score", 0.0)),
                "decisionReason": str(row.get("Decision_Reason", "No dynamic reasons generated.")),
                "rsi": float(row.get("Feature_RSI", 50.0)),
                "volRatio": float(row.get("Feature_Volume_Ratio", 1.0)),
                "atrRatio": float(row.get("Feature_ATR_Ratio", 1.0)),
                "closeStrength": float(row.get("Feature_Close_Strength", 0.5)),
                "sector": str(row.get("Sector", "UNKNOWN")),
                "sentiment": str(row.get("Sentiment", "NEUTRAL")),
                "news_catalyst": str(row.get("News_Catalyst", "No active fundamental catalyst logged.")),
                "threat": str(row.get("Strategic_Threat", "No operational risk threats identified.")),
                "shapSynthesis": str(row.get("SHAP_Synthesis") or row.get("shap_synthesis") or "No mathematical alignment synthesis generated.")
            })

        # Assemble deployment schema
        dashboard_payload = {
            "updated_at": datetime.datetime.now(zoneinfo.ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M"),
            "tickers": records
        }
        
        try:
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(dashboard_payload, f, indent=4)
            print(f"✅ Telemetry data successfully compiled locally to: {target_path}")
        except Exception as e:
            print(f"❌ Failed to write file to disk: {e}")