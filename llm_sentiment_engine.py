import os
import time
import json
import requests
import pandas as pd  # Added explicit import for type checking
from constants import (
    LLM_SENTIMENT_RESULT, LLM_MODEL_NAME
)

class GeminiSentimentEngine:
    """
    Production-grade Semantic Analyzer leveraging Gemini Flash.
    Optimized for batch processing to eliminate throttling and network latency.
    Saves and stores the final returned sentiment analysis to a structured JSON file.
    
    UPGRADE: Engineered to accept and synthesize SHAP mathematical feature 
    attributions alongside fundamental catalysts for institutional-grade explainability.
    """
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.model_name = LLM_MODEL_NAME
        self.endpoint_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        self.output_json_path = LLM_SENTIMENT_RESULT

    def analyze_batch_narratives(self, tickers) -> dict:
        """
        Processes a list of tickers, dictionary payloads, or a pandas DataFrame containing 
        Ticker + SHAP insights in optimized chunks of 25, executing a single batch API call per chunk.
        """
        if not self.api_key:
            print("[WARN] GEMINI_API_KEY environment variable is missing.")
            return {}

        # 🛠️ FIX: Natively catch and normalize Pandas DataFrames before the boolean evaluation check
        if isinstance(tickers, pd.DataFrame):
            ticker_list = []
            for _, row in tickers.iterrows():
                # Cross-reference every potential pipeline key name for the SHAP payload
                insights = (
                    row.get("shap_drivers") or 
                    row.get("quantitative_drivers") or 
                    row.get("SHAP_Drivers") or 
                    row.get("Shap_Insights") or 
                    "No explicit SHAP attributions provided."
                )
                symbol_val = row.get("Symbol") if row.get("Symbol") is not None else row.get("symbol")
                
                ticker_list.append({
                    "Symbol": symbol_val,
                    "Shap_Insights": insights
                })
            tickers = ticker_list

        # ✅ Safe from Ambiguous DataFrame Boolean evaluations now that it is guaranteed to be a list
        if not tickers:
            print("[LLM] Received empty tickers payload. Skipping batch narrative analysis.")
            return {}

        # Parse and standardize input stream to handle raw strings or rich dict payloads
        processed_payloads = []
        for item in tickers:
            if isinstance(item, dict):
                symbol = item.get("Symbol") or item.get("symbol") or item.get("ticker")
                shap_insights = item.get("Shap_Insights") or item.get("shap_insights") or "No explicit SHAP attributions provided."
                if symbol:
                    processed_payloads.append({
                        "symbol": str(symbol).strip().upper(),
                        "shap_insights": str(shap_insights).strip()
                    })
            elif isinstance(item, str):
                processed_payloads.append({
                    "symbol": item.strip().upper(),
                    "shap_insights": "No explicit SHAP attributions provided."
                })
            else:
                try:
                    if hasattr(item, "Symbol"):
                        processed_payloads.append({
                            "symbol": str(item.Symbol).strip().upper(),
                            "shap_insights": getattr(item, "Shap_Insights", "No explicit SHAP attributions provided.")
                        })
                    else:
                        processed_payloads.append({
                            "symbol": str(item).strip().upper(),
                            "shap_insights": "No explicit SHAP attributions provided."
                        })
                except Exception:
                    pass

        all_results = {}
        chunk_size = 25
        ticker_chunks = [processed_payloads[i:i + chunk_size] for i in range(0, len(processed_payloads), chunk_size)]

        print(f"[LLM] Dispatching {len(ticker_chunks)} parallelized batch requests to Gemini for {len(processed_payloads)} targets...")

        for chunk_idx, chunk in enumerate(ticker_chunks):
            print(f" -> Analyzing chunk {chunk_idx + 1}/{len(ticker_chunks)} (Size: {len(chunk)} positions with SHAP diagnostics)...")
            chunk_results = self._process_single_chunk_with_retry(chunk)
            
            if chunk_results and isinstance(chunk_results, dict):
                all_results.update(chunk_results)

        self._write_results_to_json(all_results)
        return all_results

    def _write_results_to_json(self, results: dict):
        try:
            print(f"[LLM] Exporting compiled dynamic semantic results to JSON: '{self.output_json_path}'...")
            with open(self.output_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
            print("💾 [LLM SUCCESS] Semantic database saved to disk successfully.")
        except Exception as e:
            print(f"[WARN] Failed to write sentiment results to JSON file: {e}")

    def _process_single_chunk_with_retry(self, chunk: list) -> dict:
        system_prompt = (
            "You are an expert Senior Quantitative Research Analyst and Machine Learning Explainability Engineer specializing in asset selection. "
            "You are evaluating a list of assets selected by an underlying 3-class XGBoost classification model (0=Stop-Loss, 1=Stagnation, 2=Target Success). "
            "For each asset, you are provided with its symbol and its 'SHAP Mathematical Drivers' (the features carrying the highest feature attribution scores for this prediction). "
            "Your task is to cross-reference these mathematical feature drivers with current macroeconomic, fundamental, and market-structure catalysts.\n\n"
            "You must return your response as a strict JSON object containing an 'analyses' array.\n"
            "Every object inside 'analyses' MUST explicitly contain these exact keys:\n"
            " - 'symbol' (string)\n"
            " - 'sentiment' (string: MUST be BULLISH, BEARISH, or NEUTRAL)\n"
            " - 'news_catalyst' (string: summarizing current news, institutional flow, or earnings drivers)\n"
            " - 'confidence_score' (integer between 0 and 100 based on fundamental conviction alignment)\n"
            " - 'strategic_threat' (string: identifying structural risks, competition, or operational hazards)\n"
            " - 'shap_synthesis' (string: A brief technical synthesis explaining how the provided SHAP mathematical drivers align with or are validated by the real-world fundamental catalysts. Be specific about the features mentioned.)\n\n"
            "Do not omit keys, do not truncate explanations, and do not return unstructured text or markdown outside the valid JSON block."
        )
        
        target_data_lines = []
        for item in chunk:
            target_data_lines.append(
                f"Asset Symbol: {item['symbol']}\n"
                f"SHAP Mathematical Drivers: {item['shap_insights']}\n"
                f"---"
            )
        targets_payload_str = "\n".join(target_data_lines)
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": (
                        f"{system_prompt}\n\n"
                        f"Analyze the following targets and generate their quantitative-fundamental synthesis:\n\n"
                        f"{targets_payload_str}\n\n"
                        f"Return strict schema format. If information is lean, pick a conservative confidence_score baseline like 50."
                    )
                }]
            }]
        }

        for delay in [2, 4, 8]:
            try:
                response = requests.post(
                    f"{self.endpoint_url}?key={self.api_key}",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result_json = response.json()
                    text_content = result_json["candidates"][0]["content"]["parts"][0]["text"]
                    
                    if text_content.startswith("```json"):
                        text_content = text_content.split("```json")[1].split("```")[0]
                    elif text_content.startswith("```"):
                        text_content = text_content.split("```")[1].split("```")[0]
                        
                    parsed_response = json.loads(text_content.strip())
                    
                    results_map = {}
                    for item in parsed_response.get("analyses", []):
                        symbol_key = str(item.get("symbol", "")).strip().upper()
                        if symbol_key:
                            raw_score = item.get("confidence_score")
                            try:
                                clean_score = int(raw_score) if raw_score is not None else 50
                            except (ValueError, TypeError):
                                clean_score = 50

                            results_map[symbol_key] = {
                                "sentiment": str(item.get("sentiment", "NEUTRAL")).upper(),
                                "news_catalyst": str(item.get("news_catalyst", "No active fundamental catalyst logged.")),
                                "confidence_score": clean_score,
                                "strategic_threat": str(item.get("strategic_threat", "No operational risk threats identified.")),
                                "shap_synthesis": str(item.get("shap_synthesis", "No mathematical-fundamental alignment synthesis generated."))
                            }
                    return results_map
                
                print(f"[DEBUG] API failed with status {response.status_code}: {response.text}")
                if response.status_code in [429, 500, 503]:
                    time.sleep(delay)
                    continue
                break
                
            except Exception as e:
                print(f"[DEBUG] Caught Exception during request parsing: {type(e).__name__} - {str(e)}")
                time.sleep(delay)
                continue
        return {}