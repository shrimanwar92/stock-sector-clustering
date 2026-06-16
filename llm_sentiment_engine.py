import os
import time
import json
import requests
from constants import (
    LLM_SENTIMENT_RESULT
)

class GeminiSentimentEngine:
    """
    Production-grade Semantic Analyzer leveraging Gemini Flash.
    Optimized for batch processing to eliminate throttling and network latency.
    Saves and stores the final returned sentiment analysis to a structured JSON file.
    """
    def __init__(self, output_json_path: str = "llm_sentiment_results.json"):
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.model_name = "gemini-2.5-flash-lite"
        self.endpoint_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        self.output_json_path = LLM_SENTIMENT_RESULT

    def analyze_batch_narratives(self, tickers: list) -> dict:
        """
        Processes a list of tickers in optimized chunks of 25 to remain within
        output token limits, executing a single batch API call per chunk.
        """
        if not self.api_key:
            print("[WARN] GEMINI_API_KEY environment variable is missing.")
            return {}

        if not tickers:
            print("[LLM] Received empty tickers payload. Skipping batch narrative analysis.")
            return {}

        cleaned_tickers = []
        for item in tickers:
            if isinstance(item, dict):
                symbol = item.get("Symbol") or item.get("symbol") or item.get("ticker")
                if symbol: cleaned_tickers.append(str(symbol))
            elif isinstance(item, str):
                cleaned_tickers.append(item)
            else:
                try:
                    if hasattr(item, "Symbol"): cleaned_tickers.append(str(item.Symbol))
                    else: cleaned_tickers.append(str(item))
                except Exception: pass

        cleaned_tickers = [t.strip() for t in cleaned_tickers if t]

        all_results = {}
        chunk_size = 25
        ticker_chunks = [cleaned_tickers[i:i + chunk_size] for i in range(0, len(cleaned_tickers), chunk_size)]

        print(f"[LLM] Dispatching {len(ticker_chunks)} parallelized batch requests to Gemini for {len(cleaned_tickers)} targets...")

        for chunk_idx, chunk in enumerate(ticker_chunks):
            print(f" -> Analyzing chunk {chunk_idx + 1}/{len(ticker_chunks)} (Size: {len(chunk)} tickers)...")
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
            "You are an expert Senior Quantitative Research Analyst specializing in top-tier asset selection. "
            "You must return your response as a strict JSON object containing an 'analyses' array. "
            "Every object inside 'analyses' MUST explicitly contain fields: 'symbol' (string), 'sentiment' (string), "
            "'news_catalyst' (string), 'confidence_score' (integer between 0 and 100), and 'strategic_threat' (string). "
            "Do not omit keys or return unstructured text."
        )
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"{system_prompt}\n\nAnalyze fundamental structural catalysts for the following tickers: {', '.join(chunk)}. Return strict schema format. If information is lean, pick a conservative confidence_score baseline like 50."
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
                        symbol_key = str(item.get("symbol", "")).strip()
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
                                "strategic_threat": str(item.get("strategic_threat", "No operational risk threats identified."))
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