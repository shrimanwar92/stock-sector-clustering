import os
import time
import json
import requests

class GeminiSentimentEngine:
    """
    Production-grade Semantic Analyzer leveraging Gemini Flash.
    Optimized for batch processing to eliminate throttling and network latency.
    Saves and stores the final returned sentiment analysis to a structured JSON file.
    """
    def __init__(self, output_json_path: str = "llm_sentiment_results.json"):
        # Retrieve the API key from runtime environment variables
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        # Updated to a supported stable model
        self.model_name = "gemini-2.5-flash"
        self.endpoint_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        self.output_json_path = output_json_path

    def analyze_batch_narratives(self, tickers: list) -> dict:
        """
        Processes a list of tickers in optimized chunks of 25 to remain within
        output token limits, executing a single batch API call per chunk.
        Saves the aggregated results to a local JSON file before returning.
        """
        if not self.api_key:
            print("[WARN] GEMINI_API_KEY environment variable is missing.")
            return False

        all_results = {}
        chunk_size = 25
        ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]

        print(f"[LLM] Dispatching {len(ticker_chunks)} parallelized batch requests to Gemini...")

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
            "You are a Senior Quantitative Research Analyst specializing in the Indian stock market (NSE). "
            "You must return your response as a strict JSON object matching the requested schema."
        )

        user_query = (
            "Analyze the following batch of stocks. Map each symbol back to the response schema:\n"
            f"{json.dumps(chunk, indent=2)}"
        )

        payload = {
            "contents": [{"parts": [{"text": f"{system_prompt}\n\nUser Request: {user_query}"}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "analyses": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "symbol": {"type": "STRING"},
                                    "sentiment": {"type": "STRING", "enum": ["BULLISH", "NEUTRAL", "BEARISH"]},
                                    "news_catalyst": {"type": "STRING"},
                                    "confidence_score": {"type": "INTEGER"},
                                    "strategic_threat": {"type": "STRING"}
                                },
                                "required": ["symbol", "sentiment", "news_catalyst", "confidence_score", "strategic_threat"]
                            }
                        }
                    },
                    "required": ["analyses"]
                }
            }
        }

        delays = [1, 2, 4, 8, 16]
        
        for attempt, delay in enumerate(delays):
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
                    parsed_response = json.loads(text_content.strip())
                    
                    results_map = {}
                    for item in parsed_response.get("analyses", []):
                        results_map[item["symbol"]] = {
                            "sentiment": item["sentiment"],
                            "news_catalyst": item["news_catalyst"],
                            "confidence_score": item["confidence_score"],
                            "strategic_threat": item["strategic_threat"]
                        }
                    return results_map
                
                print(f"[DEBUG] API failed with status {response.status_code}: {response.text}")
                
                if response.status_code in [429, 500, 503]:
                    time.sleep(delay)
                    continue
                break
                
            except Exception as e:
                print(f"[DEBUG] Caught Exception during request: {type(e).__name__} - {str(e)}")
                time.sleep(delay)
                continue