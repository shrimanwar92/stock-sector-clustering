import json
import base64
import datetime
import requests
import pandas as pd

class ProgrammaticDashboardDeployer:
    """
    Generates an upgraded interactive HTML dashboard containing live 
    LLM sentiment scores, catalysts, macro trend compliance, alpha tracking, and delivery metrics,
    then commits it directly to GitHub Pages.
    """
    def __init__(self, github_token: str, repo_owner: str, repo_name: str, branch: str = "main"):
        self.github_token = github_token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.branch = branch
        self.api_base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents"

    def generate_html_string(self, execution_df: pd.DataFrame) -> str:
        """
        Converts the Pandas DataFrame to an interactive dashboard environment.
        Filters out clutter and misaligned components to lock focus onto active trade setups.
        """
        if execution_df.empty:
            qualified_df = pd.DataFrame()
        else:
            # 🟢 Clear out vetoed, overextended, and unconfirmed setups from the web UI
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
                "threat": str(row.get("Strategic_Threat", "No operational risk threats identified."))
            })

        json_tickers_data = json.dumps(records, indent=12)
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quant Swing Rotation & VCP Execution Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
        body {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
        .mono-font {{ font-family: 'JetBrains Mono', monospace; }}
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: #111827; }}
        ::-webkit-scrollbar-thumb {{ background: #374151; border-radius: 3px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: #4b5563; }}
    </style>
</head>
<body class="bg-[#0b0f19] text-gray-100 min-h-screen flex flex-col antialiased">

    <header class="border-b border-gray-800 bg-[#111827]/80 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex flex-wrap justify-between items-center gap-4">
        <div class="flex items-center gap-3">
            <div class="h-10 w-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center text-white shadow-lg shadow-emerald-500/10">
                <i class="fa-solid fa-layer-group text-xl"></i>
            </div>
            <div>
                <h1 class="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                    QUANT WAVE <span class="text-xs bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded font-mono">SWING INSTITUTIONAL</span>
                </h1>
                <p class="text-xs text-gray-400">Wilder-Smoothed Volatility Tracking | Top-Down Sector Alignment and Mathematical Risk Firewalls</p>
            </div>
        </div>
        
        <div class="flex items-center gap-4">
            <div class="bg-[#1f2937] border border-gray-800 px-3 py-1.5 rounded-lg flex items-center gap-2 text-xs text-gray-300">
                <span class="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span>Workspace: VCP Production Engine</span>
            </div>
            <div class="text-xs text-gray-400 text-right hidden sm:block">
                <p class="font-semibold text-gray-200">System State Published</p>
                <p class="font-mono">{timestamp_str}</p>
            </div>
        </div>
    </header>

    <main class="flex-1 p-6 max-w-7xl w-full mx-auto grid grid-cols-1 lg:grid-cols-12 gap-6">
        <section class="lg:col-span-12 flex flex-col gap-6">
            <div class="bg-[#111827] border border-gray-800 rounded-2xl p-5 shadow-xl">
                <div id="ticker-rows" class="grid gap-4"></div>
            </div>
        </section>
    </main>

    <script>
        const initialTickers = {json_tickers_data};
        console.log("Ingested qualified strategic tickers:", initialTickers);
        
        function renderRows() {{
            const container = document.getElementById("ticker-rows");
            if (!container) return;
            if (initialTickers.length === 0) {{
                container.innerHTML = `<div class="p-8 text-center text-gray-500 font-mono text-sm">No actionable setups verified within the active technical market framework.</div>`;
                return;
            }}
            container.innerHTML = initialTickers.map(t => `
                <div class="p-5 bg-gray-900/50 rounded-xl border border-gray-800 flex flex-wrap justify-between items-center gap-4 hover:border-gray-700 transition">
                    <div>
                        <div class="flex items-center gap-2">
                            <h4 class="font-bold text-white font-mono text-base">${{t.symbol}}</h4>
                            <span class="text-[10px] px-2 py-0.5 rounded font-bold bg-blue-900/30 text-blue-400 border border-blue-500/20 font-mono">${{t.sector}}</span>
                            <span class="text-[10px] px-2 py-0.5 rounded font-bold ${{t.label.includes('BREAKOUT') ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-amber-500/10 text-amber-400 border border-amber-500/20' }}">${{t.label}}</span>
                        </div>
                        <p class="text-xs text-gray-400 mt-1">
                            Close: <span class="text-gray-200 font-mono">₹${{t.close}}</span> | 
                            Stop-Loss (2-ATR): <span class="text-red-400 font-mono">₹${{t.stopLoss}}</span> | 
                            Target (1:2 RR): <span class="text-emerald-400 font-mono">₹${{t.profitTarget}}</span>
                        </p>
                        <p class="text-xs text-gray-500 mt-2 font-mono">Wilder RSI: ${{t.rsi.toFixed(1)}} | 3/20 Vol Ratio: ${{t.volRatio.toFixed(2)}} | ATR Ratio: ${{t.atrRatio.toFixed(2)}} | Close Location: ${{t.closeStrength.toFixed(2)}}</p>
                    </div>
                    <div class="text-right max-w-md">
                        <span class="text-xs px-2 py-1 rounded font-mono ${{t.sentiment === 'BULLISH' ? 'bg-emerald-500/20 text-emerald-300' : 'bg-gray-800 text-gray-400'}}">${{t.sentiment}}</span>
                        <p class="text-xs text-gray-300 mt-1 italic">"${{t.news_catalyst}}"</p>
                        <p class="text-[10px] text-gray-500 mt-1">Confidence Layer Score: ${{t.confidence}}%</p>
                    </div>
                </div>
            `).join("");
        }}
        window.onload = renderRows;
    </script>
</body>
</html>
"""
        return html_template

    def deploy_to_github(self, file_content: str, destination_path: str = "index.html"):
        url = f"{self.api_base_url}/{destination_path}"
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }

        sha = None
        print(f"[GITHUB] Checking remote state for destination: '{destination_path}'...")
        r_get = requests.get(url, headers=headers, params={"ref": self.branch})
        
        if r_get.status_code == 200:
            sha = r_get.json().get("sha")
            print(f" -> Current remote file SHA hash resolved: {sha}")
        elif r_get.status_code == 404:
            print(" -> Target path is clean. Deploying fresh index artifact.")
        else:
            print(f"[ERR] GitHub REST validation failed. Status Code: {r_get.status_code}")
            return

        encoded_content = base64.b64encode(file_content.encode("utf-8")).decode("utf-8")

        payload = {
            "message": f"🤖 Dynamic Phase 3 VCP Dashboard Deployment [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}]",
            "content": encoded_content,
            "branch": self.branch
        }
        if sha:
            payload["sha"] = sha

        print("[GITHUB] Compiling and publishing package payload...")
        r_put = requests.put(url, headers=headers, json=payload)
        
        if r_put.status_code in [200, 201]:
            print("✅ [GITHUB PAGES SUCCESS] Dashboard with active LLM overlay deployed successfully!")
            print(f"📍 Live Link: https://{self.repo_owner}.github.io/{self.repo_name}/")
        else:
            print(f"[ERR] Update failed with code: {r_put.status_code} details: {r_put.text}")