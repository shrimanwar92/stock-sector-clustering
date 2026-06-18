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
        ::-webkit-scrollbar-track {{ background: #f3f4f6; }}
        ::-webkit-scrollbar-thumb {{ background: #d1d5db; border-radius: 3px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: #9ca3af; }}
    </style>
</head>
<body class="bg-[#f8fafc] text-[#1e293b] min-h-screen flex flex-col antialiased">

    <header class="border-b border-[#e2e8f0] bg-white/90 backdrop-blur-md sticky top-0 z-50 px-4 sm:px-6 py-4 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 shadow-sm">
        <div class="flex items-center gap-3">
            <div class="h-10 w-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-500 flex items-center justify-center text-white shadow-md shadow-emerald-500/10">
                <i class="fa-solid fa-layer-group text-xl"></i>
            </div>
            <div>
                <h1 class="text-base sm:text-lg font-bold tracking-tight text-[#0f172a] flex flex-wrap items-center gap-2">
                    QUANT WAVE <span class="text-[10px] bg-emerald-50 text-emerald-700 border border-emerald-200 px-2 py-0.5 rounded font-mono font-bold">SWING TRADE ANALYSIS</span>
                </h1>
                <p class="text-xs text-[#64748b] mt-0.5 hidden md:block">Swing trading analysis 2-12 weeks timeframe</p>
            </div>
        </div>
        
        <div class="flex items-center justify-between sm:justify-end w-full sm:w-auto gap-4 border-t sm:border-t-0 pt-3 sm:pt-0 border-[#e2e8f0]">
            <div class="text-xs text-[#64748b] text-right">
                <p class="font-semibold text-[#334155]">Data Published on</p>
                <p class="font-mono mt-0.5 text-[11px]">{timestamp_str}</p>
            </div>
        </div>
    </header>

    <main class="flex-1 p-4 sm:p-6 max-w-7xl w-full mx-auto space-y-6">
        
        <section class="bg-white border border-[#e2e8f0] rounded-2xl p-5 shadow-sm">
            <h3 class="text-xs font-bold text-[#475569] uppercase tracking-wider mb-3 flex items-center gap-2">
                <i class="fa-solid fa-circle-info text-emerald-600 text-sm"></i> Strategic Label Explanatory Guide
            </h3>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="p-3.5 bg-emerald-50/50 border border-emerald-100 rounded-xl">
                    <div class="flex items-center gap-2 mb-1.5">
                        <span class="text-xs font-bold bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded font-mono">🚀 ACTIVE BREAKOUT</span>
                    </div>
                    <p class="text-xs text-[#475569] leading-relaxed"><strong class="text-[#0f172a]">Plain Meaning:</strong> The stock is showing powerful upward trend momentum right now and has broken out of its short-term price ceiling.</p>
                    <p class="text-[11px] font-mono text-[#64748b] mt-2 bg-white/80 p-1.5 rounded border border-emerald-200/60"><strong>Quant Logic:</strong> Vol Ratio &ge; 1.25, Trend Aligned (20 > 50 > 200 EMA), ROC_20 > 2.0, & MACD Accel > 0.</p>
                </div>
                
                <div class="p-3.5 bg-cyan-50/40 border border-cyan-100 rounded-xl">
                    <div class="flex items-center gap-2 mb-1.5">
                        <span class="text-xs font-bold bg-cyan-50 text-cyan-800 border border-cyan-200 px-2 py-0.5 rounded font-mono">🚀 INSIDER BREAKOUT</span>
                    </div>
                    <p class="text-xs text-[#475569] leading-relaxed"><strong class="text-[#0f172a]">Plain Meaning:</strong> A breakout backed by high delivery conversion, indicating institutions are absorbing outstanding float supply.</p>
                    <p class="text-[11px] font-mono text-[#64748b] mt-2 bg-white/80 p-1.5 rounded border border-cyan-200/60"><strong>Quant Logic:</strong> Active Breakout parameters met + Delivery Ratio &ge; 1.15 OR Intraday Close Location &ge; 0.65.</p>
                </div>

                <div class="p-3.5 bg-amber-50/50 border border-amber-100 rounded-xl">
                    <div class="flex items-center gap-2 mb-1.5">
                        <span class="text-xs font-bold bg-amber-100 text-amber-800 px-2 py-0.5 rounded font-mono">🏢 LAUNCHPAD / INST.</span>
                    </div>
                    <p class="text-xs text-[#475569] leading-relaxed"><strong class="text-[#0f172a]">Plain Meaning:</strong> The stock is tightly consolidating inside a quiet compression base, hiding coiled springs before a macro expansion.</p>
                    <p class="text-[11px] font-mono text-[#64748b] mt-2 bg-white/80 p-1.5 rounded border border-amber-200/60"><strong>Quant Logic:</strong> Close within &plusmn;4.5% of 20 EMA, Trend Aligned, Volatility contracting (ATR Ratio < 0.98).</p>
                </div>
            </div>
        </section>

        <section class="flex flex-wrap items-center gap-2 bg-white border border-[#e2e8f0] p-3 rounded-xl shadow-sm">
            <span class="text-xs font-bold uppercase tracking-wider text-[#64748b] ml-1 mr-2"><i class="fa-solid fa-filter text-slate-500 mr-1.5"></i> Filter Setups:</span>
            <button onclick="filterTable('ALL')" id="btn-ALL" class="text-xs font-mono font-bold px-3 py-1.5 rounded-lg border transition-all duration-200 bg-[#0f172a] text-white border-[#0f172a]">
                All Verified Setups (<span id="count-ALL">0</span>)
            </button>
            <button onclick="filterTable('ACTIVE BREAKOUT')" id="btn-ACTIVE" class="text-xs font-mono font-bold px-3 py-1.5 rounded-lg border transition-all duration-200 bg-white text-[#475569] border-[#e2e8f0] hover:bg-slate-50">
                🚀 Active Breakout (<span id="count-ACTIVE">0</span>)
            </button>
            <button onclick="filterTable('INSIDER BREAKOUT')" id="btn-INSIDER" class="text-xs font-mono font-bold px-3 py-1.5 rounded-lg border transition-all duration-200 bg-white text-[#475569] border-[#e2e8f0] hover:bg-slate-50">
                💎 Insider Breakout (<span id="count-INSIDER">0</span>)
            </button>
            <button onclick="filterTable('LAUNCHPAD')" id="btn-LAUNCHPAD" class="text-xs font-mono font-bold px-3 py-1.5 rounded-lg border transition-all duration-200 bg-white text-[#475569] border-[#e2e8f0] hover:bg-slate-50">
                🏢 Launchpad (<span id="count-LAUNCHPAD">0</span>)
            </button>
            <button onclick="filterTable('INSTITUTIONAL LAUNCHPAD')" id="btn-INST-LAUNCHPAD" class="text-xs font-mono font-bold px-3 py-1.5 rounded-lg border transition-all duration-200 bg-white text-[#475569] border-[#e2e8f0] hover:bg-slate-50">
                🏛️ Institutional Launchpad (<span id="count-INST-LAUNCHPAD">0</span>)
            </button>
        </section>

        <section class="w-full">
            <div class="bg-white border border-[#e2e8f0] rounded-2xl p-4 sm:p-6 shadow-sm">
                <div id="ticker-rows" class="grid gap-4"></div>
            </div>
        </section>
    </main>

    <script>
        const initialTickers = {json_tickers_data};
        let currentFilter = 'ALL';
        console.log("Ingested qualified strategic tickers:", initialTickers);
        
        function updateFilterBadges() {{
            const counts = {{
                ALL: initialTickers.length,
                ACTIVE: initialTickers.filter(t => t.label === '🚀 ACTIVE BREAKOUT').length,
                INSIDER: initialTickers.filter(t => t.label === '🚀 INSIDER BREAKOUT').length,
                LAUNCHPAD: initialTickers.filter(t => t.label === '🏢 LAUNCHPAD').length,
                INST_LAUNCHPAD: initialTickers.filter(t => t.label === '🏢 INSTITUTIONAL LAUNCHPAD').length
            }};

            document.getElementById("count-ALL").innerText = counts.ALL;
            document.getElementById("count-ACTIVE").innerText = counts.ACTIVE;
            document.getElementById("count-INSIDER").innerText = counts.INSIDER;
            document.getElementById("count-LAUNCHPAD").innerText = counts.LAUNCHPAD;
            document.getElementById("count-INST-LAUNCHPAD").innerText = counts.INST_LAUNCHPAD;

            // Manage dynamic highlighting css toggles
            const buttons = {{
                'ALL': 'btn-ALL',
                'ACTIVE BREAKOUT': 'btn-ACTIVE',
                'INSIDER BREAKOUT': 'btn-INSIDER',
                'LAUNCHPAD': 'btn-LAUNCHPAD',
                'INSTITUTIONAL LAUNCHPAD': 'btn-INST-LAUNCHPAD'
            }};

            Object.keys(buttons).forEach(key => {{
                const btn = document.getElementById(buttons[key]);
                if (!btn) return;
                if (key === currentFilter) {{
                    btn.className = "text-xs font-mono font-bold px-3 py-1.5 rounded-lg border transition-all duration-200 bg-[#0f172a] text-white border-[#0f172a]";
                }} else {{
                    btn.className = "text-xs font-mono font-bold px-3 py-1.5 rounded-lg border transition-all duration-200 bg-white text-[#475569] border-[#e2e8f0] hover:bg-slate-50";
                }}
            }});
        }}

        function filterTable(labelKey) {{
            currentFilter = labelKey;
            renderRows();
        }}

        function renderRows() {{
            const container = document.getElementById("ticker-rows");
            if (!container) return;
            
            // Apply filtering layer array manipulation
            const filteredData = currentFilter === 'ALL' 
                ? initialTickers 
                : initialTickers.filter(t => t.label === t.label.includes('BREAKOUT') ? t.label : t.label.toUpperCase().includes(currentFilter) || t.label === currentFilter);
            
            updateFilterBadges();

            if (filteredData.length === 0) {{
                container.innerHTML = `<div class="p-12 text-center text-[#64748b] font-mono text-sm border-2 border-dashed border-[#e2e8f0] rounded-xl bg-[#f8fafc]">No active trading candidates matched the selected filter configuration.</div>`;
                return;
            }}
            
            container.innerHTML = filteredData.map(t => `
                <div class="p-5 bg-white rounded-xl border border-[#e2e8f0] flex flex-col lg:flex-row justify-between items-start lg:items-center gap-5 hover:border-slate-400 hover:shadow-md transition-all duration-200">
                    <div class="w-full lg:max-w-xl">
                        <div class="flex flex-wrap items-center gap-2">
                            <h4 class="font-bold text-[#0f172a] font-mono text-base tracking-tight">${{t.symbol}}</h4>
                            <span class="text-[10px] px-2 py-0.5 rounded font-bold bg-[#f1f5f9] text-[#475569] border border-[#e2e8f0] font-mono">${{t.sector}}</span>
                            <span class="text-[10px] px-2 py-0.5 rounded font-bold ${{t.label.includes('BREAKOUT') ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-amber-50 text-amber-700 border border-amber-200' }}">${{t.label}}</span>
                        </div>
                        <div class="text-xs text-[#334155] mt-2.5 bg-[#f8fafc] p-3 rounded-lg border border-[#f1f5f9] flex flex-wrap gap-x-4 gap-y-1.5 font-medium">
                            <div>Close: <span class="text-[#0f172a] font-mono font-bold">₹${{t.close.toLocaleString('en-IN')}}</span></div>
                            <div class="border-l border-[#e2e8f0] pl-4">Stop-Loss: <span class="text-rose-600 font-mono font-bold">₹${{t.stopLoss.toLocaleString('en-IN')}}</span></div>
                            <div class="border-l border-[#e2e8f0] pl-4">Target: <span class="text-emerald-600 font-mono font-bold">₹${{t.profitTarget.toLocaleString('en-IN')}}</span></div>
                        </div>
                        <div class="text-[11px] text-[#64748b] mt-3 font-mono flex flex-wrap gap-x-3 gap-y-1 bg-[#f1f5f9]/40 px-2.5 py-1.5 rounded border border-dashed border-[#e2e8f0]">
                            <span>RSI: <strong class="text-[#334155]">${{t.rsi.toFixed(1)}}</strong></span>
                            <span>•</span>
                            <span>Vol Ratio: <strong class="text-[#334155]">${{t.volRatio.toFixed(2)}}</strong></span>
                            <span>•</span>
                            <span>ATR Ratio: <strong class="text-[#334155]">${{t.atrRatio.toFixed(2)}}</strong></span>
                            <span>•</span>
                            <span>Close Strength: <strong class="text-[#334155]">${{t.closeStrength.toFixed(2)}}</strong></span>
                        </div>
                    </div>
                    <div class="w-full lg:text-right lg:max-w-md border-t lg:border-t-0 pt-4 lg:pt-0 border-[#f1f5f9] flex flex-col lg:items-end gap-1.5">
                        <div>
                            <span class="text-[10px] px-2.5 py-1 rounded-full font-mono font-bold tracking-wide ${{t.sentiment === 'BULLISH' ? 'bg-emerald-100 text-emerald-800 border border-emerald-200' : 'bg-slate-100 text-slate-600 border border-slate-200'}}"><i class="fa-solid fa-circle text-[6px] mr-1.5 align-middle"></i>${{t.sentiment}}</span>
                        </div>
                        <p class="text-xs font-semibold text-[#334155] mt-1 italic leading-relaxed">"${{t.news_catalyst}}"</p>
                        <p class="text-[10px] text-[#94a3b8] font-medium mt-0.5">Confidence Layer Score: <span class="font-mono text-[#475569] font-bold">${{t.confidence}}%</span></p>
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