import json
import base64
import datetime
import requests
import pandas as pd

class ProgrammaticDashboardDeployer:
    """
    Generates an upgraded interactive HTML dashboard containing live 
    LLM sentiment scores, catalysts, and confidence metrics, then commits it directly to GitHub Pages.
    """
    def __init__(self, github_token: str, repo_owner: str, repo_name: str, branch: str = "main"):
        self.github_token = github_token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.branch = branch
        self.api_base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents"

    def generate_html_string(self, execution_df: pd.DataFrame) -> str:
        """
        Converts the Pandas DataFrame (with LLM sentiment columns) to a JSON payload 
        and injects it directly into the dashboard template.
        """
        records = []
        for _, row in execution_df.iterrows():
            # Extract standard and LLM-enriched data safely
            records.append({
                "symbol": str(row["Symbol"]),
                "label": str(row["Strategic_Label"]),
                "close": float(row["Close"]),
                "rsi": float(row["Feature_RSI"]),
                "emaDist": float(row["Feature_EMA_Dist"]),
                "volRatio": float(row["Feature_Volume_Ratio"]),
                "macdAccel": float(row["Feature_MACD_Hist_Accel"]),
                "ibaSco": int(row["Feature_IBA_Score"]),
                "sector": str(row["Sector"]),
                # LLM Overlay parameters
                "sentiment": str(row.get("Sentiment", "NEUTRAL")),
                "news_catalyst": str(row.get("News_Catalyst", "No active fundamental catalyst logged.")),
                "confidence": int(row.get("Confidence_Score", 50)),
                "threat": str(row.get("Strategic_Threat", "No operational risk threats identified."))
            })

        json_tickers_data = json.dumps(records, indent=12)
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Upgraded HTML structure optimized for LLM metrics display
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quant Rotation & Execution Dashboard</title>
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

    <!-- TOP HEADER BAR -->
    <header class="border-b border-gray-800 bg-[#111827]/80 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex flex-wrap justify-between items-center gap-4">
        <div class="flex items-center gap-3">
            <div class="h-10 w-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center text-white shadow-lg shadow-emerald-500/10">
                <i class="fa-solid fa-chart-line-up text-xl"></i>
            </div>
            <div>
                <h1 class="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                    QUANT WAVE <span class="text-xs bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded font-mono">PHASE 2</span>
                </h1>
                <p class="text-xs text-gray-400">Live Top-Down Sector Rotation & Signal Firewall Environment</p>
            </div>
        </div>
        
        <div class="flex items-center gap-4">
            <div class="bg-[#1f2937] border border-gray-800 px-3 py-1.5 rounded-lg flex items-center gap-2 text-xs text-gray-300">
                <span class="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span>Workspace: Nifty 500 Engine</span>
            </div>
            <div class="text-xs text-gray-400 text-right hidden sm:block">
                <p class="font-semibold text-gray-200">System Log Executed</p>
                <p class="font-mono">{timestamp_str}</p>
            </div>
        </div>
    </header>

    <!-- MAIN BODY SECTION -->
    <main class="flex-1 p-6 max-w-7xl w-full mx-auto grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        <!-- LEFT COLUMN: CONTROLS & SECTOR ROTATION STATE -->
        <section class="lg:col-span-5 flex flex-col gap-6">
            
            <!-- CONTROLS & STRATEGY TUNER -->
            <div class="bg-[#111827] border border-gray-800 rounded-2xl p-5 shadow-xl">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-sm font-bold uppercase tracking-wider text-gray-300 flex items-center gap-2">
                        <i class="fa-solid fa-sliders text-emerald-400"></i> Rules Firewall Settings
                    </h3>
                    <button onclick="resetSliders()" class="text-xs text-emerald-400 hover:text-emerald-300 transition flex items-center gap-1">
                        <i class="fa-solid fa-rotate-left"></i> Reset Defaults
                    </button>
                </div>
                
                <div class="space-y-4">
                    <div>
                        <div class="flex justify-between text-xs font-semibold mb-1">
                            <span class="text-gray-400">Max Allowable RSI (Overbought)</span>
                            <span class="text-emerald-400 font-mono" id="rsi-val">75</span>
                        </div>
                        <input type="range" id="rsi-slider" min="65" max="85" value="75" step="1" oninput="updateRules()"
                            class="w-full h-1.5 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-emerald-500">
                    </div>

                    <div>
                        <div class="flex justify-between text-xs font-semibold mb-1">
                            <span class="text-gray-400">Max EMA Distance Allowance</span>
                            <span class="text-emerald-400 font-mono" id="ema-val">8.0%</span>
                        </div>
                        <input type="range" id="ema-slider" min="4" max="15" value="8" step="0.5" oninput="updateRules()"
                            class="w-full h-1.5 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-emerald-500">
                    </div>

                    <div>
                        <div class="flex justify-between text-xs font-semibold mb-1">
                            <span class="text-gray-400">Min Vol Ratio (Breakout)</span>
                            <span class="text-emerald-400 font-mono" id="vol-val">1.25x</span>
                        </div>
                        <input type="range" id="vol-slider" min="1.0" max="2.0" value="1.25" step="0.05" oninput="updateRules()"
                            class="w-full h-1.5 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-emerald-500">
                    </div>

                    <div>
                        <div class="flex justify-between text-xs font-semibold mb-1">
                            <span class="text-gray-400">Asset Price Floor Limit</span>
                            <span class="text-emerald-400 font-mono" id="price-val">₹15</span>
                        </div>
                        <input type="range" id="price-slider" min="5" max="100" value="15" step="5" oninput="updateRules()"
                            class="w-full h-1.5 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-emerald-500">
                    </div>
                </div>

                <div class="mt-4 p-3 bg-emerald-500/5 border border-emerald-500/10 rounded-xl flex items-start gap-3">
                    <i class="fa-solid fa-circle-info text-emerald-400 mt-0.5 text-sm"></i>
                    <p class="text-[11px] text-gray-400 leading-relaxed">
                        Tuning parameters triggers real-time execution audits across the <span class="text-gray-200 font-semibold">96 target tickers</span> in the sandbox. Overbought vetoes will auto-adjust based on the selected RSI threshold.
                    </p>
                </div>
            </div>

            <!-- REGIME ROTATION SUMMARY CARD -->
            <div class="bg-[#111827] border border-gray-800 rounded-2xl p-5 shadow-xl flex-1 flex flex-col">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-sm font-bold uppercase tracking-wider text-gray-300">
                        <i class="fa-solid fa-layer-group text-teal-400 mr-1"></i> Macro Cluster Regimes
                    </h3>
                    <span class="text-xs bg-teal-500/10 text-teal-300 px-2 py-0.5 rounded border border-teal-500/10 font-mono">5 Clusters</span>
                </div>

                <div class="space-y-3 overflow-y-auto flex-1 pr-1" style="max-height: 380px;">
                    <div class="bg-gray-900/50 border border-gray-800 hover:border-emerald-500/30 rounded-xl p-3.5 transition cursor-pointer" onclick="filterByRegime('🔥 ULTRA_MOMENTUM_LEADERS')">
                        <div class="flex justify-between items-start mb-1.5">
                            <span class="text-xs font-bold text-emerald-400 flex items-center gap-1.5">
                                <span class="h-2 w-2 rounded-full bg-emerald-500"></span>
                                🔥 ULTRA MOMENTUM LEADERS
                            </span>
                            <span class="text-[10px] font-mono text-gray-400 bg-gray-800 px-1.5 py-0.5 rounded">High Speed Highway</span>
                        </div>
                        <p class="text-xs text-gray-300 leading-relaxed mb-2">Steady upward grind with strong institutional backing holding shares overnight.</p>
                        <div class="flex items-center gap-4 text-[10px] text-gray-400 font-mono">
                            <span><i class="fa-solid fa-arrow-trend-up text-emerald-400 mr-1"></i>Avg Return: +65.9%</span>
                            <span><i class="fa-solid fa-truck-ramp-box text-blue-400 mr-1"></i>Avg Delivery: 64.6%</span>
                        </div>
                    </div>

                    <div class="bg-gray-900/50 border border-gray-800 hover:border-cyan-500/30 rounded-xl p-3.5 transition cursor-pointer" onclick="filterByRegime('🚀 ACTIVE_BREAKOUT_FIELDS')">
                        <div class="flex justify-between items-start mb-1.5">
                            <span class="text-xs font-bold text-cyan-400 flex items-center gap-1.5">
                                <span class="h-2 w-2 rounded-full bg-cyan-400"></span>
                                🚀 ACTIVE BREAKOUT FIELDS
                            </span>
                            <span class="text-[10px] font-mono text-gray-400 bg-gray-800 px-1.5 py-0.5 rounded">The Rocket Launch</span>
                        </div>
                        <p class="text-xs text-gray-300 leading-relaxed mb-2">Rapid momentum, but low overnight delivery suggests heavy intraday retail churn.</p>
                        <div class="flex items-center gap-4 text-[10px] text-gray-400 font-mono">
                            <span><i class="fa-solid fa-arrow-trend-up text-cyan-400 mr-1"></i>Avg Return: +43.9%</span>
                            <span><i class="fa-solid fa-truck-ramp-box text-blue-400 mr-1"></i>Avg Delivery: 23.0%</span>
                        </div>
                    </div>

                    <div class="bg-gray-900/50 border border-gray-800 hover:border-yellow-500/30 rounded-xl p-3.5 transition cursor-pointer" onclick="filterByRegime('📈 STABLE_UPWARD_ACCUMULATION')">
                        <div class="flex justify-between items-start mb-1.5">
                            <span class="text-xs font-bold text-yellow-400 flex items-center gap-1.5">
                                <span class="h-2 w-2 rounded-full bg-yellow-400"></span>
                                📈 STABLE UPWARD ACCUMULATION
                            </span>
                            <span class="text-[10px] font-mono text-gray-400 bg-gray-800 px-1.5 py-0.5 rounded">Packing Shopping Cart</span>
                        </div>
                        <p class="text-xs text-gray-300 leading-relaxed mb-2">Quiet daily fund purchases keeping volume flat while prices creep up.</p>
                        <div class="flex items-center gap-4 text-[10px] text-gray-400 font-mono">
                            <span><i class="fa-solid fa-arrow-trend-up text-yellow-400 mr-1"></i>Avg Return: +34.0%</span>
                            <span><i class="fa-solid fa-truck-ramp-box text-blue-400 mr-1"></i>Avg Delivery: 13.0%</span>
                        </div>
                    </div>

                    <div class="bg-gray-900/50 border border-gray-800 hover:border-indigo-500/30 rounded-xl p-3.5 transition cursor-pointer" onclick="filterByRegime('⏳ NEUTRAL_SIDEWAYS_CONSOLIDATION')">
                        <div class="flex justify-between items-start mb-1.5">
                            <span class="text-xs font-bold text-indigo-400 flex items-center gap-1.5">
                                <span class="h-2 w-2 rounded-full bg-indigo-400"></span>
                                ⏳ NEUTRAL SIDEWAYS CONSOLIDATION
                            </span>
                            <span class="text-[10px] font-mono text-gray-400 bg-gray-800 px-1.5 py-0.5 rounded">The Waiting Room</span>
                        </div>
                        <p class="text-xs text-gray-300 leading-relaxed mb-2">Sideways coiling, low active momentum. Ready to swing either way.</p>
                        <div class="flex items-center gap-4 text-[10px] text-gray-400 font-mono">
                            <span><i class="fa-solid fa-arrow-trend-up text-indigo-400 mr-1"></i>Avg Return: +2.3%</span>
                            <span><i class="fa-solid fa-truck-ramp-box text-blue-400 mr-1"></i>Avg Delivery: 43.7%</span>
                        </div>
                    </div>

                    <div class="bg-gray-900/50 border border-gray-800 hover:border-rose-500/30 rounded-xl p-3.5 transition cursor-pointer" onclick="filterByRegime('❄️ DEEP_BEARISH_CAPITULATION')">
                        <div class="flex justify-between items-start mb-1.5">
                            <span class="text-xs font-bold text-rose-400 flex items-center gap-1.5">
                                <span class="h-2 w-2 rounded-full bg-rose-400"></span>
                                ❄️ DEEP BEARISH CAPITALIZATION
                            </span>
                            <span class="text-[10px] font-mono text-gray-400 bg-gray-800 px-1.5 py-0.5 rounded">Clearance Rack</span>
                        </div>
                        <p class="text-xs text-gray-300 leading-relaxed mb-2">High-volume panic dumping across retail and institutional segments.</p>
                        <div class="flex items-center gap-4 text-[10px] text-gray-400 font-mono">
                            <span><i class="fa-solid fa-arrow-trend-up text-rose-400 mr-1"></i>Avg Return: +1.2%</span>
                            <span><i class="fa-solid fa-truck-ramp-box text-blue-400 mr-1"></i>Avg Delivery: 50.2%</span>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- RIGHT COLUMN: SIGNAL INTERACTION VIEWPORT -->
        <section class="lg:col-span-7 flex flex-col gap-6">
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div class="bg-[#111827] border border-gray-800 p-4 rounded-xl flex items-center justify-between">
                    <div>
                        <p class="text-[11px] font-semibold text-gray-400 uppercase">Insider Breakouts</p>
                        <p class="text-xl font-bold text-emerald-400 font-mono mt-1" id="stat-insiders">0</p>
                    </div>
                    <div class="h-8 w-8 bg-emerald-500/10 text-emerald-400 rounded-lg flex items-center justify-center">
                        <i class="fa-solid fa-bolt"></i>
                    </div>
                </div>
                <div class="bg-[#111827] border border-gray-800 p-4 rounded-xl flex items-center justify-between">
                    <div>
                        <p class="text-[11px] font-semibold text-gray-400 uppercase">Launchpads</p>
                        <p class="text-xl font-bold text-yellow-400 font-mono mt-1" id="stat-launchpads">0</p>
                    </div>
                    <div class="h-8 w-8 bg-yellow-500/10 text-yellow-400 rounded-lg flex items-center justify-center">
                        <i class="fa-solid fa-rocket"></i>
                    </div>
                </div>
                <div class="bg-[#111827] border border-gray-800 p-4 rounded-xl flex items-center justify-between">
                    <div>
                        <p class="text-[11px] font-semibold text-gray-400 uppercase">Buy Vetoes</p>
                        <p class="text-xl font-bold text-red-400 font-mono mt-1" id="stat-vetoes">0</p>
                    </div>
                    <div class="h-8 w-8 bg-red-500/10 text-red-400 rounded-lg flex items-center justify-center">
                        <i class="fa-solid fa-ban"></i>
                    </div>
                </div>
                <div class="bg-[#111827] border border-gray-800 p-4 rounded-xl flex items-center justify-between">
                    <div>
                        <p class="text-[11px] font-semibold text-gray-400 uppercase">Active Scanners</p>
                        <p class="text-xl font-bold text-teal-400 font-mono mt-1" id="stat-scanned">0</p>
                    </div>
                    <div class="h-8 w-8 bg-teal-500/10 text-teal-400 rounded-lg flex items-center justify-center">
                        <i class="fa-solid fa-crosshairs"></i>
                    </div>
                </div>
            </div>

            <!-- SIGNAL DATATABLE CARD -->
            <div class="bg-[#111827] border border-gray-800 rounded-2xl p-5 shadow-xl flex-1 flex flex-col min-h-[480px]">
                <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-4">
                    <div>
                        <h3 class="text-sm font-bold uppercase tracking-wider text-gray-300">
                            <i class="fa-solid fa-wallet text-emerald-400 mr-1"></i> Firewall Execution Ledger
                        </h3>
                        <p class="text-xs text-gray-400 mt-0.5">Filter, review, and extract dynamic trade orders</p>
                    </div>
                    
                    <div class="flex items-center gap-2 w-full sm:w-auto">
                        <div class="relative flex-1 sm:flex-none">
                            <input type="text" id="ticker-search" placeholder="Search Ticker..." oninput="filterData()"
                                class="bg-gray-900 border border-gray-800 rounded-lg px-3 py-1.5 text-xs text-gray-300 placeholder-gray-500 focus:outline-none focus:border-emerald-500 w-full">
                            <i class="fa-solid fa-magnifying-glass absolute right-3 top-2.5 text-xs text-gray-500"></i>
                        </div>
                        <button onclick="copyToClipboard()" class="bg-emerald-500 hover:bg-emerald-600 text-white font-semibold text-xs px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition">
                            <i class="fa-solid fa-copy"></i> Export Signals
                        </button>
                    </div>
                </div>

                <div class="flex flex-wrap gap-2 mb-4 p-2 bg-gray-900/50 rounded-xl border border-gray-800">
                    <button onclick="setFilter('ALL')" class="filter-btn text-[11px] font-bold px-2.5 py-1 rounded-md bg-gray-800 text-gray-300 hover:bg-gray-700 transition" id="filter-all">ALL</button>
                    <button onclick="setFilter('🚀 INSIDER BREAKOUT')" class="filter-btn text-[11px] font-bold px-2.5 py-1 rounded-md text-emerald-400 bg-emerald-500/5 hover:bg-emerald-500/10 border border-emerald-500/20 transition">INSIDER BREAKOUT</button>
                    <button onclick="setFilter('🚀 ACTIVE BREAKOUT')" class="filter-btn text-[11px] font-bold px-2.5 py-1 rounded-md text-cyan-400 bg-cyan-500/5 hover:bg-cyan-500/10 border border-cyan-500/20 transition">ACTIVE BREAKOUT</button>
                    <button onclick="setFilter('🏢 INSTITUTIONAL LAUNCHPAD')" class="filter-btn text-[11px] font-bold px-2.5 py-1 rounded-md text-yellow-400 bg-yellow-500/5 hover:bg-yellow-500/10 border border-yellow-500/20 transition">LAUNCHPAD</button>
                    <button onclick="setFilter('🛑 OVEREXTENDED')" class="filter-btn text-[11px] font-bold px-2.5 py-1 rounded-md text-red-400 bg-red-500/5 hover:bg-red-500/10 border border-red-500/20 transition">VETOES</button>
                </div>

                <div class="flex-1 overflow-y-auto max-h-[500px]">
                    <table class="w-full text-left text-xs text-gray-300">
                        <thead class="text-[10px] font-bold uppercase tracking-wider text-gray-500 border-b border-gray-800 bg-gray-900/20 sticky top-0 z-10">
                            <tr>
                                <th class="py-3 px-4 bg-[#111827]">Symbol</th>
                                <th class="py-3 px-4 bg-[#111827]">Strategic Label</th>
                                <th class="py-3 px-4 text-right bg-[#111827]">Close</th>
                                <th class="py-3 px-4 text-right bg-[#111827]">RSI</th>
                                <th class="py-3 px-4 text-right bg-[#111827]">EMA Dist</th>
                                <th class="py-3 px-4 bg-[#111827]">Decision Reasoning</th>
                            </tr>
                        </thead>
                        <tbody id="ticker-rows" class="divide-y divide-gray-800/60"></tbody>
                    </table>
                </div>

                <div class="mt-4 pt-3 border-t border-gray-800 flex justify-between items-center text-[11px] text-gray-500">
                    <span>Showing <strong id="visible-count" class="text-gray-400 font-bold">0</strong> of <strong class="text-gray-400 font-bold" id="total-count">0</strong> tickers</span>
                </div>
            </div>
        </section>
    </main>

    <!-- DETAILED STOCK FOCUS SUMMARY CARD FOR THE SELECTED ROW -->
    <section class="max-w-7xl w-full mx-auto px-6 pb-8">
        <div class="bg-[#111827] border border-gray-800 rounded-2xl p-6 shadow-xl" id="detail-pane" style="display: none;">
            <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-4 pb-4 border-b border-gray-800/60">
                <div>
                    <h2 class="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                        <span id="detail-symbol" class="font-mono text-emerald-400">TICKER</span> - 
                        <span id="detail-sector" class="text-sm text-gray-400 font-normal">Sector Category</span>
                    </h2>
                    <p class="text-xs text-gray-400 mt-1">LLM Semantic Analysis and Quantitative Audit Summary</p>
                </div>
                <div class="flex items-center gap-3">
                    <span id="detail-label" class="px-2.5 py-1 rounded text-[11px] font-bold border">LABEL</span>
                    <span id="detail-sentiment" class="px-2.5 py-1 rounded text-[11px] font-bold border">SENTIMENT</span>
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <!-- Catalyst -->
                <div class="bg-gray-900/40 border border-gray-800/80 p-4 rounded-xl">
                    <h4 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                        <i class="fa-solid fa-bullhorn text-emerald-400"></i> Fundamental Catalyst
                    </h4>
                    <p id="detail-catalyst" class="text-xs text-gray-200 leading-relaxed">No active fundamental catalyst logged for this ticker session.</p>
                </div>
                <!-- Strategic Threat -->
                <div class="bg-gray-900/40 border border-gray-800/80 p-4 rounded-xl">
                    <h4 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                        <i class="fa-solid fa-shield-halved text-rose-400"></i> Structural Risk Threat
                    </h4>
                    <p id="detail-threat" class="text-xs text-gray-200 leading-relaxed">No operational metrics logged.</p>
                </div>
                <!-- Confidence Score Radial -->
                <div class="bg-gray-900/40 border border-gray-800/80 p-4 rounded-xl flex flex-col justify-between">
                    <div>
                        <h4 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                            <i class="fa-solid fa-gauge-high text-cyan-400"></i> System Decision Confidence
                        </h4>
                        <p class="text-xs text-gray-400 leading-relaxed">Qualitative analysis mapping score combined with indicators.</p>
                    </div>
                    <div class="flex items-center gap-4 mt-3">
                        <div class="text-3xl font-extrabold text-white font-mono" id="detail-confidence">0%</div>
                        <div class="w-full bg-gray-800 rounded-full h-2.5">
                            <div class="bg-cyan-500 h-2.5 rounded-full" id="detail-confidence-bar" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <footer class="border-t border-gray-850 bg-[#0b0f19] py-4 px-6 text-center text-xs text-gray-500 flex flex-wrap justify-between items-center gap-4 mt-auto">
        <p>© 2026 Quant Trading Framework Engine. Phase 2 Interactive Sandbox.</p>
    </footer>

    <script>
        const initialTickers = {json_tickers_data};

        let activeTickers = JSON.parse(JSON.stringify(initialTickers));
        let activeFilter = 'ALL';
        let activeRegimeFilter = null;

        let params = {{
            maxRsi: 75,
            maxEma: 8.0,
            minVol: 1.25,
            priceFloor: 15
        }};

        function resetSliders() {{
            document.getElementById('rsi-slider').value = 75;
            document.getElementById('ema-slider').value = 8.0;
            document.getElementById('vol-slider').value = 1.25;
            document.getElementById('price-slider').value = 15;
            updateRules();
        }}

        function updateRules() {{
            params.maxRsi = parseFloat(document.getElementById('rsi-slider').value);
            params.maxEma = parseFloat(document.getElementById('ema-slider').value);
            params.minVol = parseFloat(document.getElementById('vol-slider').value);
            params.priceFloor = parseFloat(document.getElementById('price-slider').value);

            document.getElementById('rsi-val').innerText = params.maxRsi;
            document.getElementById('ema-val').innerText = params.maxEma.toFixed(1) + "%";
            document.getElementById('vol-val').innerText = params.minVol.toFixed(2) + "x";
            document.getElementById('price-val').innerText = "₹" + params.priceFloor;

            recalculateSignals();
        }}

        function recalculateSignals() {{
            activeTickers = initialTickers.map(t => {{
                let rec = {{ ...t }};
                
                if (rec.close < params.priceFloor) {{
                    rec.label = "📉 DEEP FLUSH";
                    rec.reason = `Vetoed. Asset price (₹${{rec.close.toFixed(2)}}) is below the system liquidity floor limit of ₹${{params.priceFloor}}.`;
                    return rec;
                }}

                if (rec.rsi >= params.maxRsi || rec.emaDist > params.maxEma) {{
                    rec.label = "🛑 OVEREXTENDED";
                    rec.reason = `Buy Veto. RSI (${{rec.rsi.toFixed(1)}} >= ${{params.maxRsi}}) or distance to 20EMA (${{rec.emaDist.toFixed(1)}}% > ${{params.maxEma}}%) is overextended.`;
                }}
                else if (rec.rsi <= 25 || rec.emaDist < -10) {{
                    rec.label = "📉 DEEP FLUSH";
                    rec.reason = `Avoid Asset. RSI (${{rec.rsi.toFixed(1)}} <= 25) or drop below 20EMA (${{rec.emaDist.toFixed(1)}}%) indicates active capitulation.`;
                }}
                else if (rec.volRatio >= params.minVol && rec.macdAccel > 0 && rec.emaDist > 0) {{
                    if (rec.ibaSco >= 1) {{
                        rec.label = "🚀 INSIDER BREAKOUT";
                        rec.reason = `Confirmed Entry. Volume (${{rec.volRatio.toFixed(2)}}x) with turning MACD and ${{rec.ibaSco}} block trades.`;
                    }} else {{
                        rec.label = "🚀 ACTIVE BREAKOUT";
                        rec.reason = `Momentum Entry. Volume (${{rec.volRatio.toFixed(2)}}x) and positive MACD direction. No blocks logged.`;
                    }}
                }}
                else if (rec.emaDist >= -3.5 && rec.emaDist <= 3.5) {{
                    if (rec.ibaSco >= 2) {{
                        rec.label = "🏢 INSTITUTIONAL LAUNCHPAD";
                        rec.reason = `Accumulation Area. Price coiling tight (${{rec.emaDist.toFixed(1)}}%) with ${{rec.ibaSco}} large blocks.`;
                    }} else {{
                        rec.label = "🏢 LAUNCHPAD";
                        rec.reason = `Standard Consolidation. Trading tight inside core support levels (${{rec.emaDist.toFixed(1)}}%).`;
                    }}
                }}
                else {{
                    rec.label = "⏳ UNCONFIRMED CHOP";
                    rec.reason = `Neutral Zone. Did not meet strategy triggers (RSI: ${{rec.rsi.toFixed(1)}}, Vol Ratio: ${{rec.volRatio.toFixed(2)}}x).`;
                }}
                return rec;
            }});

            filterData();
            updateStats();
        }}

        function updateStats() {{
            let insiders = activeTickers.filter(t => t.label === "🚀 INSIDER BREAKOUT").length;
            let launchpads = activeTickers.filter(t => t.label.includes("LAUNCHPAD")).length;
            let vetoes = activeTickers.filter(t => t.label === "🛑 OVEREXTENDED").length;
            let scanned = activeTickers.length;

            document.getElementById('stat-insiders').innerText = insiders;
            document.getElementById('stat-launchpads').innerText = launchpads;
            document.getElementById('stat-vetoes').innerText = vetoes;
            document.getElementById('stat-scanned').innerText = scanned;
            document.getElementById('total-count').innerText = scanned;
        }}

        function setFilter(filterType) {{
            activeFilter = filterType;
            activeRegimeFilter = null;
            filterData();
        }}

        function filterByRegime(regimeName) {{
            activeRegimeFilter = regimeName;
            filterData();
        }}

        function filterData() {{
            const searchQuery = document.getElementById('ticker-search').value.toUpperCase();
            const tbody = document.getElementById('ticker-rows');
            tbody.innerHTML = '';

            let filtered = activeTickers;

            if (activeFilter !== 'ALL') {{
                filtered = filtered.filter(t => t.label === activeFilter);
            }}

            if (activeRegimeFilter) {{
                const regimeSectors = getSectorsForRegime(activeRegimeFilter);
                filtered = filtered.filter(t => regimeSectors.includes(t.sector));
            }}

            if (searchQuery) {{
                filtered = filtered.filter(t => t.symbol.includes(searchQuery));
            }}

            document.getElementById('visible-count').innerText = filtered.length;

            if (filtered.length === 0) {{
                tbody.innerHTML = `
                    <tr>
                        <td colspan="6" class="py-8 text-center text-gray-500 font-medium">
                            <i class="fa-solid fa-triangle-exclamation text-yellow-500 text-lg mb-2"></i><br>
                            No matching tickers found within current sandbox settings.
                        </td>
                    </tr>
                `;
                return;
            }}

            filtered.forEach(t => {{
                let badgeColor = "bg-gray-800 text-gray-400 border-gray-700";
                if (t.label.includes("INSIDER")) badgeColor = "bg-emerald-500/10 text-emerald-400 border-emerald-500/20";
                else if (t.label.includes("ACTIVE")) badgeColor = "bg-cyan-500/10 text-cyan-400 border-cyan-500/20";
                else if (t.label.includes("INSTITUTIONAL")) badgeColor = "bg-yellow-500/10 text-yellow-400 border-yellow-500/20";
                else if (t.label.includes("LAUNCHPAD")) badgeColor = "bg-amber-500/10 text-amber-300 border-amber-500/20";
                else if (t.label.includes("OVEREXTENDED")) badgeColor = "bg-red-500/10 text-red-400 border-red-500/20";
                else if (t.label.includes("DEEP FLUSH")) badgeColor = "bg-rose-500/10 text-rose-300 border-rose-500/20";

                const rowHtml = `
                    <tr class="hover:bg-gray-900/40 transition cursor-pointer" onclick="viewTickerDetail('${{t.symbol}}')">
                        <td class="py-3 px-4 font-bold tracking-tight text-white font-mono">${{t.symbol}}</td>
                        <td class="py-3 px-4">
                            <span class="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold border ${{badgeColor}}">
                                ${{t.label}}
                            </span>
                        </td>
                        <td class="py-3 px-4 text-right font-mono text-gray-200">₹${{t.close.toFixed(2)}}</td>
                        <td class="py-3 px-4 text-right font-mono ${{t.rsi >= 70 ? 'text-red-400' : t.rsi <= 30 ? 'text-emerald-400' : 'text-gray-400'}}">${{t.rsi.toFixed(1)}}</td>
                        <td class="py-3 px-4 text-right font-mono ${{t.emaDist > 0 ? 'text-emerald-400' : 'text-rose-400'}}">${{t.emaDist.toFixed(2)}}%</td>
                        <td class="py-3 px-4 text-xs text-gray-400 leading-relaxed">${{t.reason || 'Consolidating and maintaining standard coiling bounds.'}}</td>
                    </tr>
                `;
                tbody.insertAdjacentHTML('beforeend', rowHtml);
            }});
        }}

        function viewTickerDetail(symbol) {{
            const ticker = activeTickers.find(t => t.symbol === symbol);
            if (!ticker) return;

            document.getElementById('detail-pane').style.display = 'block';
            document.getElementById('detail-symbol').innerText = ticker.symbol;
            document.getElementById('detail-sector').innerText = ticker.sector.replace(/_/g, " ");
            document.getElementById('detail-label').innerText = ticker.label;
            
            // Set Dynamic Label Badge Style
            const labelBadge = document.getElementById('detail-label');
            labelBadge.className = "px-2.5 py-1 rounded text-[11px] font-bold border " + 
                (ticker.label.includes("BREAKOUT") ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" : "bg-gray-800 text-gray-400 border-gray-700");

            // Set Sentiment Badge Style
            const sentimentBadge = document.getElementById('detail-sentiment');
            sentimentBadge.innerText = ticker.sentiment;
            if (ticker.sentiment === "BULLISH") {{
                sentimentBadge.className = "px-2.5 py-1 rounded text-[11px] font-bold border bg-emerald-500/10 text-emerald-400 border-emerald-500/20";
            }} else if (ticker.sentiment === "BEARISH") {{
                sentimentBadge.className = "px-2.5 py-1 rounded text-[11px] font-bold border bg-rose-500/10 text-rose-400 border-rose-500/20";
            }} else {{
                sentimentBadge.className = "px-2.5 py-1 rounded text-[11px] font-bold border bg-gray-800 text-gray-400 border-gray-700";
            }}

            document.getElementById('detail-catalyst').innerText = ticker.news_catalyst;
            document.getElementById('detail-threat').innerText = ticker.threat;
            document.getElementById('detail-confidence').innerText = ticker.confidence + "%";
            document.getElementById('detail-confidence-bar').style.width = ticker.confidence + "%";

            // Smooth Scroll to Details Pane
            document.getElementById('detail-pane').scrollIntoView({{ behavior: 'smooth' }});
        }}

        function getSectorsForRegime(regimeName) {{
            const map = {{
                "🔥 ULTRA_MOMENTUM_LEADERS": ["TELECOMMUNICATION"],
                "🚀 ACTIVE_BREAKOUT_FIELDS": ["POWER", "METALS_&_MINING"],
                "📈 STABLE_UPWARD_ACCUMULATION": ["OIL_GAS_&_CONSUMABLE_FUELS"],
                "⏳ NEUTRAL_SIDEWAYS_CONSOLIDATION": ["CONSUMER_DURABLES", "FINANCIAL_SERVICES", "FAST_MOVING_CONSUMER_GOODS", "MEDIA_ENTERTAINMENT_&_PUBLICATION", "INFORMATION_TECHNOLOGY", "REALTY"],
                "❄️ DEEP_BEARISH_CAPITULATION": ["SERVICES", "CAPITAL_GOODS", "TEXTILES", "CHEMICALS", "FOREST_MATERIALS", "AUTOMOBILE_AND_AUTO_COMPONENTS", "HEALTHCARE", "CONSUMER_SERVICES", "DIVERSIFIED", "CONSTRUCTION_MATERIALS", "CONSTRUCTION"]
            }};
            return map[regimeName] || [];
        }}

        function copyToClipboard() {{
            const orders = activeTickers.filter(t => t.label.includes("BREAKOUT"));
            let text = "SYMBOL,SIGNAL,PRICE,RSI,EMA_DIST,SENTIMENT,CATALYST\\n";
            orders.forEach(o => {{
                text += `${{o.symbol}},${{o.label}},${{o.close}},${{o.rsi}},${{o.emaDist}}%,${{o.sentiment}},"${{o.news_catalyst}}"\\n`;
            }});

            const dummy = document.createElement("textarea");
            document.body.appendChild(dummy);
            dummy.value = text;
            dummy.select();
            document.execCommand("copy");
            document.body.removeChild(dummy);

            const toast = document.createElement("div");
            toast.className = "fixed bottom-5 right-5 bg-emerald-500 text-white font-bold text-xs px-4 py-2.5 rounded-lg shadow-xl z-50 flex items-center gap-2 border border-emerald-400/20";
            toast.innerHTML = `<i class="fa-solid fa-circle-check"></i> Formatted Dynamic Manifest Exported! (${{orders.length}} orders)`;
            document.body.appendChild(toast);
            
            setTimeout(() => {{
                toast.classList.add('opacity-0', 'transition-opacity', 'duration-500');
                setTimeout(() => toast.remove(), 500);
            }}, 3000);
        }}

        window.onload = function() {{
            recalculateSignals();
        }};
    </script>
</body>
</html>
"""
        return html_template

    def deploy_to_github(self, file_content: str, destination_path: str = "index.html"):
        """Interacts with the GitHub API to deploy the enriched HTML file to Pages."""
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
            "message": f"🤖 Dynamic Phase 2 LLM Dashboard Deployment [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}]",
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
            print(f"❌ [GITHUB REJECTION] API returned error code {r_put.status_code}")
            print(r_put.text)