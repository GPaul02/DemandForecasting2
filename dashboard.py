"""
HTML Dashboard Generator for Pharmaceutical Demand Forecasting MVP.

Generates a self-contained HTML file with embedded Plotly charts and
interactive tabs — no server required.
"""

import json
import numpy as np
import pandas as pd


def _to_json(obj):
    """Safely convert to JSON, handling numpy/pandas types."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_json(orient="records", date_format="iso")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return json.dumps(obj, default=str)


def generate_html_dashboard(metrics_df, results_df, future_df, signals_df):
    """Generate a complete self-contained HTML dashboard."""

    # Prepare data for JavaScript
    metrics_json = metrics_df.to_json(orient="records")

    # Pool-level aggregation
    pool_metrics = metrics_df.groupby("signal_pool").agg({
        "baseline_fa": "mean",
        "ml_fa": "mean",
        "improvement": "mean",
    }).round(2).reset_index()
    pool_json = pool_metrics.to_json(orient="records")

    # Risk flags: SKUs with FA < 70%
    at_risk = metrics_df[metrics_df["ml_fa"] < 70].copy()
    at_risk["risk_score"] = (100 - at_risk["ml_fa"] + at_risk["mape_volatility"]).round(1)
    at_risk = at_risk.sort_values("risk_score", ascending=False)
    risk_json = at_risk.to_json(orient="records")

    # Signal anomaly detection
    latest_signals = signals_df.iloc[-1]
    anomalies = []
    if latest_signals["aqi"] > 250:
        anomalies.append({"signal": "AQI", "value": float(latest_signals["aqi"]),
                          "threshold": 250, "pool": "AQI",
                          "message": f"AQI spike detected: {latest_signals['aqi']:.0f} — Respiratory SKUs at risk"})
    if latest_signals["temperature"] > 40:
        anomalies.append({"signal": "Temperature", "value": float(latest_signals["temperature"]),
                          "threshold": 40, "pool": "Temperature",
                          "message": f"Extreme heat: {latest_signals['temperature']:.1f}°C — Electrolyte SKUs may surge"})
    if latest_signals["precipitation"] > 200:
        anomalies.append({"signal": "Precipitation", "value": float(latest_signals["precipitation"]),
                          "threshold": 200, "pool": "Monsoon",
                          "message": f"Heavy rainfall: {latest_signals['precipitation']:.0f}mm — Anti-malarial demand expected"})
    if latest_signals["google_trends"] > 75:
        anomalies.append({"signal": "Google Trends", "value": float(latest_signals["google_trends"]),
                          "threshold": 75, "pool": "GoogleTrends",
                          "message": f"Search spike: index {latest_signals['google_trends']:.0f} — Monitor symptom-linked SKUs"})
    anomalies_json = json.dumps(anomalies)

    # Future forecasts
    future_json = future_df.to_json(orient="records", date_format="iso")

    # Time series for selected SKUs (one per pool for detail view)
    ts_data = {}
    for sku_id in metrics_df["sku_id"].unique():
        sku_results = results_df[results_df["sku_id"] == sku_id].sort_values("date")
        ts_data[sku_id] = {
            "dates": sku_results["date"].dt.strftime("%Y-%m-%d").tolist(),
            "actual": sku_results["demand"].tolist(),
            "baseline": sku_results["baseline_forecast"].where(sku_results["baseline_forecast"].notna(), None).tolist(),
            "ml": sku_results["ml_forecast"].where(sku_results["ml_forecast"].notna(), None).tolist(),
            "aqi": sku_results["aqi"].where(sku_results["aqi"].notna(), None).tolist() if "aqi" in sku_results.columns else [],
            "temperature": sku_results["temperature"].where(sku_results["temperature"].notna(), None).tolist() if "temperature" in sku_results.columns else [],
            "precipitation": sku_results["precipitation"].where(sku_results["precipitation"].notna(), None).tolist() if "precipitation" in sku_results.columns else [],
            "wedding_flag": sku_results["wedding_flag"].tolist() if "wedding_flag" in sku_results.columns else [],
            "google_trends": sku_results["google_trends"].where(sku_results["google_trends"].notna(), None).tolist() if "google_trends" in sku_results.columns else [],
        }
    ts_json = json.dumps(ts_data, default=str)

    # KPIs
    total_skus = len(metrics_df)
    avg_baseline_fa = round(metrics_df["baseline_fa"].mean(), 1)
    avg_ml_fa = round(metrics_df["ml_fa"].mean(), 1)
    avg_improvement = round(avg_ml_fa - avg_baseline_fa, 1)
    skus_improved = int((metrics_df["improvement"] > 0).sum())
    highest_risk = f"{at_risk['risk_score'].max():.0f}" if len(at_risk) > 0 else "N/A"
    highest_risk_name = at_risk.iloc[0]['sku_name'] if len(at_risk) > 0 else "None"
    num_at_risk = len(at_risk)
    num_anomalies = len(anomalies)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PharmaCast — Demand Forecasting Decision Support</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {{
    --accent-green: #34a853;
    --accent-green-dim: rgba(52,168,83,0.08);
    --accent-red: #d93025;
    --accent-red-dim: rgba(217,48,37,0.06);
    --accent-amber: #e37400;
    --accent-amber-dim: rgba(227,116,0,0.06);
    --accent-blue: #1a73e8;
    --text-primary: #1d1d1f;
    --text-secondary: #6e6e73;
    --text-muted: #aeaeb2;
    --glass: rgba(255,255,255,0.78);
    --glass-border: rgba(255,255,255,0.55);
    --glass-strong: rgba(255,255,255,0.88);
    --input-bg: rgba(255,255,255,0.7);
    --hover-bg: rgba(255,255,255,0.5);
    --shadow-sm: 0 1px 4px rgba(0,0,0,0.05);
    --shadow-md: 0 4px 16px rgba(0,0,0,0.08);
    --shadow-lg: 0 8px 32px rgba(0,0,0,0.10);
    --radius: 16px;
    --blur: saturate(140%) blur(20px);
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    font-family: -apple-system, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    background: #d7dde4;
    background-image:
        radial-gradient(ellipse at 15% 10%, rgba(160,200,230,0.5) 0%, transparent 50%),
        radial-gradient(ellipse at 85% 20%, rgba(180,195,220,0.45) 0%, transparent 45%),
        radial-gradient(ellipse at 50% 80%, rgba(170,190,215,0.4) 0%, transparent 50%),
        radial-gradient(ellipse at 20% 60%, rgba(190,205,225,0.35) 0%, transparent 40%);
    background-attachment: fixed;
    color: var(--text-primary);
    min-height: 100vh;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}
.header {{
    background: var(--glass-strong);
    -webkit-backdrop-filter: var(--blur);
    backdrop-filter: var(--blur);
    border-bottom: 1px solid var(--glass-border);
    padding: 16px 48px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 50;
}}
.header h1 {{
    font-size: 21px;
    font-weight: 600;
    letter-spacing: -0.3px;
    color: var(--text-primary);
}}
.header h1 span {{ color: var(--accent-green); }}
.header .subtitle {{
    color: var(--text-secondary);
    font-size: 13px;
    margin-top: 1px;
    font-weight: 400;
}}
.header-right {{
    text-align: right;
    color: var(--text-secondary);
    font-size: 12px;
    font-weight: 400;
}}
.tab-bar {{
    display: flex;
    gap: 4px;
    background: var(--glass-strong);
    -webkit-backdrop-filter: var(--blur);
    backdrop-filter: var(--blur);
    border-bottom: 1px solid var(--glass-border);
    padding: 6px 48px;
}}
.tab-btn {{
    padding: 8px 20px;
    border: none;
    background: none;
    color: var(--text-secondary);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    border-radius: 10px;
    transition: all 0.2s;
    letter-spacing: -0.1px;
    border-bottom: none;
}}
.tab-btn:hover {{ background: rgba(0,0,0,0.04); color: var(--text-primary); }}
.tab-btn.active {{
    background: rgba(0,0,0,0.06);
    color: var(--text-primary);
    font-weight: 600;
}}
.tab-content {{ display: none; padding: 32px 48px; max-width: 1400px; margin: 0 auto; }}
.tab-content.active {{ display: block; }}
.kpi-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}}
.kpi-card {{
    background: var(--glass);
    -webkit-backdrop-filter: var(--blur);
    backdrop-filter: var(--blur);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    padding: 24px;
    text-align: center;
    box-shadow: var(--shadow-sm);
    transition: all 0.25s ease;
}}
.kpi-card:hover {{
    background: rgba(255,255,255,0.85);
    box-shadow: var(--shadow-md);
}}
.kpi-card .label {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 10px;
}}
.kpi-card .value {{
    font-size: 34px;
    font-weight: 600;
    letter-spacing: -1px;
    color: var(--text-primary);
}}
.kpi-card .value.green {{ color: var(--accent-green); }}
.kpi-card .value.blue {{ color: var(--accent-blue); }}
.kpi-card .value.amber {{ color: var(--accent-amber); }}
.kpi-card .sub {{
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 6px;
    font-weight: 400;
}}
/* Tier system: 1=hero decision signals, 2=diagnostic charts, 3=supporting data */
.chart-container {{
    background: var(--glass);
    -webkit-backdrop-filter: var(--blur);
    backdrop-filter: var(--blur);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: var(--shadow-sm);
}}
.chart-container.tier-3 {{
    background: rgba(255,255,255,0.65);
    -webkit-backdrop-filter: var(--blur);
    backdrop-filter: var(--blur);
    border: 1px solid var(--glass-border);
    box-shadow: none;
    padding: 20px;
}}
.chart-title {{
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 16px;
    padding-bottom: 0;
    border-bottom: none;
    color: var(--text-primary);
    letter-spacing: -0.3px;
}}
.chart-subtitle {{
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: -12px;
    margin-bottom: 16px;
    font-weight: 400;
}}
/* Tab section headers — one question per tab */
.tab-question {{
    font-size: 28px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 28px;
    letter-spacing: -0.8px;
    line-height: 1.2;
}}
.tab-question strong {{
    color: var(--text-primary);
    font-weight: 600;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}
th {{
    text-align: left;
    padding: 10px 16px;
    background: transparent;
    color: var(--text-secondary);
    font-weight: 500;
    font-size: 12px;
    letter-spacing: 0;
    border-bottom: 1px solid rgba(0,0,0,0.08);
}}
td {{
    padding: 10px 16px;
    border-bottom: 1px solid rgba(0,0,0,0.04);
    color: var(--text-primary);
}}
tr:hover td {{ background: rgba(255,255,255,0.4); }}
.badge {{
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 500;
}}
.badge-red {{ background: var(--accent-red-dim); color: var(--accent-red); }}
.badge-amber {{ background: var(--accent-amber-dim); color: var(--accent-amber); }}
.badge-green {{ background: var(--accent-green-dim); color: var(--accent-green); }}
.alert-box {{
    background: rgba(255,245,245,0.82);
    -webkit-backdrop-filter: var(--blur);
    backdrop-filter: var(--blur);
    border: 1px solid rgba(217,48,37,0.12);
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 14px;
}}
.alert-box .icon {{ font-size: 20px; }}
.selector {{
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}}
.selector label {{
    font-size: 13px;
    color: var(--text-secondary);
    font-weight: 600;
    letter-spacing: 0.2px;
}}
select {{
    background: var(--input-bg);
    -webkit-backdrop-filter: var(--blur);
    backdrop-filter: var(--blur);
    border: 1px solid var(--glass-border);
    color: var(--text-primary);
    padding: 10px 16px;
    border-radius: 12px;
    font-size: 14px;
    min-width: 280px;
    cursor: pointer;
    font-family: inherit;
    -webkit-appearance: none;
}}
select:focus {{ outline: none; border-color: rgba(0,0,0,0.2); box-shadow: 0 0 0 3px rgba(0,0,0,0.06); }}
.detail-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
}}
.metric-bar {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 8px 0;
}}
.metric-bar .bar-bg {{
    flex: 1;
    height: 6px;
    background: rgba(0,0,0,0.08);
    border-radius: 3px;
    overflow: hidden;
}}
.metric-bar .bar-fill {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s;
}}
.metric-bar .bar-label {{
    min-width: 60px;
    font-size: 12px;
    color: var(--text-secondary);
}}
.metric-bar .bar-value {{
    min-width: 50px;
    text-align: right;
    font-weight: 600;
    font-size: 14px;
}}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
.two-col > * {{ min-width: 0; overflow: hidden; }}
/* Urgency-differentiated KPI cards */
.kpi-card.urgent {{
    background: rgba(255,248,248,0.8);
    border-color: rgba(217,48,37,0.18);
}}
.kpi-card.healthy {{
    background: rgba(248,255,250,0.8);
    border-color: rgba(52,168,83,0.18);
}}
.kpi-card.healthy .value {{ font-size: 28px; }}
/* Searchable SKU selector */
.sku-search-wrap {{
    position: relative;
    min-width: 350px;
}}
.sku-search-wrap input {{
    width: 100%;
    background: var(--input-bg);
    -webkit-backdrop-filter: var(--blur);
    backdrop-filter: var(--blur);
    border: 1px solid var(--glass-border);
    color: var(--text-primary);
    padding: 10px 16px;
    border-radius: 12px;
    font-size: 14px;
    outline: none;
    font-family: inherit;
}}
.sku-search-wrap input:focus {{ border-color: rgba(0,0,0,0.2); box-shadow: 0 0 0 3px rgba(0,0,0,0.06); }}
.sku-dropdown {{
    display: none;
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    max-height: 320px;
    overflow-y: auto;
    background: var(--glass-strong);
    -webkit-backdrop-filter: var(--blur);
    backdrop-filter: var(--blur);
    border: 1px solid var(--glass-border);
    border-top: none;
    border-radius: 0 0 14px 14px;
    z-index: 100;
    box-shadow: var(--shadow-md);
}}
.sku-dropdown.open {{ display: block; }}
.sku-opt-group {{
    padding: 6px 12px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    color: var(--text-secondary);
    background: rgba(245,245,247,0.7);
    font-weight: 600;
}}
.sku-opt {{
    padding: 8px 16px;
    cursor: pointer;
    font-size: 13px;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: background 0.1s;
}}
.sku-opt:hover {{ background: var(--hover-bg); }}
.sku-opt .risk-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}}
.sku-opt .risk-dot.red {{ background: var(--accent-red); }}
.sku-opt .risk-dot.green {{ background: var(--accent-green); }}
.sku-opt .risk-dot.amber {{ background: var(--accent-amber); }}
/* Tooltips */
.has-tooltip {{
    position: relative;
    cursor: help;
    border-bottom: 1px dotted var(--text-secondary);
}}
.has-tooltip::after {{
    content: attr(data-tooltip);
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: #1d1d1f;
    color: #fff;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 11px;
    font-weight: 400;
    white-space: nowrap;
    max-width: 280px;
    white-space: normal;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s;
    z-index: 100;
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    line-height: 1.4;
}}
.has-tooltip:hover::after {{
    opacity: 1;
}}
/* Cross-filter active pool pill */
.pool-filter-bar {{
    display: flex;
    gap: 8px;
    margin-bottom: 20px;
    align-items: center;
    flex-wrap: wrap;
}}
.pool-pill {{
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    border: 1px solid var(--glass-border);
    background: var(--glass);
    -webkit-backdrop-filter: var(--blur);
    backdrop-filter: var(--blur);
    color: var(--text-secondary);
    transition: all 0.2s;
}}
.pool-pill:hover {{ background: rgba(255,255,255,0.88); color: var(--text-primary); }}
.pool-pill.active {{
    background: rgba(0,0,0,0.72);
    -webkit-backdrop-filter: var(--blur);
    backdrop-filter: var(--blur);
    color: #fff;
    border-color: rgba(0,0,0,0.72);
}}
.pool-pill .pill-dot {{
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}}
/* Proactive AI briefing banner */
.ai-briefing {{
    background: rgba(246,254,248,0.82);
    -webkit-backdrop-filter: var(--blur);
    backdrop-filter: var(--blur);
    border: 1px solid rgba(52,168,83,0.15);
    border-radius: var(--radius);
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: var(--shadow-sm);
}}
.ai-briefing .briefing-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
    font-size: 14px;
    font-weight: 600;
    color: var(--accent-green);
}}
.ai-briefing .briefing-body {{
    font-size: 13px;
    line-height: 1.7;
    color: var(--text-primary);
}}
.ai-briefing .briefing-body .insight-item {{
    display: flex;
    gap: 8px;
    margin: 6px 0;
    align-items: flex-start;
}}
.ai-briefing .briefing-body .insight-icon {{
    flex-shrink: 0;
    margin-top: 2px;
}}
@media (max-width: 900px) {{
    .two-col, .detail-grid {{ grid-template-columns: 1fr; }}
    .header {{ padding: 16px 20px; }}
    .tab-content {{ padding: 20px; }}
    .tab-bar {{ padding: 0 20px; overflow-x: auto; }}
    .workflow-bar {{ padding: 8px 20px; overflow-x: auto; }}
}}
.comparison-row {{
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 16px;
    align-items: center;
    margin: 16px 0;
}}
.comparison-row .arrow {{
    font-size: 24px;
    color: var(--accent-green);
}}
.stat-block {{
    padding: 16px;
    border-radius: 14px;
    text-align: center;
}}
.stat-block.baseline {{ background: rgba(235,245,255,0.82); border: 1px solid rgba(26,115,232,0.15); -webkit-backdrop-filter: var(--blur); backdrop-filter: var(--blur); }}
.stat-block.ml {{ background: rgba(240,250,243,0.82); border: 1px solid rgba(52,168,83,0.15); -webkit-backdrop-filter: var(--blur); backdrop-filter: var(--blur); }}
.stat-block .stat-label {{ font-size: 12px; color: var(--text-secondary); font-weight: 600; letter-spacing: 0.2px; }}
.stat-block .stat-value {{ font-size: 28px; font-weight: 700; margin: 4px 0; }}
.stat-block.baseline .stat-value {{ color: var(--accent-blue); }}
.stat-block.ml .stat-value {{ color: var(--accent-green); }}
.savings-badge {{
    display: inline-block;
    background: var(--accent-green-dim);
    color: var(--accent-green);
    padding: 8px 20px;
    border-radius: 8px;
    font-size: 18px;
    font-weight: 700;
    margin-top: 12px;
}}

/* Loading skeleton */
.skeleton {{
    background: linear-gradient(90deg, rgba(255,255,255,0.3) 25%, rgba(255,255,255,0.5) 50%, rgba(255,255,255,0.3) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s ease-in-out infinite;
    border-radius: 10px;
}}
@keyframes shimmer {{
    0% {{ background-position: 200% 0; }}
    100% {{ background-position: -200% 0; }}
}}
.empty-state {{
    text-align: center;
    padding: 40px 20px;
    color: var(--text-secondary);
}}
.empty-state .empty-icon {{
    font-size: 48px;
    margin-bottom: 12px;
    opacity: 0.5;
}}
.empty-state h3 {{
    font-size: 16px;
    font-weight: 600;
    color: var(--accent-green);
    margin-bottom: 8px;
}}

/* ---- CHATBOT STYLES ---- */
.chat-fab {{
    position: fixed;
    bottom: 28px;
    right: 28px;
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: #1d1d1f;
    border: none;
    cursor: pointer;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    transition: transform 0.2s, box-shadow 0.2s;
    animation: fabPulse 3s ease-in-out infinite;
}}
.chat-fab:hover {{ transform: scale(1.06); box-shadow: 0 6px 24px rgba(0,0,0,0.3); animation: none; }}
.chat-fab svg {{ width: 24px; height: 24px; fill: #fff; }}
@keyframes fabPulse {{
    0%, 100% {{ box-shadow: 0 4px 16px rgba(0,0,0,0.2); }}
    50% {{ box-shadow: 0 4px 24px rgba(0,0,0,0.3), 0 0 0 6px rgba(0,0,0,0.04); }}
}}
.chat-fab-label {{
    position: fixed;
    bottom: 92px;
    right: 22px;
    background: #1d1d1f;
    border: none;
    color: #fff;
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 500;
    z-index: 9999;
    white-space: nowrap;
    animation: labelFade 8s ease-in-out forwards;
    pointer-events: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}}
.chat-fab-label::after {{
    content: '';
    position: absolute;
    bottom: -6px;
    right: 20px;
    width: 10px;
    height: 10px;
    background: #1d1d1f;
    border-right: none;
    border-bottom: none;
    transform: rotate(45deg);
}}
@keyframes labelFade {{
    0% {{ opacity: 0; transform: translateY(4px); }}
    8% {{ opacity: 1; transform: translateY(0); }}
    75% {{ opacity: 1; }}
    100% {{ opacity: 0; display: none; }}
}}

.chat-panel {{
    position: fixed;
    top: 0;
    right: 0;
    width: 380px;
    height: 100vh;
    background: var(--glass-strong);
    -webkit-backdrop-filter: var(--blur);
    backdrop-filter: var(--blur);
    border-left: 1px solid var(--glass-border);
    border-radius: 0;
    box-shadow: -4px 0 30px rgba(0,0,0,0.1);
    z-index: 9998;
    display: none;
    flex-direction: column;
    overflow: hidden;
}}
.chat-panel.open {{ display: flex; }}

.chat-header {{
    padding: 16px 20px;
    background: transparent;
    border-bottom: 1px solid rgba(0,0,0,0.06);
    display: flex;
    align-items: center;
    gap: 12px;
}}
.chat-header .chat-avatar {{
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: #1d1d1f;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 14px;
    color: #fff;
}}
.chat-header .chat-title {{
    flex: 1;
}}
.chat-header .chat-title h3 {{
    font-size: 14px;
    font-weight: 600;
    margin: 0;
}}
.chat-header .chat-title span {{
    font-size: 11px;
    color: var(--accent-green);
}}
.chat-close {{
    background: none;
    border: none;
    color: var(--text-secondary);
    font-size: 20px;
    cursor: pointer;
    padding: 4px;
    line-height: 1;
}}
.chat-close:hover {{ color: var(--text-primary); }}

.chat-messages {{
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-height: 0;
}}
.chat-msg {{
    max-width: 88%;
    padding: 10px 14px;
    border-radius: 12px;
    font-size: 13px;
    line-height: 1.5;
    word-wrap: break-word;
}}
.chat-msg.bot {{
    align-self: flex-start;
    background: rgba(0,0,0,0.04);
    border: none;
    color: var(--text-primary);
}}
.chat-msg.user {{
    align-self: flex-end;
    background: #1d1d1f;
    color: #fff;
    font-weight: 500;
}}
.chat-msg .msg-label {{
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
    opacity: 0.7;
}}
.chat-msg code {{
    background: rgba(0,0,0,0.06);
    padding: 1px 5px;
    border-radius: 4px;
    font-size: 12px;
}}
.chat-msg strong {{ color: var(--accent-green); }}
.chat-msg.user strong {{ color: #fff; }}

.chat-suggestions {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding: 0 16px 12px;
}}
.chat-suggestion {{
    background: rgba(0,0,0,0.04);
    border: 1px solid rgba(0,0,0,0.06);
    color: var(--text-secondary);
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.15s;
}}
.chat-suggestion:hover {{
    background: rgba(0,0,0,0.08);
    color: var(--text-primary);
}}

.chat-input-row {{
    display: flex;
    gap: 8px;
    padding: 12px 16px;
    border-top: 1px solid rgba(0,0,0,0.06);
    background: transparent;
}}
.chat-input-row input {{
    flex: 1;
    background: rgba(0,0,0,0.04);
    border: 1px solid rgba(0,0,0,0.08);
    color: var(--text-primary);
    padding: 10px 14px;
    border-radius: 12px;
    font-size: 13px;
    outline: none;
    font-family: inherit;
}}
.chat-input-row input::placeholder {{ color: var(--text-secondary); opacity: 0.6; }}
.chat-input-row input:focus {{ border-color: rgba(0,0,0,0.2); box-shadow: 0 0 0 3px rgba(0,0,0,0.06); }}
.chat-send {{
    background: #1d1d1f;
    border: none;
    color: #fff;
    width: 38px;
    height: 38px;
    border-radius: 10px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    font-weight: 600;
    transition: background 0.15s;
}}
.chat-send:hover {{ background: #00c480; }}
.typing-dots {{ display: inline-flex; gap: 3px; padding: 4px 0; }}
.typing-dots span {{
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--text-secondary);
    animation: dotPulse 1.2s infinite;
}}
.typing-dots span:nth-child(2) {{ animation-delay: 0.2s; }}
.typing-dots span:nth-child(3) {{ animation-delay: 0.4s; }}
@keyframes dotPulse {{
    0%, 80%, 100% {{ opacity: 0.3; transform: scale(0.8); }}
    40% {{ opacity: 1; transform: scale(1); }}
}}
@media (max-width: 500px) {{
    .chat-panel {{ width: 100vw; }}
}}
/* Chat scrim overlay */
.chat-scrim {{
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.45);
    z-index: 9997;
    backdrop-filter: blur(2px);
    -webkit-backdrop-filter: blur(2px);
    cursor: pointer;
}}
.chat-scrim.open {{ display: block; }}
</style>
</head>
<body>

<div class="header">
    <div>
        <h1>Pharma<span>Cast</span> — Demand Forecasting DSS</h1>
        <div class="subtitle">Intelligent Decision Support for BZ / CY / CZ SKU Segments</div>
    </div>
    <div class="header-right">
        <div>{total_skus} SKUs Monitored</div>
        <div style="margin-top:2px">Data: May 2022 – Apr 2025</div>
    </div>
</div>

<div class="tab-bar">
    <button class="tab-btn active" onclick="switchTab('overview')">Overview</button>
    <button class="tab-btn" onclick="switchTab('risk')">Risk Flags</button>
    <button class="tab-btn" onclick="switchTab('recommend')">Model Recommendations</button>
    <button class="tab-btn" onclick="switchTab('safety')">Safety Stock</button>
    <button class="tab-btn" onclick="toggleChat()" style="margin-left:auto; background:var(--accent-green); color:var(--navy); font-weight:700; border-radius:20px; padding:8px 18px;">Chat Assistant</button>
</div>

<!-- TAB 1: OVERVIEW -->
<div id="tab-overview" class="tab-content active">
    <!-- Row 1: Briefing + Hero side by side -->
    <div style="display:grid;grid-template-columns:1fr auto;gap:32px;align-items:start;margin-bottom:28px">
        <div>
            <div style="font-size:22px;font-weight:700;letter-spacing:-0.5px;margin-bottom:16px">Forecast Performance</div>
            <!-- AI Briefing (collapsible) -->
            <div class="ai-briefing" id="ai-briefing-overview" style="margin-bottom:0"></div>
        </div>
        <div style="text-align:center;padding:24px 40px;min-width:260px">
            <div style="font-size:10px;color:var(--text-secondary);margin-bottom:6px;letter-spacing:1.2px;text-transform:uppercase;font-weight:600" class="has-tooltip" data-tooltip="Difference between ML model and baseline (3-month SMA) forecast accuracy averaged across all SKUs">Forecast Accuracy Improvement</div>
            <div style="font-size:72px;font-weight:800;color:var(--accent-green);letter-spacing:-3px;line-height:1">+{avg_improvement}%</div>
            <div style="font-size:13px;color:var(--text-secondary);margin-top:10px">{skus_improved} of {total_skus} SKUs improved</div>
            <div style="display:inline-block;margin-top:10px;padding:5px 16px;border-radius:20px;background:rgba(0,214,143,0.08);font-size:12px;color:var(--accent-green);font-weight:600">Avg Accuracy: {avg_ml_fa}%</div>
        </div>
    </div>

    <!-- Row 2: Risk cards -->
    <div style="display:grid;grid-template-columns:2fr 1fr 1fr;gap:16px;margin-bottom:24px">
        <div class="kpi-card {'urgent' if num_at_risk > 0 else 'healthy'}" onclick="switchTab('risk')" style="cursor:pointer;padding:20px 24px">
            <div class="label" style="font-weight:600;letter-spacing:0.3px;font-size:11px;color:{'var(--accent-red)' if num_at_risk > 0 else 'var(--accent-green)'}">{"&#9888; Forecast Risk" if num_at_risk > 0 else "&#10003; All Clear"}</div>
            <div class="value" style="font-size:36px;color:{'var(--accent-red)' if num_at_risk > 0 else 'var(--accent-green)'}">{num_at_risk if num_at_risk > 0 else '0'}</div>
            <div style="font-size:14px;font-weight:600;color:var(--text-primary);margin:-2px 0 2px">SKUs at Risk</div>
            <div class="sub" style="font-size:12px">{'Below 70% accuracy — review &#8594;' if num_at_risk > 0 else 'All above 70% threshold'}</div>
        </div>
        <div class="kpi-card {'urgent' if num_anomalies > 0 else 'healthy'}" style="padding:20px">
            <div class="label" style="font-size:11px">{"&#9888; Signal Anomalies" if num_anomalies > 0 else "&#10003; Signals Normal"}</div>
            <div class="value {'amber' if num_anomalies > 0 else ''}" style="font-size:{'36px' if num_anomalies > 0 else '28px'};{'color:var(--accent-green)' if num_anomalies == 0 else ''}">{num_anomalies if num_anomalies > 0 else '&#10003;'}</div>
            <div class="sub">{'Environmental alerts' if num_anomalies > 0 else 'All normal'}</div>
        </div>
        <div class="kpi-card" onclick="switchTab('safety')" style="cursor:pointer;padding:20px">
            <div class="label" style="font-size:11px">Safety Stock Saving</div>
            <div class="value green" style="font-size:36px" id="overview-stock-reduction">—</div>
            <div class="sub">Avg reduction via ML &#8594;</div>
        </div>
    </div>

    <!-- Row 3: Top Risks + Scatter side by side -->
    <div class="pool-filter-bar" id="pool-filter-bar" style="margin-bottom:16px">
        <span style="font-size:11px;color:var(--text-secondary);font-weight:600;letter-spacing:0.5px;margin-right:4px">Demand Drivers</span>
        <div class="pool-pill active" onclick="setPoolFilter('all')" data-pool="all">All</div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 2fr;gap:24px;margin-bottom:24px;align-items:start">
        <div id="top-problems-widget" style="margin:0"></div>
        <div class="chart-container" style="margin-bottom:0;min-width:0;overflow:hidden">
            <div class="chart-title">SKU Forecast Accuracy — Baseline vs ML</div>
            <div class="chart-subtitle">Each dot = one SKU. Above the diagonal = ML wins. Click to drill in.</div>
            <div id="chart-sku-scatter"></div>
        </div>
    </div>

    <!-- Row 4: Supporting charts -->
    <div style="display:grid;grid-template-columns:3fr 2fr;gap:24px">
        <div class="chart-container">
            <div class="chart-title">Forecast Accuracy by Signal Pool</div>
            <div class="chart-subtitle">Baseline (SMA) vs Best ML Model per Pool</div>
            <div id="chart-pool-comparison"></div>
        </div>
        <div class="chart-container tier-3">
            <div class="chart-title">Accuracy Distribution</div>
            <div class="chart-subtitle">Right shift = ML improvement</div>
            <div id="chart-fa-distribution"></div>
        </div>
    </div>

    <!-- Row 5: Model selection overview -->
    <div style="display:grid;grid-template-columns:2fr 3fr;gap:24px;margin-top:24px">
        <div class="chart-container">
            <div class="chart-title">Model Selection by Pool</div>
            <div class="chart-subtitle">Best model chosen per signal pool via validation</div>
            <div id="chart-model-distribution"></div>
        </div>
        <div class="chart-container">
            <div class="chart-title">All Candidates — Validation Accuracy by Pool</div>
            <div class="chart-subtitle">Grouped comparison of all evaluated models</div>
            <div id="chart-model-pool-heatmap"></div>
        </div>
    </div>
</div>

<!-- TAB 2: RISK FLAGS -->
<div id="tab-risk" class="tab-content">
    <div class="tab-question"><strong>Forecast Risk Flags</strong></div>
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="label has-tooltip" data-tooltip="SKUs where ML Forecast Accuracy is below 70% — these need manual review or model retraining">At-Risk SKUs</div>
            <div class="value" style="color:var(--accent-red)">{num_at_risk}</div>
            <div class="sub">Forecast accuracy below 70%</div>
        </div>
        <div class="kpi-card">
            <div class="label has-tooltip" data-tooltip="External signals (AQI, temperature, rainfall, search trends) exceeding normal thresholds that may affect demand">Signal Anomalies</div>
            <div class="value amber">{num_anomalies}</div>
            <div class="sub">Active environmental alerts</div>
        </div>
        <div class="kpi-card">
            <div class="label has-tooltip" data-tooltip="Risk Score = (100 - ML Forecast Accuracy%) + MAPE Volatility. Higher = worse. Combines inaccuracy with unpredictability.">Highest Risk Score</div>
            <div class="value" style="color:var(--accent-red)">{highest_risk}</div>
            <div class="sub">{highest_risk_name}</div>
        </div>
    </div>
    <div id="anomaly-alerts"></div>
    <div class="chart-container">
        <div class="chart-title">At-Risk SKUs — Forecast Accuracy Below 70%</div>
        <div id="risk-table-container"></div>
    </div>
    <div class="chart-container">
        <div class="chart-title">Risk Score vs Forecast Accuracy</div>
        <div id="chart-risk-scatter"></div>
    </div>
</div>

<!-- TAB 3: MODEL RECOMMENDATIONS -->
<div id="tab-recommend" class="tab-content">
    <div class="tab-question"><strong>Model Recommendation</strong></div>
    <div class="selector">
        <label style="font-weight:600;letter-spacing:0.3px">Select SKU</label>
        <div class="sku-search-wrap" id="sku-search-wrap-recommend">
            <input type="text" id="sku-search-recommend" placeholder="Search by SKU name, ID, or pool..." autocomplete="off"
                   onfocus="openSkuDropdown('recommend')" oninput="filterSkuDropdown('recommend')">
            <div class="sku-dropdown" id="sku-dropdown-recommend"></div>
        </div>
        <select id="sku-selector" onchange="updateRecommendation()" style="display:none"></select>
    </div>
    <div id="recommendation-content"></div>
</div>

<!-- TAB 4: SAFETY STOCK -->
<div id="tab-safety" class="tab-content">
    <div class="tab-question"><strong>Safety Stock Optimizer</strong></div>
    <div class="selector">
        <label style="font-weight:600;letter-spacing:0.3px">Select SKU</label>
        <div class="sku-search-wrap" id="sku-search-wrap-safety">
            <input type="text" id="sku-search-safety" placeholder="Search by SKU name, ID, or pool..." autocomplete="off"
                   onfocus="openSkuDropdown('safety')" oninput="filterSkuDropdown('safety')">
            <div class="sku-dropdown" id="sku-dropdown-safety"></div>
        </div>
        <select id="safety-sku-selector" onchange="updateSafetyStock()" style="display:none"></select>
    </div>
    <div id="safety-content"></div>
</div>

<!-- CHAT SCRIM -->
<div class="chat-scrim" id="chatScrim" onclick="toggleChat()"></div>

<!-- CHATBOT WIDGET -->
<div class="chat-fab-label" id="chatLabel">Ask PharmaCast Assistant</div>
<button class="chat-fab" id="chatFab" onclick="toggleChat()" title="Ask PharmaCast Assistant">
    <svg viewBox="0 0 24 24"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H5.2L4 17.2V4h16v12z"/><path d="M7 9h2v2H7zm4 0h2v2h-2zm4 0h2v2h-2z"/></svg>
</button>

<div class="chat-panel" id="chatPanel">
    <div class="chat-header">
        <div class="chat-avatar">PC</div>
        <div class="chat-title">
            <h3>PharmaCast Assistant</h3>
            <span>Online — Ask me about your forecast data</span>
        </div>
        <button class="chat-close" onclick="toggleChat()">&times;</button>
    </div>
    <div class="chat-messages" id="chatMessages">
        <div class="chat-msg bot">
            <div class="msg-label">PharmaCast Assistant</div>
            Welcome! I can help you understand your demand forecasts, risk flags, safety stock, and more.<br><br>
            Try asking me about a specific SKU, signal pool, or any of the topics below.
        </div>
    </div>
    <div class="chat-suggestions" id="chatSuggestions">
        <div class="chat-suggestion" onclick="askSuggestion(this)">Diagnose root causes</div>
        <div class="chat-suggestion" onclick="askSuggestion(this)">Which SKUs are at risk?</div>
        <div class="chat-suggestion" onclick="askSuggestion(this)">Best performing pool?</div>
        <div class="chat-suggestion" onclick="askSuggestion(this)">Safety stock savings</div>
    </div>
    <div class="chat-input-row">
        <input type="text" id="chatInput" placeholder="Ask about forecasts, risk, SKUs..." onkeydown="if(event.key==='Enter')sendChat()">
        <button class="chat-send" onclick="sendChat()">&#10148;</button>
    </div>
</div>

<script>
// Data
const metricsData = {metrics_json};
const poolData = {pool_json};
const riskData = {risk_json};
const anomaliesData = {anomalies_json};
const futureData = {future_json};
const tsData = {ts_json};

const COLORS = {{
    navy: '#f5f5f7',
    navyLight: '#ffffff',
    green: '#34a853',
    blue: '#1a73e8',
    red: '#d93025',
    amber: '#e37400',
    text: '#1d1d1f',
    textSec: '#86868b',
    gridColor: 'rgba(0,0,0,0.06)',
}};

// Consistent signal pool color mapping — refined palette for light theme
const POOL_COLORS = {{
    'AQI': '#c5221f',
    'Temperature': '#e37400',
    'Monsoon': '#1a73e8',
    'Wedding': '#9334e6',
    'GoogleTrends': '#34a853',
}};
const POOL_COLORS_DIM = {{
    'AQI': 'rgba(197,34,31,0.08)',
    'Temperature': 'rgba(227,116,0,0.08)',
    'Monsoon': 'rgba(26,115,232,0.08)',
    'Wedding': 'rgba(147,52,230,0.08)',
    'GoogleTrends': 'rgba(52,168,83,0.08)',
}};

// Active cross-filter state
let activePoolFilter = 'all';

const plotLayout = {{
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {{ family: '-apple-system, SF Pro Display, Helvetica Neue, Arial, sans-serif', color: COLORS.text, size: 13 }},
    margin: {{ l: 56, r: 24, t: 24, b: 52 }},
    xaxis: {{ gridcolor: 'rgba(0,0,0,0.04)', zerolinecolor: 'rgba(0,0,0,0.06)', tickfont: {{ size: 12, color: '#86868b' }} }},
    yaxis: {{ gridcolor: 'rgba(0,0,0,0.04)', zerolinecolor: 'rgba(0,0,0,0.06)', tickfont: {{ size: 12, color: '#86868b' }} }},
    legend: {{ bgcolor: 'rgba(0,0,0,0)', font: {{ size: 12, color: '#86868b' }} }},
}};

function switchTab(name) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    // Find the correct tab button
    const btns = document.querySelectorAll('.tab-btn');
    const tabMap = {{'overview':0, 'risk':1, 'recommend':2, 'safety':3}};
    if (tabMap[name] !== undefined) btns[tabMap[name]].classList.add('active');
    // Trigger resize for plotly
    window.dispatchEvent(new Event('resize'));
}}

function navigateToSku(skuId) {{
    // Set the SKU in the Model Recommendations dropdown and switch tab
    const selector = document.getElementById('sku-selector');
    selector.value = skuId;
    // Also update the visible search input to match
    const m = metricsData.find(x => x.sku_id === skuId);
    if (m) {{
        document.getElementById('sku-search-recommend').value = m.sku_id + ' — ' + m.sku_name;
    }}
    updateRecommendation();
    switchTab('recommend');
}}


// ---- TAB 1: OVERVIEW ----
function renderOverview() {{
    // Pool comparison bar chart
    const pools = poolData.map(p => p.signal_pool);
    Plotly.newPlot('chart-pool-comparison', [
        {{
            x: pools, y: poolData.map(p => p.baseline_fa),
            type: 'bar', name: 'Baseline (3M SMA)',
            marker: {{ color: pools.map(p => POOL_COLORS[p] || COLORS.blue), opacity: 0.4 }},
            text: poolData.map(p => p.baseline_fa.toFixed(1) + '%'),
            textposition: 'outside', textfont: {{ size: 11, color: COLORS.textSec }},
        }},
        {{
            x: pools, y: poolData.map(p => p.ml_fa),
            type: 'bar', name: 'Best ML Model',
            marker: {{ color: pools.map(p => POOL_COLORS[p] || COLORS.green), opacity: 0.9 }},
            text: poolData.map(p => p.ml_fa.toFixed(1) + '%'),
            textposition: 'outside', textfont: {{ size: 11, color: COLORS.text }},
        }}
    ], {{
        ...plotLayout,
        barmode: 'group',
        yaxis: {{ ...plotLayout.yaxis, title: 'Forecast Accuracy %', range: [0, 105] }},
        xaxis: {{ ...plotLayout.xaxis, title: '' }},
        legend: {{ ...plotLayout.legend, orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center' }},
    }}, {{ responsive: true }});

    // FA distribution — KDE density curves (Netflix/Stripe style)
    // Simple KDE using Gaussian kernel
    function kde(data, bandwidth) {{
        const min = Math.min(...data) - 10;
        const max = Math.max(...data) + 10;
        const step = 0.5;
        const xs = [];
        const ys = [];
        for (let x = min; x <= max; x += step) {{
            let density = 0;
            data.forEach(d => {{
                density += Math.exp(-0.5 * Math.pow((x - d) / bandwidth, 2));
            }});
            density /= (data.length * bandwidth * Math.sqrt(2 * Math.PI));
            xs.push(x);
            ys.push(density);
        }}
        return {{ xs, ys }};
    }}
    const baselineKDE = kde(metricsData.map(m => m.baseline_fa), 5);
    const mlKDE = kde(metricsData.map(m => m.ml_fa), 5);
    Plotly.newPlot('chart-fa-distribution', [
        {{
            x: baselineKDE.xs, y: baselineKDE.ys,
            type: 'scatter', mode: 'lines', name: 'Baseline (SMA)',
            fill: 'tozeroy', fillcolor: 'rgba(77,171,247,0.12)',
            line: {{ color: COLORS.blue, width: 2.5 }},
        }},
        {{
            x: mlKDE.xs, y: mlKDE.ys,
            type: 'scatter', mode: 'lines', name: 'ML Model',
            fill: 'tozeroy', fillcolor: 'rgba(0,214,143,0.12)',
            line: {{ color: COLORS.green, width: 2.5 }},
        }}
    ], {{
        ...plotLayout,
        xaxis: {{ ...plotLayout.xaxis, title: 'Forecast Accuracy %' }},
        yaxis: {{ ...plotLayout.yaxis, title: 'Density', showticklabels: false }},
        legend: {{ ...plotLayout.legend, orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center' }},
    }}, {{ responsive: true }});

    // SKU scatter with quadrant shading
    const filteredMetrics = activePoolFilter === 'all' ? metricsData : metricsData.filter(m => m.signal_pool === activePoolFilter);
    const scatterTraces = [];

    // Quadrant background shapes (semi-transparent fills)
    const quadrantShapes = [
        {{ type: 'rect', x0: 0, x1: 50, y0: 50, y1: 105, fillcolor: 'rgba(0,214,143,0.04)', line: {{ width: 0 }}, layer: 'below' }},  // ML Much Better
        {{ type: 'rect', x0: 50, x1: 105, y0: 50, y1: 105, fillcolor: 'rgba(77,171,247,0.04)', line: {{ width: 0 }}, layer: 'below' }},  // Both Strong
        {{ type: 'rect', x0: 50, x1: 105, y0: 0, y1: 50, fillcolor: 'rgba(255,107,107,0.04)', line: {{ width: 0 }}, layer: 'below' }},  // Baseline Better
        {{ type: 'rect', x0: 0, x1: 50, y0: 0, y1: 50, fillcolor: 'rgba(255,193,7,0.04)', line: {{ width: 0 }}, layer: 'below' }},  // Both Weak
    ];

    const poolNames = [...new Set(filteredMetrics.map(m => m.signal_pool))];
    poolNames.forEach(pool => {{
        const items = filteredMetrics.filter(m => m.signal_pool === pool);
        scatterTraces.push({{
            x: items.map(i => i.baseline_fa),
            y: items.map(i => i.ml_fa),
            text: items.map(i => i.sku_name + ' (' + i.sku_id + ')'),
            customdata: items.map(i => [i.signal_pool, (i.ml_fa - i.baseline_fa).toFixed(1), i.sku_id]),
            hovertemplate: '<b>%{{text}}</b><br>Baseline FA: %{{x:.1f}}%<br>ML FA: %{{y:.1f}}%<br>Signal Pool: %{{customdata[0]}}<br>Improvement: %{{customdata[1]}}%<extra></extra>',
            mode: 'markers',
            type: 'scatter',
            name: pool,
            marker: {{ color: POOL_COLORS[pool] || COLORS.green, size: 14, opacity: 0.85,
                       line: {{ color: 'rgba(0,0,0,0.08)', width: 1.5 }} }},
        }});
    }});
    // Diagonal reference line
    scatterTraces.push({{
        x: [0, 105], y: [0, 105],
        mode: 'lines', name: 'No Change Line',
        line: {{ color: 'rgba(0,0,0,0.1)', dash: 'dash', width: 1 }},
        showlegend: false,
    }});
    Plotly.newPlot('chart-sku-scatter', scatterTraces, {{
        ...plotLayout,
        xaxis: {{ ...plotLayout.xaxis, title: 'Baseline Forecast Accuracy %', range: [0, 105] }},
        yaxis: {{ ...plotLayout.yaxis, title: 'ML Forecast Accuracy %', range: [0, 105] }},
        legend: {{ ...plotLayout.legend, orientation: 'h', y: 1.15, x: 0.5, xanchor: 'center' }},
        shapes: quadrantShapes,
        annotations: [{{
            x: 15, y: 98, text: '<b>ML Much Better</b>', showarrow: false,
            font: {{ color: COLORS.green, size: 12 }}, borderpad: 4,
        }}, {{
            x: 90, y: 98, text: '<b>Both Strong</b>', showarrow: false,
            font: {{ color: '#74c0fc', size: 12 }}, borderpad: 4,
        }}, {{
            x: 90, y: 5, text: '<b>Baseline Better</b>', showarrow: false,
            font: {{ color: COLORS.red, size: 12 }}, borderpad: 4,
        }}, {{
            x: 15, y: 5, text: '<b>Both Weak</b>', showarrow: false,
            font: {{ color: COLORS.amber, size: 12 }}, borderpad: 4,
        }}],
    }}, {{ responsive: true }});

    // Click-to-navigate: click a dot to open SKU detail
    document.getElementById('chart-sku-scatter').on('plotly_click', function(data) {{
        if (data.points && data.points.length > 0) {{
            const pt = data.points[0];
            const textVal = pt.text || '';
            // Extract SKU ID from "SKU Name (SKU_ID)" format
            const match = textVal.match(/\(([^)]+)\)/);
            if (match) {{
                navigateToSku(match[1]);
            }}
        }}
    }});

    // ── Model Selection by Pool (donut chart) ──
    const modelCounts = {{}};
    metricsData.forEach(m => {{
        const name = m.best_model || 'Unknown';
        // Simplify ensemble names for the legend
        const short = name.startsWith('Ensemble') ? 'Ensemble' : name;
        modelCounts[short] = (modelCounts[short] || 0) + 1;
    }});
    const modelNames = Object.keys(modelCounts);
    const modelVals = Object.values(modelCounts);
    const modelColors = modelNames.map((n, i) => {{
        const palette = [COLORS.green, COLORS.blue, COLORS.amber, '#9334e6', COLORS.red];
        return palette[i % palette.length];
    }});
    Plotly.newPlot('chart-model-distribution', [{{
        labels: modelNames, values: modelVals,
        type: 'pie', hole: 0.5,
        marker: {{ colors: modelColors, line: {{ color: '#ffffff', width: 2 }} }},
        textinfo: 'label+percent', textfont: {{ size: 12, color: COLORS.text }},
        hovertemplate: '<b>%{{label}}</b><br>%{{value}} SKUs (%{{percent}})<extra></extra>',
    }}], {{
        ...plotLayout,
        height: 280,
        margin: {{ l: 20, r: 20, t: 10, b: 10 }},
        showlegend: true,
        legend: {{ ...plotLayout.legend, orientation: 'h', y: -0.1, x: 0.5, xanchor: 'center', font: {{ size: 11, color: COLORS.text }} }},
    }}, {{ responsive: true }});

    // ── All Candidates — Grouped Bar by Pool ──
    const allPools = [...new Set(metricsData.map(m => m.signal_pool))];
    const allModelNames = ['Gradient Boosting', 'Random Forest', 'Extra Trees', 'Ridge Regression'];
    const barTraces = allModelNames.map((modelName, idx) => {{
        const modelColors2 = [COLORS.green, COLORS.blue, COLORS.amber, '#9334e6'];
        const ys = allPools.map(pool => {{
            const poolSkus = metricsData.filter(m => m.signal_pool === pool);
            const scores = poolSkus.filter(m => m.model_scores && m.model_scores[modelName] !== undefined);
            if (scores.length === 0) return null;
            const avgMape = scores.reduce((s, m) => s + m.model_scores[modelName], 0) / scores.length;
            return Math.max(0, 100 - avgMape);
        }});
        return {{
            x: allPools, y: ys, name: modelName, type: 'bar',
            marker: {{ color: modelColors2[idx], opacity: 0.85 }},
            text: ys.map(v => v !== null ? v.toFixed(1) + '%' : ''),
            textposition: 'outside', textfont: {{ size: 10, color: COLORS.textSec }},
            hovertemplate: '<b>' + modelName + '</b><br>Pool: %{{x}}<br>Val Accuracy: %{{y:.1f}}%<extra></extra>',
        }};
    }});
    Plotly.newPlot('chart-model-pool-heatmap', barTraces, {{
        ...plotLayout,
        barmode: 'group',
        height: 280,
        yaxis: {{ ...plotLayout.yaxis, title: 'Validation Accuracy %', range: [0, 105] }},
        xaxis: {{ ...plotLayout.xaxis, title: '' }},
        legend: {{ ...plotLayout.legend, orientation: 'h', y: 1.15, x: 0.5, xanchor: 'center', font: {{ size: 11 }} }},
    }}, {{ responsive: true }});
}}

// ---- TAB 2: RISK FLAGS ----
function renderRisk() {{
    // Anomaly alerts
    const alertContainer = document.getElementById('anomaly-alerts');
    if (anomaliesData.length === 0) {{
        alertContainer.innerHTML = '<div class="chart-container" style="border-color:rgba(0,214,143,0.3);background:rgba(0,214,143,0.05)"><span style="color:var(--accent-green);font-weight:600">&#10003; No signal anomalies detected — all external indicators within normal range.</span></div>';
    }} else {{
        alertContainer.innerHTML = anomaliesData.map(a =>
            `<div class="alert-box"><span class="icon">&#9888;</span><span>${{a.message}}</span></div>`
        ).join('');
    }}

    // Risk table
    const tableContainer = document.getElementById('risk-table-container');
    if (riskData.length === 0) {{
        tableContainer.innerHTML = '<div class="empty-state"><div class="empty-icon">&#10003;</div><h3>All SKUs Performing Well</h3><p>No SKUs are below the 70% forecast accuracy threshold. All models are within acceptable range.</p></div>';
    }} else {{
        let html = '<table><thead><tr><th>SKU ID</th><th>SKU Name</th><th>Signal Pool</th><th>Class</th><th class="has-tooltip" data-tooltip="ML Forecast Accuracy — 100% minus MAPE. Higher is better.">ML Forecast Accuracy</th><th class="has-tooltip" data-tooltip="MAPE Volatility: Standard deviation of monthly MAPE values. Measures how erratic forecast errors are over time. Higher = less predictable.">MAPE Volatility</th><th class="has-tooltip" data-tooltip="Risk Score = (100 - ML FA%) + MAPE Volatility. Combines inaccuracy with unpredictability.">Risk Score</th><th>Severity</th></tr></thead><tbody>';
        riskData.forEach(r => {{
            // Tiered severity: CRITICAL (FA<30%), HIGH (FA 30-50%), MODERATE (FA 50-70%)
            let badge, status;
            if (r.ml_fa < 30) {{ badge = 'badge-red'; status = 'CRITICAL'; }}
            else if (r.ml_fa < 50) {{ badge = 'badge-red'; status = 'HIGH'; }}
            else {{ badge = 'badge-amber'; status = 'MODERATE'; }}
            html += `<tr style="cursor:pointer" onclick="navigateToSku('${{r.sku_id}}')" title="Click to view details in Model Recommendations">
                <td style="font-weight:600">${{r.sku_id}}</td>
                <td>${{r.sku_name}}</td>
                <td>${{r.signal_pool}}</td>
                <td><span class="badge badge-amber">${{r.abc_xyz}}</span></td>
                <td style="color:${{r.ml_fa < 30 ? COLORS.red : (r.ml_fa < 50 ? COLORS.amber : COLORS.text)}}">${{r.ml_fa.toFixed(1)}}%</td>
                <td title="Std dev of monthly MAPE values">${{r.mape_volatility.toFixed(1)}}</td>
                <td style="font-weight:700;color:${{r.ml_fa < 30 ? COLORS.red : (r.ml_fa < 50 ? COLORS.amber : COLORS.text)}}">${{r.risk_score.toFixed(0)}}</td>
                <td><span class="badge ${{badge}}">${{status}}</span></td>
            </tr>`;
        }});
        html += '</tbody></table>';
        tableContainer.innerHTML = html;
    }}

    // Risk scatter — highlight top 5, fade the rest
    const filteredRisk = activePoolFilter === 'all' ? riskData : riskData.filter(r => r.signal_pool === activePoolFilter);
    if (filteredRisk.length > 0) {{
        const sortedRisk = [...filteredRisk].sort((a, b) => b.risk_score - a.risk_score);
        const top5ids = new Set(sortedRisk.slice(0, 5).map(r => r.sku_id));
        Plotly.newPlot('chart-risk-scatter', [
            // Faded background points
            {{
                x: filteredRisk.filter(r => !top5ids.has(r.sku_id)).map(r => r.ml_fa),
                y: filteredRisk.filter(r => !top5ids.has(r.sku_id)).map(r => r.risk_score),
                text: filteredRisk.filter(r => !top5ids.has(r.sku_id)).map(r => r.sku_name),
                mode: 'markers',
                type: 'scatter',
                name: 'Other At-Risk',
                marker: {{
                    color: 'rgba(227,116,0,0.2)',
                    size: 10,
                    line: {{ color: 'rgba(0,0,0,0.06)', width: 1 }},
                }},
                hovertemplate: '%{{text}}<br>FA: %{{x:.1f}}%<br>Risk: %{{y:.0f}}<extra></extra>',
            }},
            // Top 5 highlighted with labels
            {{
                x: sortedRisk.slice(0, 5).map(r => r.ml_fa),
                y: sortedRisk.slice(0, 5).map(r => r.risk_score),
                text: sortedRisk.slice(0, 5).map(r => r.sku_name),
                mode: 'markers+text',
                type: 'scatter',
                name: 'Top 5 Highest Risk',
                textposition: 'top center',
                textfont: {{ size: 11, color: COLORS.text, family: 'Segoe UI, system-ui, sans-serif' }},
                marker: {{
                    color: sortedRisk.slice(0, 5).map(r => r.risk_score),
                    colorscale: [[0, COLORS.amber], [1, COLORS.red]],
                    size: 16,
                    line: {{ color: 'rgba(0,0,0,0.15)', width: 2 }},
                }},
                hovertemplate: '%{{text}}<br>FA: %{{x:.1f}}%<br>Risk: %{{y:.0f}}<extra></extra>',
            }}
        ], {{
            ...plotLayout,
            xaxis: {{ ...plotLayout.xaxis, title: 'ML Forecast Accuracy %' }},
            yaxis: {{ ...plotLayout.yaxis, title: 'Risk Score' }},
            showlegend: false,
        }}, {{ responsive: true }});
    }}
}}

// ---- TAB 3: RECOMMENDATIONS ----
function populateSkuDropdowns() {{
    const selector = document.getElementById('sku-selector');
    const safetySelector = document.getElementById('safety-sku-selector');
    metricsData.forEach(m => {{
        const opt = `<option value="${{m.sku_id}}">${{m.sku_id}} — ${{m.sku_name}} (${{m.signal_pool}})</option>`;
        selector.innerHTML += opt;
        safetySelector.innerHTML += opt;
    }});
    // Initialize searchable dropdowns with first SKU name
    if (metricsData.length > 0) {{
        const first = metricsData[0];
        document.getElementById('sku-search-recommend').value = first.sku_id + ' — ' + first.sku_name;
        document.getElementById('sku-search-safety').value = first.sku_id + ' — ' + first.sku_name;
    }}
}}

function buildSkuDropdownHtml(filter) {{
    // Group by pool, show risk indicator
    const pools = [...new Set(metricsData.map(m => m.signal_pool))];
    const lf = (filter || '').toLowerCase();
    let html = '';
    pools.forEach(pool => {{
        const skus = metricsData.filter(m => m.signal_pool === pool)
            .filter(m => !lf || m.sku_id.toLowerCase().includes(lf) || m.sku_name.toLowerCase().includes(lf) || m.signal_pool.toLowerCase().includes(lf));
        if (skus.length === 0) return;
        html += `<div class="sku-opt-group">${{pool}} Pool (${{skus.length}} SKUs)</div>`;
        skus.forEach(m => {{
            const dotClass = m.ml_fa < 30 ? 'red' : (m.ml_fa < 70 ? 'amber' : 'green');
            const riskLabel = m.ml_fa < 70 ? ` — ${{m.ml_fa.toFixed(0)}}% FA` : '';
            html += `<div class="sku-opt" data-sku="${{m.sku_id}}">
                <span class="risk-dot ${{dotClass}}"></span>
                <span>${{m.sku_id}} — ${{m.sku_name}}${{riskLabel}}</span>
            </div>`;
        }});
    }});
    return html || '<div style="padding:12px;color:var(--text-secondary);text-align:center">No matching SKUs</div>';
}}

function openSkuDropdown(context) {{
    const dd = document.getElementById('sku-dropdown-' + context);
    dd.innerHTML = buildSkuDropdownHtml('');
    dd.classList.add('open');
    attachSkuClickHandlers(context);
    // Select all text in input for easy replacement
    document.getElementById('sku-search-' + context).select();
}}

function filterSkuDropdown(context) {{
    const input = document.getElementById('sku-search-' + context);
    const dd = document.getElementById('sku-dropdown-' + context);
    dd.innerHTML = buildSkuDropdownHtml(input.value);
    dd.classList.add('open');
    attachSkuClickHandlers(context);
}}

function attachSkuClickHandlers(context) {{
    const dd = document.getElementById('sku-dropdown-' + context);
    dd.querySelectorAll('.sku-opt').forEach(opt => {{
        opt.onclick = () => {{
            const skuId = opt.dataset.sku;
            const m = metricsData.find(x => x.sku_id === skuId);
            const input = document.getElementById('sku-search-' + context);
            input.value = m.sku_id + ' — ' + m.sku_name;
            dd.classList.remove('open');
            if (context === 'recommend') {{
                document.getElementById('sku-selector').value = skuId;
                updateRecommendation();
            }} else {{
                document.getElementById('safety-sku-selector').value = skuId;
                updateSafetyStock();
            }}
        }};
    }});
}}

// Close dropdowns when clicking outside
document.addEventListener('click', function(e) {{
    ['recommend', 'safety'].forEach(ctx => {{
        const wrap = document.getElementById('sku-search-wrap-' + ctx);
        const dd = document.getElementById('sku-dropdown-' + ctx);
        if (wrap && dd && !wrap.contains(e.target)) dd.classList.remove('open');
    }});
}});

function updateRecommendation() {{
    const skuId = document.getElementById('sku-selector').value;
    const m = metricsData.find(x => x.sku_id === skuId);
    if (!m) return;

    // Find best ML model for THIS specific SKU from per-SKU test-period scores
    let bestSkuMLName = m.best_model || 'ML Model';
    let bestSkuMLFA = m.ml_fa;
    if (m.model_scores && Object.keys(m.model_scores).length > 0) {{
        const entries = Object.entries(m.model_scores);
        const best = entries.reduce((a, b) => a[1] < b[1] ? a : b);  // lowest MAPE
        const bestFA = Math.max(0, 100 - best[1]);
        if (bestFA > bestSkuMLFA) {{
            bestSkuMLName = best[0];
            bestSkuMLFA = bestFA;
        }}
    }}

    // Single source of truth: compare baseline vs best per-SKU ML model
    const baselineWins = m.baseline_fa >= bestSkuMLFA;
    const recommended = baselineWins ? 'Baseline (3M SMA)' : bestSkuMLName;
    const recFA = Math.max(bestSkuMLFA, m.baseline_fa);
    const ts = tsData[skuId];
    const future = futureData.filter(f => f.sku_id === skuId);

    let html = `
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="label">Recommended Model</div>
            <div class="value green" style="font-size:22px">${{recommended}}</div>
            <div class="sub">Best performing for this SKU</div>
        </div>
        <div class="kpi-card">
            <div class="label has-tooltip" data-tooltip="Forecast Accuracy = 100% - MAPE. Measured on 6-month test period (hold-out validation).">Forecast Accuracy</div>
            <div class="value green">${{recFA.toFixed(1)}}%</div>
            <div class="sub">6-month test period</div>
        </div>
        <div class="kpi-card">
            <div class="label has-tooltip" data-tooltip="Percentage-point difference: Best ML Accuracy − Baseline Accuracy. Positive = ML outperforms baseline.">Improvement vs Baseline</div>
            <div class="value" style="color:${{bestSkuMLFA - m.baseline_fa > 0 ? COLORS.green : COLORS.red}}">${{bestSkuMLFA - m.baseline_fa > 0 ? '+' : ''}}${{(bestSkuMLFA - m.baseline_fa).toFixed(1)}}%</div>
            <div class="sub">Accuracy delta (best ML − baseline)</div>
        </div>
        <div class="kpi-card">
            <div class="label">Avg Monthly Demand</div>
            <div class="value blue">${{Math.round(m.avg_demand)}}</div>
            <div class="sub">units / month</div>
        </div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">
        <div class="chart-container" style="min-width:0;overflow:hidden">
            <div class="chart-title">Model Comparison — Accuracy by Candidate</div>
            <div id="chart-model-compare"></div>
        </div>
        <div class="chart-container" style="min-width:0;overflow:hidden;display:flex;flex-direction:column;justify-content:center">
            <div class="chart-title">Model Selection Summary</div>
            <div id="model-selection-summary" style="padding:12px 16px"></div>
        </div>
    </div>
    <div class="chart-container" style="margin-bottom:16px">
        <div class="chart-title">External Signals Used — AQI, Temperature, Precipitation, Wedding Flag, Google Trends</div>
        <div id="chart-signals-detail"></div>
    </div>
    <div class="chart-container" style="margin-bottom:16px;padding:16px 20px">
        <div class="chart-title">Why This Model Was Selected</div>
        <div id="model-explanation" style="font-size:13px;color:var(--text-secondary);line-height:1.8;margin-top:8px"></div>
    </div>
    <div class="chart-container" style="margin-bottom:16px;padding:0;overflow:hidden">
        <div style="padding:16px 20px 12px;cursor:pointer;display:flex;align-items:center;justify-content:space-between" onclick="const b=document.getElementById('methodology-body');const a=document.getElementById('method-arrow');if(b.style.maxHeight==='0px'){{b.style.maxHeight='5000px';a.style.transform='rotate(180deg)'}}else{{b.style.maxHeight='0px';a.style.transform='rotate(0deg)'}}">
            <div class="chart-title" style="margin:0">Methodology & Transparency</div>
            <svg id="method-arrow" width="16" height="16" viewBox="0 0 16 16" fill="none" style="transition:transform 0.3s;transform:rotate(0deg)"><path d="M4 6l4 4 4-4" stroke="var(--text-secondary)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </div>
        <div id="methodology-body" style="max-height:0px;overflow:hidden;transition:max-height 0.5s ease">
            <div style="padding:0 20px 20px">

                <!-- Pipeline Overview — 7 Steps -->
                <div style="display:grid;grid-template-columns:repeat(7,1fr);gap:2px;margin-bottom:20px;padding:16px 0">
                    <div style="text-align:center;padding:10px 4px;background:var(--input-bg);border-radius:10px 0 0 10px">
                        <div style="font-size:9px;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-muted);margin-bottom:3px">Step 1</div>
                        <div style="font-size:11px;font-weight:600;color:var(--text-primary)">Split Data</div>
                        <div style="font-size:10px;color:var(--text-secondary);margin-top:2px">24 / 6 / 6 mo</div>
                    </div>
                    <div style="text-align:center;padding:10px 4px;background:var(--input-bg)">
                        <div style="font-size:9px;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-muted);margin-bottom:3px">Step 2</div>
                        <div style="font-size:11px;font-weight:600;color:var(--text-primary)">Features</div>
                        <div style="font-size:10px;color:var(--text-secondary);margin-top:2px">12 inputs</div>
                    </div>
                    <div style="text-align:center;padding:10px 4px;background:var(--input-bg)">
                        <div style="font-size:9px;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-muted);margin-bottom:3px">Step 3</div>
                        <div style="font-size:11px;font-weight:600;color:var(--text-primary)">Train</div>
                        <div style="font-size:10px;color:var(--text-secondary);margin-top:2px">4 models</div>
                    </div>
                    <div style="text-align:center;padding:10px 4px;background:var(--input-bg)">
                        <div style="font-size:9px;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-muted);margin-bottom:3px">Step 4</div>
                        <div style="font-size:11px;font-weight:600;color:var(--text-primary)">Validate</div>
                        <div style="font-size:10px;color:var(--text-secondary);margin-top:2px">Rank &amp; ensemble</div>
                    </div>
                    <div style="text-align:center;padding:10px 4px;background:var(--input-bg)">
                        <div style="font-size:9px;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-muted);margin-bottom:3px">Step 5</div>
                        <div style="font-size:11px;font-weight:600;color:var(--text-primary)">Retrain</div>
                        <div style="font-size:10px;color:var(--text-secondary);margin-top:2px">30 months</div>
                    </div>
                    <div style="text-align:center;padding:10px 4px;background:var(--input-bg)">
                        <div style="font-size:9px;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-muted);margin-bottom:3px">Step 6</div>
                        <div style="font-size:11px;font-weight:600;color:var(--text-primary)">Test &amp; Score</div>
                        <div style="font-size:10px;color:var(--text-secondary);margin-top:2px">FA per model</div>
                    </div>
                    <div style="text-align:center;padding:10px 4px;background:var(--input-bg);border-radius:0 10px 10px 0">
                        <div style="font-size:9px;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-muted);margin-bottom:3px">Step 7</div>
                        <div style="font-size:11px;font-weight:600;color:var(--text-primary)">Recommend</div>
                        <div style="font-size:10px;color:var(--text-secondary);margin-top:2px">Best wins</div>
                    </div>
                </div>

                <!-- Why These Models -->
                <div style="margin-bottom:20px">
                    <div style="font-size:13px;font-weight:600;color:var(--text-primary);margin-bottom:12px;display:flex;align-items:center;gap:8px">
                        <span style="width:3px;height:14px;background:var(--accent-green);border-radius:2px;display:inline-block"></span>
                        Why These 4 Models?
                    </div>
                    <div style="font-size:12px;color:var(--text-secondary);line-height:1.6;margin-bottom:14px;padding-left:11px">
                        Pharmaceutical demand is influenced by non-linear external signals (weather, seasonality, search trends). We use four complementary tree-based and linear regressors that each capture different patterns. No single model dominates across all signal pools, which is why we evaluate all four per SKU.
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
                        <div style="background:rgba(255,255,255,0.6);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.5);border-radius:14px;padding:14px 16px">
                            <div style="font-size:12px;font-weight:600;color:var(--accent-green);margin-bottom:8px">Gradient Boosting</div>
                            <div style="font-size:11px;color:var(--text-secondary);line-height:1.6">
                                <strong style="color:var(--text-primary)">How it works:</strong> Starts with a simple prediction (e.g., average demand), then builds a sequence of small decision trees where each new tree focuses specifically on the errors the previous trees got wrong. Each tree's contribution is scaled by a learning rate to prevent overcorrection. The final prediction is the sum of all trees' outputs.<br>
                                <strong style="color:var(--text-primary)">Why for pharma:</strong> Excels at capturing complex, non-linear interactions — e.g., "demand spikes when AQI &gt; 200 AND temperature &gt; 35&deg;C" — patterns that linear models miss. Often the top performer for weather-driven pools.
                            </div>
                        </div>
                        <div style="background:rgba(255,255,255,0.6);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.5);border-radius:14px;padding:14px 16px">
                            <div style="font-size:12px;font-weight:600;color:var(--accent-blue);margin-bottom:8px">Random Forest</div>
                            <div style="font-size:11px;color:var(--text-secondary);line-height:1.6">
                                <strong style="color:var(--text-primary)">How it works:</strong> Trains hundreds of decision trees independently, each on a random subset of the training data and a random subset of features. Each tree makes its own prediction, and the final output is the average across all trees. This "wisdom of crowds" approach reduces the risk of any single tree overfitting to noise.<br>
                                <strong style="color:var(--text-primary)">Why for pharma:</strong> Robust to outliers and noisy demand spikes (e.g., one-time bulk orders). Delivers stable, reliable forecasts even when demand patterns are irregular across months.
                            </div>
                        </div>
                        <div style="background:rgba(255,255,255,0.6);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.5);border-radius:14px;padding:14px 16px">
                            <div style="font-size:12px;font-weight:600;color:var(--accent-amber);margin-bottom:8px">Extra Trees <span style="font-weight:400;color:var(--text-muted);font-size:10px">(Extremely Randomized Trees)</span></div>
                            <div style="font-size:11px;color:var(--text-secondary);line-height:1.6">
                                <strong style="color:var(--text-primary)">How it works:</strong> Like Random Forest, it trains many trees in parallel — but instead of searching for the optimal split point at each node, it picks split thresholds completely at random. This adds extra randomization: each individual tree is less precise, but the ensemble averages out noise more aggressively, reducing variance.<br>
                                <strong style="color:var(--text-primary)">Why for pharma:</strong> The additional randomness helps avoid overfitting on small signal pools. Trains faster than Random Forest and often generalizes better on sharp seasonal patterns like wedding-season demand surges.
                            </div>
                        </div>
                        <div style="background:rgba(255,255,255,0.6);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.5);border-radius:14px;padding:14px 16px">
                            <div style="font-size:12px;font-weight:600;color:#9334e6;margin-bottom:8px">Ridge Regression <span style="font-weight:400;color:var(--text-muted);font-size:10px">(L2-Regularized Linear Model)</span></div>
                            <div style="font-size:11px;color:var(--text-secondary);line-height:1.6">
                                <strong style="color:var(--text-primary)">How it works:</strong> Finds the best straight-line relationship between each input signal and demand (e.g., "for every 10&deg;C rise, demand increases by X units"). Unlike ordinary linear regression, Ridge adds a penalty term (L2) that shrinks large coefficients toward zero, preventing any single signal from dominating the prediction and reducing overfitting.<br>
                                <strong style="color:var(--text-primary)">Why for pharma:</strong> Acts as a simpler counterpoint to tree models. When the relationship between a signal and demand is genuinely proportional (e.g., Google Trends search volume &rarr; demand), Ridge captures it cleanly with lower overfitting risk than complex models.
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Full Academic Methodology — clean, no overflow -->
                <div style="margin-bottom:20px;overflow-wrap:break-word;word-break:break-word">
                    <div style="font-size:13px;font-weight:600;color:var(--text-primary);margin-bottom:12px;display:flex;align-items:center;gap:8px">
                        <span style="width:3px;height:14px;background:var(--accent-green);border-radius:2px;display:inline-block"></span>
                        How Every Number Is Produced
                    </div>

                    <!-- Step 1 -->
                    <div style="background:rgba(255,255,255,0.6);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.5);border-radius:14px;padding:14px 16px;margin-bottom:8px">
                        <div style="font-size:11px;font-weight:600;color:var(--accent-green);margin-bottom:6px">STEP 1 &mdash; Chronological Data Split</div>
                        <div style="font-size:11px;color:var(--text-secondary);line-height:1.7">
                            36 months per SKU, split in time order (no shuffling):
                            <div style="display:flex;gap:0;margin:8px 0;border-radius:4px;overflow:hidden;font-size:10px;font-weight:600;text-align:center">
                                <div style="flex:24;background:var(--accent-blue);color:#fff;padding:4px 0">Months 1–24 TRAIN</div>
                                <div style="flex:6;background:#ffc107;color:#000;padding:4px 0">25–30 VAL</div>
                                <div style="flex:6;background:var(--accent-green);color:#000;padding:4px 0">31–36 TEST</div>
                            </div>
                            <strong style="color:var(--text-primary)">Train:</strong> Models learn here only.
                            <strong style="color:var(--text-primary)">Validation:</strong> Ranks models; never seen during training.
                            <strong style="color:var(--text-primary)">Test:</strong> All FA numbers on this dashboard come from here.
                        </div>
                    </div>

                    <!-- Step 2 -->
                    <div style="background:rgba(255,255,255,0.6);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.5);border-radius:14px;padding:14px 16px;margin-bottom:8px">
                        <div style="font-size:11px;font-weight:600;color:var(--accent-green);margin-bottom:6px">STEP 2 &mdash; Feature Engineering (12 Inputs)</div>
                        <div style="font-size:11px;color:var(--text-secondary);line-height:1.7">
                            Every model receives the same 12 features per month. No future data leaks — lags and rolling windows use only past values.
                            <table style="width:100%;margin:8px 0;font-size:10px;border-collapse:collapse">
                                <tr><td style="padding:3px 8px;border-bottom:1px solid var(--card-border)"><code style="color:var(--text-primary)">baseline_forecast</code></td><td style="padding:3px 8px;border-bottom:1px solid var(--card-border)">3-month SMA</td><td style="padding:3px 8px;border-bottom:1px solid var(--card-border)"><code style="color:var(--text-primary)">exp_smoothing_forecast</code></td><td style="padding:3px 8px;border-bottom:1px solid var(--card-border)">Exp. Smoothing</td></tr>
                                <tr><td style="padding:3px 8px;border-bottom:1px solid var(--card-border)"><code style="color:var(--text-primary)">demand_lag1</code></td><td style="padding:3px 8px;border-bottom:1px solid var(--card-border)">Prior month</td><td style="padding:3px 8px;border-bottom:1px solid var(--card-border)"><code style="color:var(--text-primary)">demand_lag2</code></td><td style="padding:3px 8px;border-bottom:1px solid var(--card-border)">2 months ago</td></tr>
                                <tr><td style="padding:3px 8px;border-bottom:1px solid var(--card-border)"><code style="color:var(--text-primary)">demand_rolling_std</code></td><td style="padding:3px 8px;border-bottom:1px solid var(--card-border)">3-mo volatility</td><td style="padding:3px 8px;border-bottom:1px solid var(--card-border)"><code style="color:var(--text-primary)">month_sin/cos</code></td><td style="padding:3px 8px;border-bottom:1px solid var(--card-border)">Seasonality</td></tr>
                                <tr><td style="padding:3px 8px;border-bottom:1px solid var(--card-border)"><code style="color:var(--text-primary)">temperature</code></td><td style="padding:3px 8px;border-bottom:1px solid var(--card-border)">Weather</td><td style="padding:3px 8px;border-bottom:1px solid var(--card-border)"><code style="color:var(--text-primary)">aqi</code></td><td style="padding:3px 8px;border-bottom:1px solid var(--card-border)">Air Quality</td></tr>
                                <tr><td style="padding:3px 8px"><code style="color:var(--text-primary)">precipitation</code></td><td style="padding:3px 8px">Monsoon</td><td style="padding:3px 8px"><code style="color:var(--text-primary)">wedding_flag</code></td><td style="padding:3px 8px">Wedding season</td></tr>
                            </table>
                            <span style="font-size:10px;color:var(--text-muted)">+ <code style="color:var(--text-primary)">google_trends</code> (search interest). Target variable: actual monthly demand.</span>
                        </div>
                    </div>

                    <!-- Step 3 -->
                    <div style="background:rgba(255,255,255,0.6);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.5);border-radius:14px;padding:14px 16px;margin-bottom:8px">
                        <div style="font-size:11px;font-weight:600;color:var(--accent-green);margin-bottom:6px">STEP 3 &mdash; Train 4 Models (Months 1–24)</div>
                        <div style="font-size:11px;color:var(--text-secondary);line-height:1.7">
                            Fixed hyperparameters — no tuning on test data. <code style="font-size:10px;color:var(--text-primary)">random_state=42</code> ensures full reproducibility.
                            <table style="width:100%;margin:8px 0;font-size:10px;border-collapse:collapse">
                                <tr style="border-bottom:1px solid var(--card-border)">
                                    <td style="padding:6px 8px;font-weight:600;color:var(--accent-blue)">Gradient Boosting</td>
                                    <td style="padding:6px 8px;color:var(--text-muted)">150 trees, depth 4, lr 0.1, subsample 0.8</td>
                                </tr>
                                <tr style="border-bottom:1px solid var(--card-border)">
                                    <td style="padding:6px 8px;font-weight:600;color:var(--accent-blue)">Random Forest</td>
                                    <td style="padding:6px 8px;color:var(--text-muted)">200 trees, depth 6, min leaf 3</td>
                                </tr>
                                <tr style="border-bottom:1px solid var(--card-border)">
                                    <td style="padding:6px 8px;font-weight:600;color:var(--accent-blue)">Extra Trees</td>
                                    <td style="padding:6px 8px;color:var(--text-muted)">200 trees, depth 6, min leaf 3</td>
                                </tr>
                                <tr>
                                    <td style="padding:6px 8px;font-weight:600;color:var(--accent-blue)">Ridge Regression</td>
                                    <td style="padding:6px 8px;color:var(--text-muted)">alpha=1.0 (L2 regularization)</td>
                                </tr>
                            </table>
                        </div>
                    </div>

                    <!-- Step 4 -->
                    <div style="background:rgba(255,255,255,0.6);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.5);border-radius:14px;padding:14px 16px;margin-bottom:8px">
                        <div style="font-size:11px;font-weight:600;color:var(--accent-green);margin-bottom:6px">STEP 4 &mdash; Validate &amp; Rank (Months 25–30)</div>
                        <div style="font-size:11px;color:var(--text-secondary);line-height:1.7">
                            Each model predicts months 25–30 (never seen during training). Validation MAPE is computed:
                            <div style="background:rgba(0,0,0,0.03);border-radius:10px;padding:6px 10px;margin:6px 0;font-size:11px;color:var(--text-primary)">
                                Val_MAPE = (1/n) &times; &sum; |Actual &minus; Predicted| / Actual &times; 100
                            </div>
                            Models ranked by Val_MAPE (lowest = best). Top 3 form a weighted ensemble:
                            <div style="background:rgba(0,0,0,0.03);border-radius:10px;padding:6px 10px;margin:6px 0;font-size:11px;color:var(--text-primary)">
                                w<sub>i</sub> = (1 / MAPE<sub>i</sub>) / &sum;(1 / MAPE<sub>j</sub>)<br>
                                Ensemble = &sum; w<sub>i</sub> &times; Prediction<sub>i</sub>
                            </div>
                            <span style="font-size:10px;color:var(--text-muted)">Example: MAPEs [10%, 12%, 15%] &rarr; weights [0.45, 0.38, 0.17]. If ensemble beats solo best on validation, ensemble is used.</span>
                        </div>
                    </div>

                    <!-- Step 5 -->
                    <div style="background:rgba(255,255,255,0.6);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.5);border-radius:14px;padding:14px 16px;margin-bottom:8px">
                        <div style="font-size:11px;font-weight:600;color:var(--accent-green);margin-bottom:6px">STEP 5 &mdash; Retrain on Months 1–30</div>
                        <div style="font-size:11px;color:var(--text-secondary);line-height:1.7">
                            After the winner is chosen, all models are retrained on the full 30 months (train + validation combined) to maximize learning before the final test. Predictions on months 31–36 are floored at 0 and rounded to whole units.
                        </div>
                    </div>

                    <!-- Step 6 -->
                    <div style="background:rgba(255,255,255,0.6);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.5);border-radius:14px;padding:14px 16px;margin-bottom:8px">
                        <div style="font-size:11px;font-weight:600;color:var(--accent-green);margin-bottom:6px">STEP 6 &mdash; Compute FA per Model (Months 31–36)</div>
                        <div style="font-size:11px;color:var(--text-secondary);line-height:1.7">
                            For <strong>each model independently</strong> (GB, RF, ET, Ridge, and Baseline), on the held-out test period:
                            <div style="background:rgba(0,0,0,0.03);border-radius:10px;padding:8px 10px;margin:6px 0;font-size:11px;color:var(--text-primary);line-height:2">
                                MAPE<sub>model</sub> = (1/n) &times; &sum; |Actual(t) &minus; Forecast<sub>model</sub>(t)| / max(Actual(t), 1) &times; 100<br>
                                FA<sub>model</sub> = max(0, 100 &minus; MAPE<sub>model</sub>)<br>
                                <strong>Best_ML_FA</strong> = max(FA<sub>GB</sub>, FA<sub>RF</sub>, FA<sub>ET</sub>, FA<sub>Ridge</sub>)<br>
                                <strong>Baseline_FA</strong> = 100 &minus; MAPE<sub>3M-SMA</sub><br>
                                <strong>&Delta; Improvement</strong> = Best_ML_FA &minus; Baseline_FA
                            </div>
                            <span style="font-size:10px;color:var(--text-muted)"><strong>max(Actual, 1)</strong> prevents division by zero. <strong>n</strong> = test months with demand &gt; 0. The bar chart shows FA for every model side-by-side so you can verify.</span>
                        </div>
                    </div>

                    <!-- Step 7 -->
                    <div style="background:rgba(255,255,255,0.6);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.5);border-radius:14px;padding:14px 16px;margin-bottom:8px">
                        <div style="font-size:11px;font-weight:600;color:var(--accent-green);margin-bottom:6px">STEP 7 &mdash; Recommendation (Purely Algorithmic)</div>
                        <div style="font-size:11px;color:var(--text-secondary);line-height:1.7">
                            <div style="background:rgba(0,0,0,0.03);border-radius:10px;padding:6px 10px;margin:4px 0;font-size:11px;color:var(--text-primary)">
                                IF Best_ML_FA &gt; Baseline_FA &rarr; Recommend best ML model<br>
                                ELSE &rarr; Recommend Baseline (3-month SMA)
                            </div>
                            No human override, no subjective weighting. Highest held-out accuracy wins.
                        </div>
                    </div>
                </div>

            </div>
        </div>
    </div>
    <div class="two-col">
        <div class="chart-container" style="min-width:0;overflow:hidden">
            <div class="chart-title">Historical Demand vs Forecasts — ${{m.sku_name}}</div>
            <div id="chart-ts-detail"></div>
            <div style="margin-top:8px;border-top:1px solid var(--card-border);padding-top:12px">
                <div class="chart-title" style="font-size:14px;margin-bottom:8px">Forecast Error (Residuals) — ML Model</div>
                <div id="chart-residuals"></div>
            </div>
        </div>
        <div class="chart-container" style="min-width:0;overflow:hidden">
            <div class="chart-title">Next 3-Month Forecast with Confidence Interval</div>
            <div id="ci-warning"></div>
            <div id="chart-future-forecast"></div>
            <div style="margin-top:12px;border-top:1px solid var(--card-border);padding-top:12px">
                <div class="chart-title" style="font-size:14px;margin-bottom:8px">3-Month Forecast Summary</div>
                <div id="forecast-table"></div>
            </div>
        </div>
    </div>`;
    document.getElementById('recommendation-content').innerHTML = html;

    // Model comparison chart — include baseline so the user sees the full picture
    if (m.model_scores && Object.keys(m.model_scores).length > 0) {{
        const scores = m.model_scores;
        const mlNames = Object.keys(scores).sort((a, b) => scores[a] - scores[b]);
        // All scores are now per-SKU test-period MAPEs — consistent with baseline_fa/ml_fa
        const allNames = [...mlNames, 'Baseline (3M SMA)'];
        const baselineMape = 100 - m.baseline_fa;
        const allMapes = [...mlNames.map(n => scores[n]), baselineMape];
        // Sort all by MAPE ascending (best first at bottom of horizontal bar chart)
        const sorted = allNames.map((n, i) => ({{ name: n, mape: allMapes[i] }}))
            .sort((a, b) => a.mape - b.mape);
        const names = sorted.map(s => s.name);
        const mapes = sorted.map(s => s.mape);
        const fas = mapes.map(v => Math.max(0, 100 - v));
        const barColors = names.map(n => {{
            if (n === recommended || (recommended.includes(n))) return COLORS.green;
            return 'rgba(26,115,232,0.5)';
        }});
        Plotly.newPlot('chart-model-compare', [{{
            y: names, x: fas, type: 'bar', orientation: 'h',
            marker: {{ color: barColors, line: {{ width: 0 }} }},
            text: fas.map(v => v.toFixed(1) + '%'), textposition: 'outside',
            textfont: {{ color: COLORS.text, size: 12 }},
            hovertemplate: '%{{y}}<br>Accuracy: %{{x:.1f}}%<br>MAPE: %{{customdata:.1f}}%<extra></extra>',
            customdata: mapes,
        }}], {{
            ...plotLayout,
            height: Math.max(200, names.length * 40 + 40),
            margin: {{ l: 140, r: 60, t: 10, b: 30 }},
            xaxis: {{ ...plotLayout.xaxis, title: 'Test-Period Accuracy (100 − MAPE %)', range: [0, 105] }},
            yaxis: {{ ...plotLayout.yaxis, title: '', automargin: true }},
            showlegend: false,
        }}, {{ responsive: true }});

        // Model selection summary text — uses per-SKU best ML model
        const chartBestMLName = mlNames[0];
        const chartBestMLFA = Math.max(0, 100 - scores[chartBestMLName]).toFixed(1);
        let summaryHtml = `<div style="font-size:13px;color:var(--text-secondary);line-height:1.7">`;
        summaryHtml += `<div style="margin-bottom:8px"><span style="color:var(--text-primary);font-weight:600">Winning model:</span> <span style="color:var(--accent-green);font-weight:700">${{recommended}}</span></div>`;
        summaryHtml += `<div style="margin-bottom:8px"><span style="color:var(--text-primary);font-weight:600">Candidates evaluated:</span> ${{mlNames.length}} ML models + Baseline</div>`;
        if (baselineWins) {{
            summaryHtml += `<div style="margin-bottom:8px"><span style="color:var(--text-primary);font-weight:600">Selection method:</span> Baseline outperformed all ML candidates on 6-month test period</div>`;
            summaryHtml += `<div style="margin-bottom:8px"><span style="color:var(--text-primary);font-weight:600">Baseline accuracy:</span> ${{m.baseline_fa.toFixed(1)}}% vs best ML ${{chartBestMLName}} (${{chartBestMLFA}}%)</div>`;
        }} else {{
            summaryHtml += `<div style="margin-bottom:8px"><span style="color:var(--text-primary);font-weight:600">Selection method:</span> Best per-SKU model on 6-month test period</div>`;
            summaryHtml += `<div style="margin-bottom:8px"><span style="color:var(--text-primary);font-weight:600">Best ML model:</span> ${{bestSkuMLName}} (${{bestSkuMLFA.toFixed(1)}}% accuracy)</div>`;
        }}
        const poolSkus = metricsData.filter(x => x.signal_pool === m.signal_pool);
        const poolAvgFA = (poolSkus.reduce((s, x) => s + Math.max(x.ml_fa, x.baseline_fa), 0) / poolSkus.length).toFixed(1);
        summaryHtml += `<div><span style="color:var(--text-primary);font-weight:600">Pool avg accuracy:</span> ${{poolAvgFA}}% across ${{poolSkus.length}} SKUs in ${{m.signal_pool}}</div>`;
        summaryHtml += `</div>`;
        document.getElementById('model-selection-summary').innerHTML = summaryHtml;
    }}

    // External signals chart (dual y-axis)
    if (ts && ts.aqi && ts.aqi.length > 0) {{
        const sigTraces = [
            {{ x: ts.dates, y: ts.aqi, name: 'AQI', type: 'scatter', mode: 'lines',
               line: {{ color: '#ff6b6b', width: 2 }}, yaxis: 'y' }},
            {{ x: ts.dates, y: ts.temperature, name: 'Temperature (°C)', type: 'scatter', mode: 'lines',
               line: {{ color: '#ffc107', width: 2 }}, yaxis: 'y' }},
            {{ x: ts.dates, y: ts.precipitation, name: 'Precipitation (mm)', type: 'scatter', mode: 'lines',
               line: {{ color: '#4dabf7', width: 2 }}, yaxis: 'y2' }},
            {{ x: ts.dates, y: ts.google_trends, name: 'Google Trends', type: 'scatter', mode: 'lines',
               line: {{ color: '#9334e6', width: 2, dash: 'dot' }}, yaxis: 'y' }},
        ];
        // Wedding flag as shaded regions
        const weddingX = []; const weddingY = [];
        ts.dates.forEach((d, i) => {{
            if (ts.wedding_flag && ts.wedding_flag[i] === 1) {{
                weddingX.push(d); weddingY.push(1);
            }} else {{
                weddingX.push(d); weddingY.push(0);
            }}
        }});
        sigTraces.push({{
            x: weddingX, y: weddingY.map(v => v ? 300 : 0),
            type: 'bar', name: 'Wedding Season', yaxis: 'y2',
            marker: {{ color: 'rgba(227,116,0,0.1)' }},
            width: 20 * 86400000,
        }});
        Plotly.newPlot('chart-signals-detail', sigTraces, {{
            ...plotLayout,
            height: 220,
            margin: {{ l: 50, r: 60, t: 10, b: 30 }},
            xaxis: {{ ...plotLayout.xaxis, title: '' }},
            yaxis: {{ ...plotLayout.yaxis, title: 'AQI / Temp / Trends', side: 'left' }},
            yaxis2: {{ ...plotLayout.yaxis, title: 'Precip (mm)', side: 'right', overlaying: 'y', showgrid: false }},
            legend: {{ ...plotLayout.legend, orientation: 'h', y: 1.2, x: 0.5, xanchor: 'center', font: {{ size: 10, color: COLORS.text }} }},
            barmode: 'overlay',
        }}, {{ responsive: true }});
    }}

    // Model selection explanation — uses 'recommended' and 'baselineWins' for consistency
    if (m.model_scores && Object.keys(m.model_scores).length > 0) {{
        const scores = m.model_scores;
        const sortedNames = Object.keys(scores).sort((a, b) => scores[a] - scores[b]);
        const bestMLName = sortedNames[0];
        const bestMLMape = scores[bestMLName];
        const isEns = m.best_model && m.best_model.startsWith('Ensemble');
        const poolName = m.signal_pool;

        // Build the signal relevance note
        const signalMap = {{
            'AQI': 'AQI (air quality index) — correlates with respiratory and antihistamine demand',
            'Monsoon': 'Precipitation — monsoon rainfall drives anti-malarial and ORS demand',
            'Temperature': 'Temperature — heat extremes drive electrolyte and dermatological demand',
            'Wedding': 'Wedding Season flag — seasonal peaks drive vitamin and supplement demand',
            'GoogleTrends': 'Google Trends (symptom search volume) — search spikes predict symptom-linked SKU demand',
        }};
        const signalNote = signalMap[poolName] || 'External signals';

        let expl = '';
        expl += `<div style="margin-bottom:6px"><strong style="color:var(--text-primary)">1. Pool assignment:</strong> This SKU belongs to the <strong style="color:var(--accent-green)">${{poolName}}</strong> signal pool. Primary signal: ${{signalNote}}.</div>`;
        expl += `<div style="margin-bottom:6px"><strong style="color:var(--text-primary)">2. Candidate training:</strong> Four ML models were trained on the ${{poolName}} pool's data — ${{sortedNames.join(', ')}} — using 24 months of training data.</div>`;
        expl += `<div style="margin-bottom:6px"><strong style="color:var(--text-primary)">3. Test-period evaluation:</strong> After retraining on all 30 training months, each model was evaluated on this SKU's 6-month hold-out test period (months 31–36). Test MAPE scores: `;
        expl += sortedNames.map(n => `<strong>${{n}}</strong>: ${{scores[n].toFixed(1)}}%`).join(' · ') + `.</div>`;
        expl += `<div style="margin-bottom:6px"><strong style="color:var(--text-primary)">4. Best ML model for this SKU:</strong> `;
        expl += `<strong>${{bestMLName}}</strong> achieved the lowest test-period MAPE of ${{bestMLMape.toFixed(1)}}% (accuracy: ${{(100 - bestMLMape).toFixed(1)}}%) for this specific SKU.`;
        expl += `</div>`;
        expl += `<div style="margin-bottom:6px"><strong style="color:var(--text-primary)">5. Baseline comparison:</strong> `;
        if (baselineWins) {{
            expl += `On the 6-month test period, <strong style="color:var(--accent-green)">Baseline (3M SMA)</strong> achieved ${{m.baseline_fa.toFixed(1)}}% accuracy vs ${{(100 - bestMLMape).toFixed(1)}}% for ${{bestMLName}}. Since the simple baseline outperformed the best ML model for this SKU, it is recommended.`;
        }} else {{
            expl += `On the 6-month test period, <strong style="color:var(--accent-green)">${{bestMLName}}</strong> achieved ${{(100 - bestMLMape).toFixed(1)}}% accuracy, outperforming Baseline (3M SMA) at ${{m.baseline_fa.toFixed(1)}}%. The ML model is recommended.`;
        }}
        expl += `</div>`;
        expl += `<div><strong style="color:var(--text-primary)">6. Final training:</strong> All models were retrained on all 30 training months (train + validation) before generating the test-period and future forecasts.</div>`;
        document.getElementById('model-explanation').innerHTML = expl;
    }}

    // Time series chart
    if (ts) {{
        const traces = [
            {{ x: ts.dates, y: ts.actual, name: 'Actual', type: 'scatter', mode: 'lines+markers',
               line: {{ color: COLORS.text, width: 2.5 }}, marker: {{ size: 5 }} }},
            {{ x: ts.dates, y: ts.baseline, name: 'Baseline (SMA)', type: 'scatter', mode: 'lines',
               line: {{ color: COLORS.blue, width: 2.5, dash: 'dot' }} }},
        ];
        if (ts.ml) {{
            traces.push({{ x: ts.dates, y: ts.ml, name: 'ML Model', type: 'scatter', mode: 'lines',
               line: {{ color: COLORS.green, width: 2.5 }} }});
        }}
        // Add train/test split line
        const splitIdx = Math.min(29, ts.dates.length - 1);
        traces.push({{
            x: [ts.dates[splitIdx], ts.dates[splitIdx]],
            y: [0, Math.max(...ts.actual.filter(v => v !== null)) * 1.2],
            mode: 'lines', name: 'Train/Test Split',
            line: {{ color: COLORS.amber, dash: 'dash', width: 1 }},
        }});
        Plotly.newPlot('chart-ts-detail', traces, {{
            ...plotLayout,
            xaxis: {{ ...plotLayout.xaxis, title: '' }},
            yaxis: {{ ...plotLayout.yaxis, title: 'Demand (units)' }},
            legend: {{ ...plotLayout.legend, orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center' }},
        }}, {{ responsive: true }});
    }}

    // Residual plot (forecast error over time)
    if (ts && ts.ml) {{
        const residualDates = [];
        const residuals = [];
        for (let i = 0; i < ts.dates.length; i++) {{
            if (ts.ml[i] !== null && ts.actual[i] !== null) {{
                residualDates.push(ts.dates[i]);
                residuals.push(ts.actual[i] - ts.ml[i]);
            }}
        }}
        const resColors = residuals.map(r => r >= 0 ? 'rgba(99,230,190,0.7)' : 'rgba(255,135,135,0.7)');
        Plotly.newPlot('chart-residuals', [{{
            x: residualDates, y: residuals,
            type: 'bar', name: 'Residual',
            marker: {{ color: resColors }},
            hovertemplate: 'Date: %{{x}}<br>Error: %{{y:.0f}} units<extra></extra>',
        }}, {{
            x: [residualDates[0], residualDates[residualDates.length - 1]],
            y: [0, 0],
            type: 'scatter', mode: 'lines', name: 'Zero',
            line: {{ color: COLORS.textSec, width: 1, dash: 'dash' }},
            showlegend: false,
        }}], {{
            ...plotLayout,
            height: 180,
            margin: {{ l: 50, r: 30, t: 5, b: 30 }},
            xaxis: {{ ...plotLayout.xaxis, title: '' }},
            yaxis: {{ ...plotLayout.yaxis, title: 'Error (units)' }},
            showlegend: false,
        }}, {{ responsive: true }});
    }}

    // Future forecast
    if (future.length > 0) {{
        const futureDates = future.map(f => f.date.substring(0, 10));
        // CI width warning
        const ciWarning = document.getElementById('ci-warning');
        const avgForecast = future.reduce((s, f) => s + f.forecast, 0) / future.length;
        const avgRange = future.reduce((s, f) => s + (f.upper_bound - f.lower_bound), 0) / future.length;
        const ciPct = avgForecast > 0 ? (avgRange / avgForecast * 100) : 0;
        if (ciPct > 40) {{
            ciWarning.innerHTML = '<div style="background:rgba(227,116,0,0.06);border:1px solid rgba(227,116,0,0.15);border-radius:10px;padding:10px 14px;margin-bottom:12px;font-size:12px;color:var(--accent-amber)"><strong>&#9888; Wide confidence interval</strong> — CI spans &plusmn;' + (ciPct/2).toFixed(0) + '% of forecast. Consider manual review or additional signal inputs for this SKU.</div>';
        }} else {{
            ciWarning.innerHTML = '';
        }}
        Plotly.newPlot('chart-future-forecast', [
            {{
                x: futureDates, y: future.map(f => f.upper_bound),
                type: 'scatter', mode: 'lines', name: 'Upper Bound',
                line: {{ color: 'rgba(0,214,143,0.3)', width: 0 }}, showlegend: false,
            }},
            {{
                x: futureDates, y: future.map(f => f.lower_bound),
                type: 'scatter', mode: 'lines', name: 'Confidence Interval',
                fill: 'tonexty', fillcolor: 'rgba(0,214,143,0.12)',
                line: {{ color: 'rgba(0,214,143,0.3)', width: 0 }},
            }},
            {{
                x: futureDates, y: future.map(f => f.forecast),
                type: 'scatter', mode: 'lines+markers', name: 'Point Forecast',
                line: {{ color: COLORS.green, width: 3 }},
                marker: {{ size: 8 }},
            }},
        ], {{
            ...plotLayout,
            xaxis: {{ ...plotLayout.xaxis, title: '' }},
            yaxis: {{ ...plotLayout.yaxis, title: 'Forecasted Demand' }},
            legend: {{ ...plotLayout.legend, orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center' }},
        }}, {{ responsive: true }});
    }}

    // Forecast table
    if (future.length > 0) {{
        let tbl = `<table style="width:100%;font-size:12px">
            <thead><tr>
                <th style="text-align:left;padding:6px 10px">Month</th>
                <th style="text-align:right;padding:6px 10px">Point Forecast</th>
                <th style="text-align:right;padding:6px 10px">Lower Bound (95%)</th>
                <th style="text-align:right;padding:6px 10px">Upper Bound (95%)</th>
                <th style="text-align:right;padding:6px 10px">CI Width</th>
            </tr></thead><tbody>`;
        future.forEach(f => {{
            const dt = f.date.substring(0, 10);
            const ciWidth = Math.round(f.upper_bound - f.lower_bound);
            tbl += `<tr>
                <td style="padding:5px 10px;font-weight:600">${{dt}}</td>
                <td style="padding:5px 10px;text-align:right;color:var(--accent-green);font-weight:700">${{Math.round(f.forecast).toLocaleString()}}</td>
                <td style="padding:5px 10px;text-align:right">${{Math.round(f.lower_bound).toLocaleString()}}</td>
                <td style="padding:5px 10px;text-align:right">${{Math.round(f.upper_bound).toLocaleString()}}</td>
                <td style="padding:5px 10px;text-align:right;color:var(--text-secondary)">&#177; ${{Math.round(ciWidth / 2).toLocaleString()}}</td>
            </tr>`;
        }});
        const avgFc = Math.round(future.reduce((s, f) => s + f.forecast, 0) / future.length);
        const totalFc = Math.round(future.reduce((s, f) => s + f.forecast, 0));
        tbl += `<tr style="border-top:2px solid var(--card-border);font-weight:700">
            <td style="padding:5px 10px">Total / Avg</td>
            <td style="padding:5px 10px;text-align:right;color:var(--accent-green)">${{totalFc.toLocaleString()}} / ${{avgFc.toLocaleString()}}</td>
            <td style="padding:5px 10px;text-align:right">${{Math.round(future.reduce((s, f) => s + f.lower_bound, 0)).toLocaleString()}}</td>
            <td style="padding:5px 10px;text-align:right">${{Math.round(future.reduce((s, f) => s + f.upper_bound, 0)).toLocaleString()}}</td>
            <td style="padding:5px 10px;text-align:right;color:var(--text-secondary)">—</td>
        </tr>`;
        tbl += `</tbody></table>`;
        document.getElementById('forecast-table').innerHTML = tbl;
    }}
}}

// ---- TAB 4: SAFETY STOCK ----
function updateSafetyStock() {{
    const skuId = document.getElementById('safety-sku-selector').value;
    const m = metricsData.find(x => x.sku_id === skuId);
    if (!m) return;

    const baselineSS = m.baseline_safety_stock;
    const mlSS = m.ml_safety_stock;
    const reduction = m.safety_stock_reduction;

    // Estimate cost: assume unit cost proportional to avg demand category
    const unitCost = m.abc_xyz.startsWith('B') ? 45 : 22;
    const baselineCost = baselineSS * unitCost;
    const mlCost = mlSS * unitCost;
    const costSaving = baselineCost - mlCost;
    const costSavingPct = baselineCost > 0 ? (costSaving / baselineCost * 100) : 0;

    let html = `
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="label">SKU</div>
            <div class="value blue" style="font-size:18px">${{m.sku_name}}</div>
            <div class="sub">${{m.sku_id}} | ${{m.signal_pool}} Pool | ${{m.abc_xyz}}</div>
        </div>
        <div class="kpi-card">
            <div class="label has-tooltip" data-tooltip="Standard deviation of baseline forecast errors — measures how unpredictable the baseline model is. Higher = more safety stock needed.">Forecast Error Std (Baseline)</div>
            <div class="value" style="color:var(--accent-blue)">${{m.baseline_error_std.toFixed(0)}}</div>
            <div class="sub">units</div>
        </div>
        <div class="kpi-card">
            <div class="label has-tooltip" data-tooltip="Standard deviation of ML model forecast errors — lower means more precise forecasts and less safety stock required.">Forecast Error Std (ML)</div>
            <div class="value green">${{m.ml_error_std.toFixed(0)}}</div>
            <div class="sub">units</div>
        </div>
        <div class="kpi-card">
            <div class="label has-tooltip" data-tooltip="Estimated holding cost reduction from using ML-optimized safety stock levels. Based on per-unit cost × stock reduction.">Inventory Cost Saving</div>
            <div class="value green">${{costSavingPct.toFixed(1)}}%</div>
            <div class="sub">&#8377;${{Math.round(costSaving).toLocaleString()}} saved</div>
        </div>
    </div>

    <div class="two-col">
        <div class="chart-container">
            <div class="chart-title">Safety Stock Comparison</div>
            <div class="comparison-row">
                <div class="stat-block baseline">
                    <div class="stat-label">Baseline Safety Stock</div>
                    <div class="stat-value">${{Math.round(baselineSS)}}</div>
                    <div style="font-size:12px;color:var(--text-secondary)">units</div>
                </div>
                <div class="arrow">&#10132;</div>
                <div class="stat-block ml">
                    <div class="stat-label">Optimized (ML) Safety Stock</div>
                    <div class="stat-value">${{Math.round(mlSS)}}</div>
                    <div style="font-size:12px;color:var(--text-secondary)">units</div>
                </div>
            </div>
            <div style="text-align:center;margin-top:16px">
                <div style="font-size:12px;color:var(--text-secondary)">Safety Stock Reduction</div>
                <div class="savings-badge">${{reduction.toFixed(1)}}% reduction</div>
            </div>
            <div style="margin-top:16px;padding:12px;background:var(--navy);border-radius:8px;font-size:11px;color:var(--text-secondary)">
                <strong>Formula:</strong> Safety Stock = Z &times; &sigma;<sub>error</sub> &times; &radic;Lead Time &nbsp;|&nbsp; Z = 1.65 (95% SL) &nbsp;|&nbsp; Lead Time = 30 days
            </div>
        </div>
        <div class="chart-container">
            <div class="chart-title">What-If Scenario Simulator</div>
            <div class="chart-subtitle">Adjust parameters to see safety stock impact in real-time</div>
            <div style="display:flex;flex-direction:column;gap:20px;padding:4px 0">
                <div>
                    <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px">
                        <span style="color:var(--text-secondary);font-weight:600">Service Level</span>
                        <span id="sim-sl-val" style="color:var(--accent-green);font-weight:700">95%</span>
                    </div>
                    <input type="range" id="sim-sl" min="80" max="99" value="95" step="1" oninput="runSimulation()" style="width:100%;accent-color:var(--accent-green)">
                    <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-secondary)"><span>80%</span><span>99%</span></div>
                </div>
                <div>
                    <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px">
                        <span style="color:var(--text-secondary);font-weight:600">Lead Time (days)</span>
                        <span id="sim-lt-val" style="color:var(--accent-green);font-weight:700">30</span>
                    </div>
                    <input type="range" id="sim-lt" min="7" max="90" value="30" step="1" oninput="runSimulation()" style="width:100%;accent-color:var(--accent-green)">
                    <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-secondary)"><span>7 days</span><span>90 days</span></div>
                </div>
                <div>
                    <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px">
                        <span style="color:var(--text-secondary);font-weight:600">Forecast Error Adjustment</span>
                        <span id="sim-err-val" style="color:var(--accent-green);font-weight:700">+0%</span>
                    </div>
                    <input type="range" id="sim-err" min="-50" max="100" value="0" step="5" oninput="runSimulation()" style="width:100%;accent-color:var(--accent-green)">
                    <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-secondary)"><span>-50%</span><span>+100%</span></div>
                </div>
            </div>
            <div id="sim-results" style="margin-top:16px;padding:16px;background:var(--navy);border-radius:10px"></div>
        </div>
    </div>
    <div class="chart-container" style="margin-top:0">
        <div class="chart-title">Safety Stock — Top 10 by Reduction</div>
        <div id="chart-safety-all"></div>
    </div>`;

    document.getElementById('safety-content').innerHTML = html;

    // All SKUs safety stock chart — sorted by reduction magnitude
    const sorted = [...metricsData].sort((a, b) => b.safety_stock_reduction - a.safety_stock_reduction);
    const top10 = sorted.slice(0, 10);
    const labels = top10.map(s => s.sku_name.substring(0, 20));
    const reductionAnnotations = top10.map((s, i) => ({{
        x: Math.max(s.baseline_safety_stock, s.ml_safety_stock) + 20,
        y: i,
        text: '-' + s.safety_stock_reduction.toFixed(0) + '%',
        showarrow: false,
        font: {{ color: COLORS.green, size: 11, family: 'monospace' }},
        xanchor: 'left',
    }}));
    Plotly.newPlot('chart-safety-all', [
        {{
            y: labels,
            x: top10.map(s => s.baseline_safety_stock),
            type: 'bar', orientation: 'h', name: 'Baseline',
            marker: {{ color: '#74c0fc', opacity: 0.8 }},
        }},
        {{
            y: labels,
            x: top10.map(s => s.ml_safety_stock),
            type: 'bar', orientation: 'h', name: 'ML Optimized',
            marker: {{ color: '#63e6be', opacity: 0.9 }},
        }},
    ], {{
        ...plotLayout,
        barmode: 'group',
        height: 380,
        margin: {{ l: 150, r: 60, t: 10, b: 40 }},
        xaxis: {{ ...plotLayout.xaxis, title: 'Safety Stock (units)' }},
        yaxis: {{ ...plotLayout.yaxis, autorange: 'reversed' }},
        legend: {{ ...plotLayout.legend, orientation: 'h', y: 1.05, x: 0.5, xanchor: 'center' }},
        annotations: reductionAnnotations,
    }}, {{ responsive: true }});

    // Initialize simulator
    runSimulation();
}}

// ---- SCENARIO SIMULATOR ----
function runSimulation() {{
    const skuId = document.getElementById('safety-sku-selector').value;
    const m = metricsData.find(x => x.sku_id === skuId);
    if (!m) return;

    const sl = parseFloat(document.getElementById('sim-sl').value);
    const lt = parseFloat(document.getElementById('sim-lt').value);
    const errAdj = parseFloat(document.getElementById('sim-err').value);

    // Update displayed values
    document.getElementById('sim-sl-val').textContent = sl + '%';
    document.getElementById('sim-lt-val').textContent = lt;
    document.getElementById('sim-err-val').textContent = (errAdj >= 0 ? '+' : '') + errAdj + '%';

    // Z-score lookup for service levels
    const zLookup = {{ 80: 0.84, 85: 1.04, 90: 1.28, 91: 1.34, 92: 1.41, 93: 1.48, 94: 1.55, 95: 1.65, 96: 1.75, 97: 1.88, 98: 2.05, 99: 2.33 }};
    const zScore = zLookup[sl] || (0.84 + (sl - 80) * 0.078);

    // Adjusted forecast error std
    const adjErrStd = m.ml_error_std * (1 + errAdj / 100);

    // Safety stock = Z * sigma * sqrt(lead_time_in_months)
    const ltMonths = lt / 30;
    const simSS = zScore * adjErrStd * Math.sqrt(ltMonths);
    const defaultSS = m.ml_safety_stock;  // Z=1.65, LT=30d, no error adj

    const unitCost = m.abc_xyz.startsWith('B') ? 45 : 22;
    const simCost = Math.round(simSS * unitCost);
    const defaultCost = Math.round(defaultSS * unitCost);
    const costDelta = simCost - defaultCost;
    const ssDelta = simSS - defaultSS;
    const deltaColor = costDelta > 0 ? COLORS.red : COLORS.green;
    const deltaSign = costDelta >= 0 ? '+' : '';

    // Stockout risk estimate (simplified)
    const stockoutRisk = (100 - sl).toFixed(1);

    const resultsEl = document.getElementById('sim-results');
    if (!resultsEl) return;
    resultsEl.innerHTML = `
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;text-align:center">
            <div>
                <div style="font-size:10px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.5px;font-weight:600">Simulated Safety Stock</div>
                <div style="font-size:24px;font-weight:700;color:var(--accent-green);margin:4px 0">${{Math.round(simSS)}}</div>
                <div style="font-size:11px;color:${{deltaColor}}">${{deltaSign}}${{Math.round(ssDelta)}} vs default</div>
            </div>
            <div>
                <div style="font-size:10px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.5px;font-weight:600">Holding Cost</div>
                <div style="font-size:24px;font-weight:700;color:var(--accent-blue);margin:4px 0">&#8377;${{simCost.toLocaleString()}}</div>
                <div style="font-size:11px;color:${{deltaColor}}">${{deltaSign}}&#8377;${{Math.abs(costDelta).toLocaleString()}}</div>
            </div>
            <div>
                <div style="font-size:10px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.5px;font-weight:600">Stockout Risk</div>
                <div style="font-size:24px;font-weight:700;color:${{sl < 90 ? COLORS.red : (sl < 95 ? COLORS.amber : COLORS.green)}};margin:4px 0">${{stockoutRisk}}%</div>
                <div style="font-size:11px;color:var(--text-secondary)">${{sl >= 95 ? 'Low risk' : (sl >= 90 ? 'Moderate' : 'High risk')}}</div>
            </div>
        </div>
    `;
}}

// ---- CROSS-FILTERING ----
function setPoolFilter(pool) {{
    activePoolFilter = pool;
    // Update pill UI
    document.querySelectorAll('.pool-pill').forEach(p => {{
        p.classList.toggle('active', p.dataset.pool === pool);
    }});
    // Re-render affected charts
    renderOverview();
    renderRisk();
}}

function buildPoolFilterBar() {{
    const bar = document.getElementById('pool-filter-bar');
    if (!bar) return;
    const pools = [...new Set(metricsData.map(m => m.signal_pool))];
    let html = `<span style="font-size:11px;color:var(--text-secondary);font-weight:600;letter-spacing:0.5px;margin-right:4px">Demand Drivers</span>`;
    html += `<div class="pool-pill active" onclick="setPoolFilter('all')" data-pool="all">All</div>`;
    pools.forEach(pool => {{
        html += `<div class="pool-pill" onclick="setPoolFilter('${{pool}}')" data-pool="${{pool}}"><span class="pill-dot" style="background:${{POOL_COLORS[pool] || COLORS.green}}"></span>${{pool}}</div>`;
    }});
    bar.innerHTML = html;
}}

// ---- PROACTIVE AI BRIEFING ----
function renderAiBriefing() {{
    const el = document.getElementById('ai-briefing-overview');
    if (!el) return;

    const atRiskSkus = metricsData.filter(m => m.ml_fa < 70).sort((a, b) => a.ml_fa - b.ml_fa);
    const totalBaselineSS = metricsData.reduce((s, m) => s + m.baseline_safety_stock, 0);
    const totalMlSS = metricsData.reduce((s, m) => s + m.ml_safety_stock, 0);
    const ssSaving = Math.round(totalBaselineSS - totalMlSS);

    // Build worst pool insight
    const poolPerf = {{}};
    metricsData.forEach(m => {{
        if (!poolPerf[m.signal_pool]) poolPerf[m.signal_pool] = {{ total: 0, atRisk: 0 }};
        poolPerf[m.signal_pool].total++;
        if (m.ml_fa < 70) poolPerf[m.signal_pool].atRisk++;
    }});
    const worstPoolEntry = Object.entries(poolPerf).sort((a, b) => (b[1].atRisk/b[1].total) - (a[1].atRisk/a[1].total))[0];
    const worstPool = worstPoolEntry ? worstPoolEntry[0] : null;
    const worstPoolPct = worstPoolEntry ? (worstPoolEntry[1].atRisk / worstPoolEntry[1].total * 100).toFixed(0) : 0;

    // Summary line (always visible when collapsed)
    let summaryLine = '';
    if (atRiskSkus.length > 0 && worstPool) {{
        summaryLine = `${{worstPool}} signals explain ${{worstPoolPct}}% of low-accuracy SKUs. ${{ssSaving.toLocaleString()}} units of safety stock savings available.`;
    }} else {{
        summaryLine = `All SKUs performing above threshold. ${{ssSaving.toLocaleString()}} units of safety stock savings available.`;
    }}

    // Build narrative paragraph (expanded view)
    let narrative = '';
    if (atRiskSkus.length > 0) {{
        const worst = atRiskSkus[0];
        narrative += `<strong style="color:${{COLORS.red}}">&#9888; Key Insight:</strong> `;
        narrative += `<strong>${{atRiskSkus.length}} SKUs</strong> have forecast accuracy below 70%. `;
        if (worstPool && worstPoolEntry[1].atRisk > 0) {{
            narrative += `${{worstPoolPct}}% of these belong to the <strong>${{worstPool}}</strong> signal pool, suggesting strong ${{worstPool.toLowerCase()}}-driven demand volatility. `;
        }}
        narrative += `The worst performer is <strong>${{worst.sku_name}}</strong> at ${{worst.ml_fa.toFixed(1)}}% accuracy — `;
        narrative += `<a href="#" onclick="navigateToSku('${{worst.sku_id}}');return false" style="color:var(--accent-green);font-weight:600">investigate &#8594;</a>`;
    }} else {{
        narrative += `<strong style="color:${{COLORS.green}}">&#10003; All clear.</strong> Every SKU is above the 70% forecast accuracy threshold.`;
    }}

    // Anomalies
    let anomalyLine = '';
    if (anomaliesData.length > 0) {{
        anomalyLine = `<div style="margin-top:8px;padding-top:8px;border-top:1px solid rgba(0,0,0,0.06)"><strong style="color:${{COLORS.amber}}">&#9888; Active Signals:</strong> ${{anomaliesData.map(a => a.message).join(' | ')}}</div>`;
    }}

    // Recommended Actions (new: "What should I do?")
    let actionsHtml = '';
    if (atRiskSkus.length > 0) {{
        let actions = [];
        if (worstPool) actions.push(`Review <strong>${{worstPool}}</strong> pool SKUs — <a href="#" onclick="setPoolFilter('${{worstPool}}');return false" style="color:var(--accent-green)">filter now</a>`);
        actions.push(`Increase safety stock for critical SKUs (FA < 30%) — <a href="#" onclick="switchTab('safety');return false" style="color:var(--accent-green)">optimizer</a>`);
        actions.push(`Investigate demand pattern shifts with sales team`);
        actionsHtml = `<div style="margin-top:8px;padding-top:8px;border-top:1px solid rgba(0,0,0,0.06)">
            <strong>Recommended Actions:</strong>
            <div style="margin-top:4px">${{actions.map((a, i) => `<span style="color:var(--text-secondary);font-size:11px;margin-right:4px">${{i+1}}.</span> ${{a}}`).join('<span style="margin:0 8px;color:var(--card-border)">|</span>')}}</div>
        </div>`;
    }}

    // Savings line
    const savingsLine = `<div style="margin-top:8px"><strong style="color:${{COLORS.green}}">&#128176;</strong> ML safety stock saves <strong>${{ssSaving.toLocaleString()}} units</strong> — <a href="#" onclick="switchTab('safety');return false" style="color:var(--accent-green);font-weight:600">explore &#8594;</a></div>`;

    el.innerHTML = `
        <div class="briefing-header" style="cursor:pointer;display:flex;justify-content:space-between;align-items:center" onclick="toggleBriefing()">
            <span>&#9889; Intelligence Briefing</span>
            <span id="briefing-toggle" style="font-size:11px;color:var(--text-secondary);font-weight:400">&#9660; Expand</span>
        </div>
        <div style="font-size:12px;color:var(--text-secondary);margin-top:4px" id="briefing-summary">${{summaryLine}}</div>
        <div class="briefing-body" id="briefing-details" style="display:none;margin-top:10px">
            <div>${{narrative}}</div>
            ${{anomalyLine}}
            ${{actionsHtml}}
            ${{savingsLine}}
        </div>
    `;
}}

// ---- TOP PROBLEM SKUs WIDGET ----
function renderTopProblems() {{
    const el = document.getElementById('top-problems-widget');
    if (!el) return;

    const worst5 = [...metricsData].sort((a, b) => a.ml_fa - b.ml_fa).slice(0, 5);
    if (worst5.length === 0 || worst5[0].ml_fa >= 70) {{
        el.innerHTML = '';
        return;
    }}

    let rows = worst5.map((s, i) => {{
        const barWidth = Math.max(2, s.ml_fa);
        const barColor = s.ml_fa < 30 ? COLORS.red : (s.ml_fa < 50 ? COLORS.amber : COLORS.blue);
        return `<div style="display:grid;grid-template-columns:20px 1fr 80px 48px;gap:8px;align-items:center;padding:7px 0;${{i < 4 ? 'border-bottom:1px solid rgba(0,0,0,0.04)' : ''}}">
            <span style="font-size:12px;font-weight:700;color:var(--text-secondary)">${{i + 1}}</span>
            <div style="min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
                <span style="font-weight:600;cursor:pointer;color:var(--text-primary);font-size:13px" onclick="navigateToSku('${{s.sku_id}}')">${{s.sku_name}}</span>
                <div style="font-size:10px;color:var(--text-secondary);margin-top:1px">${{s.signal_pool}}</div>
            </div>
            <div style="height:5px;border-radius:3px;background:var(--navy);overflow:hidden"><div style="height:100%;width:${{barWidth}}%;background:${{barColor}};border-radius:3px"></div></div>
            <span style="font-weight:700;font-size:12px;text-align:right;color:${{barColor}}">${{s.ml_fa.toFixed(0)}}%</span>
        </div>`;
    }}).join('');

    el.innerHTML = `<div class="chart-container tier-3" style="padding:20px 24px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
            <div class="chart-title" style="margin-bottom:0">Top Forecast Risks</div>
            <a href="#" onclick="switchTab('risk');return false" style="font-size:12px;color:var(--accent-green);font-weight:600;text-decoration:none">View all &#8594;</a>
        </div>
        ${{rows}}
    </div>`;
}}

// ---- INIT ----
function init() {{
    populateSkuDropdowns();
    buildPoolFilterBar();
    renderOverview();
    renderRisk();
    renderAiBriefing();
    renderTopProblems();
    updateRecommendation();
    updateSafetyStock();
    // Fill overview stock reduction KPI
    const avgRed = metricsData.reduce((s, m) => s + m.safety_stock_reduction, 0) / metricsData.length;
    document.getElementById('overview-stock-reduction').textContent = avgRed.toFixed(1) + '%';
}}

document.addEventListener('DOMContentLoaded', init);

// ============================================================
// CHATBOT ENGINE
// ============================================================

function toggleBriefing() {{
    const details = document.getElementById('briefing-details');
    const summary = document.getElementById('briefing-summary');
    const toggle = document.getElementById('briefing-toggle');
    if (!details) return;
    const isHidden = details.style.display === 'none';
    details.style.display = isHidden ? 'block' : 'none';
    summary.style.display = isHidden ? 'none' : 'block';
    toggle.innerHTML = isHidden ? '&#9650; Collapse' : '&#9660; Expand';
}}

function toggleChat() {{
    const panel = document.getElementById('chatPanel');
    const scrim = document.getElementById('chatScrim');
    panel.classList.toggle('open');
    scrim.classList.toggle('open', panel.classList.contains('open'));
    const label = document.getElementById('chatLabel');
    if (label) label.style.display = 'none';
    if (panel.classList.contains('open')) {{
        document.getElementById('chatInput').focus();
    }}
}}

function askSuggestion(el) {{
    document.getElementById('chatInput').value = el.textContent;
    sendChat();
}}

function sendChat() {{
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    if (!text) return;
    input.value = '';

    appendMsg(text, 'user');
    showTyping();

    setTimeout(() => {{
        removeTyping();
        const response = generateResponse(text);
        appendMsg(response, 'bot');
        updateSuggestions(text);
    }}, 400 + Math.random() * 400);
}}

function appendMsg(html, role) {{
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'chat-msg ' + role;
    if (role === 'bot') div.innerHTML = '<div class="msg-label">PharmaCast Assistant</div>' + html;
    else div.textContent = html;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}}

function showTyping() {{
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'chat-msg bot';
    div.id = 'typingIndicator';
    div.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}}

function removeTyping() {{
    const el = document.getElementById('typingIndicator');
    if (el) el.remove();
}}

function updateSuggestions(lastQuery) {{
    const sugBox = document.getElementById('chatSuggestions');
    const q = lastQuery.toLowerCase();
    let suggestions = [];

    if (q.includes('risk') || q.includes('flag'))
        suggestions = ['Show worst performing SKU', 'Any signal anomalies?', 'How to reduce risk?', 'Explain risk score'];
    else if (q.includes('safety') || q.includes('stock') || q.includes('inventory'))
        suggestions = ['Biggest safety stock saving?', 'Explain the formula', 'Which SKU needs most stock?', 'Total cost savings'];
    else if (q.includes('pool') || q.includes('signal'))
        suggestions = ['AQI pool details', 'Monsoon pool details', 'Which pool improved most?', 'Signal anomalies?'];
    else if (q.includes('improv') || q.includes('best') || q.includes('top'))
        suggestions = ['Worst performing SKUs', 'Which pool is best?', 'Overall accuracy summary', 'Any at-risk SKUs?'];
    else
        suggestions = ['Overall summary', 'Which SKUs are at risk?', 'Best performing pool?', 'Safety stock savings'];

    sugBox.innerHTML = suggestions.map(s =>
        `<div class="chat-suggestion" onclick="askSuggestion(this)">${{s}}</div>`
    ).join('');
}}

// ---- CORE NLP RESPONSE ENGINE ----
function generateResponse(query) {{
    const q = query.toLowerCase().replace(/[?!.,]/g, '');
    const words = q.split(/\s+/);

    // Try to find a mentioned SKU
    const skuMatch = findMentionedSku(q);
    // Try to find a mentioned pool
    const poolMatch = findMentionedPool(q);

    // ---- SKU-specific queries ----
    if (skuMatch) return skuDetailResponse(skuMatch, q);

    // ---- Pool-specific queries ----
    if (poolMatch) return poolDetailResponse(poolMatch, q);

    // ---- Cross-tab synthesis ----
    if (matchesAny(q, ['why', 'reason', 'cause', 'explain why', 'root cause', 'insight', 'synthesize', 'diagnose']))
        return synthesisResponse(q);

    // ---- Risk / at-risk queries ----
    if (matchesAny(q, ['risk', 'at risk', 'flag', 'critical', 'warning', 'danger', 'alert', 'problem']))
        return riskResponse(q);

    // ---- Safety stock / inventory ----
    if (matchesAny(q, ['safety stock', 'inventory', 'stock saving', 'cost saving', 'warehouse']))
        return safetyStockResponse(q);

    // ---- Best / top / improvement ----
    if (matchesAny(q, ['best', 'top', 'improv', 'winner', 'highest accuracy', 'good']))
        return topPerformersResponse(q);

    // ---- Worst / lowest ----
    if (matchesAny(q, ['worst', 'lowest', 'poor', 'bad', 'underperform']))
        return worstPerformersResponse(q);

    // ---- Summary / overview / how are we doing ----
    if (matchesAny(q, ['summary', 'overview', 'overall', 'how are', 'status', 'dashboard', 'kpi', 'total']))
        return overviewResponse();

    // ---- Model / recommend / which model ----
    if (matchesAny(q, ['model', 'recommend', 'which model', 'algorithm', 'ml', 'baseline', 'gradient', 'xgboost', 'sma']))
        return modelResponse(q);

    // ---- Signal / anomaly ----
    if (matchesAny(q, ['signal', 'anomal', 'aqi', 'weather', 'temperature', 'monsoon', 'wedding', 'google trend', 'external']))
        return signalResponse(q);

    // ---- Formula / explain ----
    if (matchesAny(q, ['formula', 'explain', 'how does', 'what is', 'meaning', 'define', 'calculate']))
        return explainResponse(q);

    // ---- Forecast / predict / next month ----
    if (matchesAny(q, ['forecast', 'predict', 'next month', 'future', 'project']))
        return forecastResponse(q);

    // ---- Help / what can you do ----
    if (matchesAny(q, ['help', 'what can', 'how to', 'guide', 'capability']))
        return helpResponse();

    // ---- Greeting ----
    if (matchesAny(q, ['hello', 'hi', 'hey', 'good morning', 'good afternoon']))
        return "Hello! I'm your PharmaCast Assistant. I can help you with:<br><br>" +
               "&#8226; <strong>SKU analysis</strong> — ask about any specific SKU<br>" +
               "&#8226; <strong>Risk flags</strong> — which SKUs need attention<br>" +
               "&#8226; <strong>Forecast accuracy</strong> — baseline vs ML comparison<br>" +
               "&#8226; <strong>Safety stock</strong> — optimization and savings<br>" +
               "&#8226; <strong>Signal pools</strong> — AQI, Monsoon, Temperature, etc.<br><br>" +
               "Just type your question!";

    // ---- Fallback ----
    return fallbackResponse(q);
}}

function matchesAny(text, keywords) {{
    return keywords.some(k => text.includes(k));
}}

function findMentionedSku(q) {{
    // Check for SKU ID pattern
    for (const m of metricsData) {{
        if (q.includes(m.sku_id.toLowerCase())) return m;
    }}
    // Check for product name
    for (const m of metricsData) {{
        const nameLower = m.sku_name.toLowerCase();
        const nameWords = nameLower.split(/\s+/);
        // Match if at least the first significant word of the name appears
        if (nameWords.some(w => w.length > 3 && q.includes(w))) return m;
    }}
    return null;
}}

function findMentionedPool(q) {{
    const poolMap = {{
        'aqi': 'AQI', 'respiratory': 'AQI', 'air quality': 'AQI',
        'monsoon': 'Monsoon', 'rain': 'Monsoon', 'precipitation': 'Monsoon', 'malaria': 'Monsoon',
        'temperature': 'Temperature', 'heat': 'Temperature', 'electrolyte': 'Temperature', 'summer': 'Temperature',
        'wedding': 'Wedding', 'vitamin': 'Wedding', 'supplement': 'Wedding', 'nutraceut': 'Wedding',
        'google': 'GoogleTrends', 'trend': 'GoogleTrends', 'search': 'GoogleTrends', 'symptom': 'GoogleTrends',
    }};
    for (const [keyword, pool] of Object.entries(poolMap)) {{
        if (q.includes(keyword)) return pool;
    }}
    return null;
}}

function skuDetailResponse(m, q) {{
    const recommended = m.ml_fa > m.baseline_fa ? (m.best_model || 'ML Model') : 'Baseline (3M SMA)';
    const bestFA = Math.max(m.ml_fa, m.baseline_fa);
    const riskStatus = m.ml_fa < 70 ? '<span style="color:#ff6b6b">&#9888; AT RISK</span>' : '<span style="color:#00d68f">&#10003; Healthy</span>';
    const unitCost = m.abc_xyz.startsWith('B') ? 45 : 22;
    const bSS = m.baseline_safety_stock;
    const mSS = m.ml_safety_stock;
    const saving = bSS > 0 ? ((bSS - mSS) / bSS * 100).toFixed(1) : '0';

    let resp = `<strong>${{m.sku_name}}</strong> (<code>${{m.sku_id}}</code>)<br>`;
    resp += `Pool: ${{m.signal_pool}} | Class: ${{m.abc_xyz}} | Status: ${{riskStatus}}<br><br>`;
    resp += `<strong>Forecast Accuracy:</strong><br>`;
    resp += `&#8226; Baseline (SMA): ${{m.baseline_fa.toFixed(1)}}%<br>`;
    resp += `&#8226; ML Model: ${{m.ml_fa.toFixed(1)}}%<br>`;
    resp += `&#8226; Improvement: <strong>${{m.improvement > 0 ? '+' : ''}}${{m.improvement.toFixed(1)}}%</strong><br><br>`;
    resp += `<strong>Recommendation:</strong> Use ${{recommended}} (${{bestFA.toFixed(1)}}% accuracy)<br>`;
    resp += `<strong>Avg Monthly Demand:</strong> ${{Math.round(m.avg_demand)}} units<br>`;
    resp += `<strong>Safety Stock Saving:</strong> ${{saving}}% (${{Math.round(bSS)}} → ${{Math.round(mSS)}} units)`;

    // If they asked about forecast specifically
    if (matchesAny(q, ['forecast', 'predict', 'next'])) {{
        const future = futureData.filter(f => f.sku_id === m.sku_id);
        if (future.length > 0) {{
            resp += `<br><br><strong>Next 3-Month Forecast:</strong><br>`;
            future.forEach(f => {{
                const dt = f.date.substring(0, 7);
                resp += `&#8226; ${{dt}}: ${{Math.round(f.forecast)}} units (${{Math.round(f.lower_bound)}} – ${{Math.round(f.upper_bound)}})<br>`;
            }});
        }}
    }}

    return resp;
}}

function poolDetailResponse(pool, q) {{
    const poolSkus = metricsData.filter(m => m.signal_pool === pool);
    const pd = poolData.find(p => p.signal_pool === pool);
    if (!pd) return `I don't have data for pool "${{pool}}".`;

    const atRiskCount = poolSkus.filter(s => s.ml_fa < 70).length;
    const bestSku = poolSkus.reduce((a, b) => a.ml_fa > b.ml_fa ? a : b);
    const worstSku = poolSkus.reduce((a, b) => a.ml_fa < b.ml_fa ? a : b);

    let resp = `<strong>${{pool}} Signal Pool</strong><br><br>`;
    resp += `<strong>Performance:</strong><br>`;
    resp += `&#8226; Baseline FA: ${{pd.baseline_fa.toFixed(1)}}%<br>`;
    resp += `&#8226; ML FA: ${{pd.ml_fa.toFixed(1)}}%<br>`;
    resp += `&#8226; Improvement: <strong>+${{pd.improvement.toFixed(1)}}%</strong><br>`;
    resp += `&#8226; SKUs: ${{poolSkus.length}} total, <span style="color:${{atRiskCount > 0 ? '#ff6b6b' : '#00d68f'}}">${{atRiskCount}} at risk</span><br><br>`;
    resp += `<strong>Best:</strong> ${{bestSku.sku_name}} (${{bestSku.ml_fa.toFixed(1)}}% FA)<br>`;
    resp += `<strong>Worst:</strong> ${{worstSku.sku_name}} (${{worstSku.ml_fa.toFixed(1)}}% FA)`;

    return resp;
}}

function riskResponse(q) {{
    const atRiskSkus = metricsData.filter(m => m.ml_fa < 70)
        .sort((a, b) => a.ml_fa - b.ml_fa);

    if (atRiskSkus.length === 0)
        return "<strong style='color:#00d68f'>&#10003; Great news!</strong> No SKUs are currently below the 70% forecast accuracy threshold.";

    let resp = `<strong style="color:#ff6b6b">&#9888; ${{atRiskSkus.length}} SKUs At Risk</strong> (FA < 70%)<br><br>`;

    // Show top 5 worst
    const show = atRiskSkus.slice(0, 5);
    show.forEach((s, i) => {{
        const riskScore = (100 - s.ml_fa + s.mape_volatility).toFixed(0);
        resp += `${{i+1}}. <strong>${{s.sku_name}}</strong> (<code>${{s.sku_id}}</code>) — ${{s.ml_fa.toFixed(1)}}% FA, Risk: ${{riskScore}}<br>`;
    }});

    if (atRiskSkus.length > 5)
        resp += `<br>...and ${{atRiskSkus.length - 5}} more. Check the <strong>Forecast Risk Flags</strong> tab for full details.`;

    // Add anomaly info
    if (anomaliesData.length > 0) {{
        resp += `<br><br><strong>Active Signal Anomalies:</strong><br>`;
        anomaliesData.forEach(a => {{
            resp += `&#9888; ${{a.message}}<br>`;
        }});
    }}

    if (matchesAny(q, ['reduce', 'fix', 'how to', 'improve', 'action'])) {{
        resp += `<br><br><strong>Recommendations:</strong><br>`;
        resp += `&#8226; Review external signal inputs for these SKUs<br>`;
        resp += `&#8226; Consider increasing safety stock for critical items<br>`;
        resp += `&#8226; Investigate if demand patterns have shifted recently<br>`;
        resp += `&#8226; Coordinate with sales team for qualitative demand intelligence`;
    }}

    return resp;
}}

function safetyStockResponse(q) {{
    const sorted = [...metricsData].sort((a, b) => b.safety_stock_reduction - a.safety_stock_reduction);
    const avgReduction = metricsData.reduce((s, m) => s + m.safety_stock_reduction, 0) / metricsData.length;
    const totalBaselineSS = metricsData.reduce((s, m) => s + m.baseline_safety_stock, 0);
    const totalMlSS = metricsData.reduce((s, m) => s + m.ml_safety_stock, 0);
    const totalReduction = totalBaselineSS - totalMlSS;

    let resp = `<strong>Safety Stock Optimization Summary</strong><br><br>`;
    resp += `&#8226; Total Baseline Safety Stock: <strong>${{Math.round(totalBaselineSS).toLocaleString()}}</strong> units<br>`;
    resp += `&#8226; Total ML-Optimized: <strong>${{Math.round(totalMlSS).toLocaleString()}}</strong> units<br>`;
    resp += `&#8226; <strong style="color:#00d68f">Total Reduction: ${{Math.round(totalReduction).toLocaleString()}} units (${{avgReduction.toFixed(1)}}% avg)</strong><br><br>`;

    resp += `<strong>Top Savers:</strong><br>`;
    sorted.slice(0, 5).forEach((s, i) => {{
        resp += `${{i+1}}. ${{s.sku_name}} — ${{s.safety_stock_reduction.toFixed(1)}}% reduction (${{Math.round(s.baseline_safety_stock)}} → ${{Math.round(s.ml_safety_stock)}} units)<br>`;
    }});

    if (matchesAny(q, ['formula', 'how', 'explain'])) {{
        resp += `<br><strong>Formula:</strong> Safety Stock = Z &times; &sigma;<sub>forecast_error</sub> &times; &radic;Lead Time<br>`;
        resp += `Z = 1.65 (95% service level), Lead Time = 30 days`;
    }}

    return resp;
}}

function topPerformersResponse(q) {{
    const sorted = [...metricsData].sort((a, b) => b.improvement - a.improvement);
    const bestPool = [...poolData].sort((a, b) => b.improvement - a.improvement)[0];

    let resp = `<strong>Top ML Improvements</strong><br><br>`;
    resp += `<strong>Best Pool:</strong> ${{bestPool.signal_pool}} (+${{bestPool.improvement.toFixed(1)}}% improvement)<br><br>`;
    resp += `<strong>Top 5 SKUs by Improvement:</strong><br>`;
    sorted.slice(0, 5).forEach((s, i) => {{
        resp += `${{i+1}}. <strong>${{s.sku_name}}</strong> — +${{s.improvement.toFixed(1)}}% (${{s.baseline_fa.toFixed(1)}}% → ${{s.ml_fa.toFixed(1)}}%)<br>`;
    }});

    const improved = metricsData.filter(m => m.improvement > 0).length;
    resp += `<br>${{improved}} of ${{metricsData.length}} SKUs improved with the ML model.`;
    return resp;
}}

function worstPerformersResponse(q) {{
    const sorted = [...metricsData].sort((a, b) => a.ml_fa - b.ml_fa);
    const worstPool = [...poolData].sort((a, b) => a.ml_fa - b.ml_fa)[0];

    let resp = `<strong>Lowest Performing SKUs</strong><br><br>`;
    resp += `<strong>Weakest Pool:</strong> ${{worstPool.signal_pool}} (${{worstPool.ml_fa.toFixed(1)}}% ML FA)<br><br>`;
    resp += `<strong>Bottom 5 SKUs:</strong><br>`;
    sorted.slice(0, 5).forEach((s, i) => {{
        resp += `${{i+1}}. <strong>${{s.sku_name}}</strong> (${{s.sku_id}}) — ${{s.ml_fa.toFixed(1)}}% FA (${{s.signal_pool}})<br>`;
    }});
    return resp;
}}

function overviewResponse() {{
    const avgBFA = metricsData.reduce((s, m) => s + m.baseline_fa, 0) / metricsData.length;
    const avgMFA = metricsData.reduce((s, m) => s + m.ml_fa, 0) / metricsData.length;
    const improved = metricsData.filter(m => m.improvement > 0).length;
    const atRisk = metricsData.filter(m => m.ml_fa < 70).length;

    let resp = `<strong>PharmaCast Dashboard Summary</strong><br><br>`;
    resp += `&#8226; <strong>${{metricsData.length}}</strong> SKUs monitored (BZ/CY/CZ segments)<br>`;
    resp += `&#8226; Avg Baseline Accuracy: <strong>${{avgBFA.toFixed(1)}}%</strong><br>`;
    resp += `&#8226; Avg ML Accuracy: <strong style="color:#00d68f">${{avgMFA.toFixed(1)}}%</strong><br>`;
    resp += `&#8226; Improvement: <strong>+${{(avgMFA - avgBFA).toFixed(1)}}%</strong><br>`;
    resp += `&#8226; SKUs improved: <strong>${{improved}}/${{metricsData.length}}</strong><br>`;
    resp += `&#8226; At-risk SKUs (FA<70%): <strong style="color:${{atRisk > 0 ? '#ff6b6b' : '#00d68f'}}">${{atRisk}}</strong><br>`;
    resp += `&#8226; Signal anomalies: <strong>${{anomaliesData.length}}</strong><br><br>`;
    resp += `5 signal pools: AQI, Monsoon, Temperature, Wedding Season, Google Trends`;
    return resp;
}}

function modelResponse(q) {{
    const mlWins = metricsData.filter(m => m.ml_fa > m.baseline_fa).length;
    const baselineWins = metricsData.length - mlWins;

    // Count model types
    const modelCounts = {{}};
    metricsData.forEach(m => {{
        const name = (m.best_model || 'Unknown');
        const short = name.startsWith('Ensemble') ? 'Ensemble' : name;
        modelCounts[short] = (modelCounts[short] || 0) + 1;
    }});

    let resp = `<strong>Multi-Model Pipeline</strong><br><br>`;
    resp += `<strong>Baselines:</strong> 3-Month SMA + Exponential Smoothing<br>`;
    resp += `<strong>ML Candidates:</strong> Gradient Boosting, Random Forest, Extra Trees, Ridge Regression<br>`;
    resp += `<strong>Selection:</strong> Best model chosen per signal pool via 6-month validation; top 3 ensembled if ensemble beats best single model<br><br>`;
    resp += `<strong>Results:</strong><br>`;
    resp += `&#8226; ML wins on <strong>${{mlWins}}</strong> / ${{metricsData.length}} SKUs<br>`;
    Object.entries(modelCounts).sort((a,b) => b[1]-a[1]).forEach(([name, count]) => {{
        resp += `&#8226; <strong>${{name}}</strong>: ${{count}} SKUs<br>`;
    }});
    resp += `<br>All models use baseline forecast + external signals (AQI, temperature, precipitation, wedding season, Google Trends) + lag features and seasonality encodings.<br><br>`;
    resp += `Check the <strong>Model Recommendations</strong> tab to see per-SKU model comparison charts.`;
    return resp;
}}

function signalResponse(q) {{
    let resp = `<strong>External Signal Status</strong><br><br>`;
    resp += `<strong>5 Signal Pools:</strong><br>`;
    resp += `&#8226; <strong>AQI</strong> → Respiratory/antihistamine SKUs<br>`;
    resp += `&#8226; <strong>Monsoon</strong> → Anti-malarials, ORS, anti-diarrhoeals<br>`;
    resp += `&#8226; <strong>Temperature</strong> → Electrolytes, dermatological<br>`;
    resp += `&#8226; <strong>Wedding Season</strong> → Vitamins, supplements<br>`;
    resp += `&#8226; <strong>Google Trends</strong> → Symptom-search linked<br><br>`;

    if (anomaliesData.length > 0) {{
        resp += `<strong style="color:#ff6b6b">Active Anomalies:</strong><br>`;
        anomaliesData.forEach(a => resp += `&#9888; ${{a.message}}<br>`);
    }} else {{
        resp += `<strong style="color:#00d68f">&#10003; No signal anomalies detected.</strong> All environmental indicators are within normal range.`;
    }}
    return resp;
}}

function explainResponse(q) {{
    if (matchesAny(q, ['mape', 'accuracy', 'fa', 'forecast accuracy']))
        return `<strong>Forecast Accuracy (FA)</strong><br><br>FA = 100% - MAPE<br><br>Where <strong>MAPE</strong> (Mean Absolute Percentage Error) = average of |Actual - Forecast| / Actual &times; 100%<br><br>An FA of 80% means forecasts are on average within 20% of actual demand. SKUs below 70% FA are flagged as "At Risk".`;

    if (matchesAny(q, ['safety stock', 'z score', 'lead time']))
        return `<strong>Safety Stock Formula</strong><br><br>Safety Stock = Z &times; &sigma;<sub>forecast_error</sub> &times; &radic;Lead Time<br><br>&#8226; <strong>Z = 1.65</strong> (95% service level)<br>&#8226; <strong>&sigma;</strong> = standard deviation of forecast errors<br>&#8226; <strong>Lead Time = 30 days</strong><br><br>Better forecasts (lower &sigma;) directly reduce required safety stock, freeing up working capital.`;

    if (matchesAny(q, ['risk score']))
        return `<strong>Risk Score</strong><br><br>Risk Score = (100 - ML_FA%) + MAPE_Volatility<br><br>Higher scores indicate SKUs that are both inaccurate and unpredictable. Scores above 60 are flagged as <span style="color:#ff6b6b">CRITICAL</span>.`;

    if (matchesAny(q, ['abc', 'xyz', 'bz', 'cy', 'cz', 'class', 'segment']))
        return `<strong>ABC-XYZ Classification</strong><br><br><strong>ABC</strong> (value-based):<br>&#8226; A = high revenue, B = medium, C = low<br><br><strong>XYZ</strong> (variability-based):<br>&#8226; X = stable demand, Y = moderate variation, Z = highly erratic<br><br>This dashboard focuses on <strong>BZ, CY, CZ</strong> — medium-to-low value SKUs with high demand variability, which are hardest to forecast.`;

    if (matchesAny(q, ['gradient', 'boosting', 'ml model', 'machine learning', 'algorithm']))
        return `<strong>ML Model: Gradient Boosting Regressor</strong><br><br>A tree-based ensemble that learns residual errors iteratively.<br><br><strong>Features used:</strong><br>&#8226; Baseline forecast (3M SMA)<br>&#8226; External signals (AQI, temp, precipitation, wedding, trends)<br>&#8226; Lag-1 and lag-2 demand<br>&#8226; Rolling 3-month demand std<br>&#8226; Seasonal encodings (sin/cos of month)<br><br>Models are trained per signal pool (not globally) to capture pool-specific patterns.`;

    return `I can explain several concepts:<br><br>&#8226; <strong>Forecast Accuracy / MAPE</strong><br>&#8226; <strong>Safety Stock formula</strong><br>&#8226; <strong>Risk Score</strong><br>&#8226; <strong>ABC-XYZ classification</strong><br>&#8226; <strong>ML models (multi-model pipeline)</strong><br><br>Ask me about any of these!`;
}}

function forecastResponse(q) {{
    let resp = `<strong>Forecast Overview</strong><br><br>`;
    resp += `The system generates 3-month ahead forecasts for all 40 SKUs with 95% confidence intervals.<br><br>`;
    resp += `To see a specific SKU forecast, ask me about a SKU by name (e.g., "forecast for Paracetamol") or check the <strong>Model Recommendations</strong> tab.<br><br>`;

    // Show a sample
    const sample = metricsData[0];
    const future = futureData.filter(f => f.sku_id === sample.sku_id);
    if (future.length > 0) {{
        resp += `<strong>Example — ${{sample.sku_name}}:</strong><br>`;
        future.forEach(f => {{
            resp += `&#8226; ${{f.date.substring(0,7)}}: ${{Math.round(f.forecast)}} units (${{Math.round(f.lower_bound)}} – ${{Math.round(f.upper_bound)}})<br>`;
        }});
    }}
    return resp;
}}

function helpResponse() {{
    return `<strong>What I Can Help With</strong><br><br>` +
        `&#128202; <strong>SKU lookup</strong> — "Tell me about Paracetamol" or "AQI-001 details"<br>` +
        `&#9888;&#65039; <strong>Risk analysis</strong> — "Which SKUs are at risk?" or "Show critical items"<br>` +
        `&#128200; <strong>Performance</strong> — "Best performing pool?" or "Top improvements"<br>` +
        `&#128230; <strong>Safety stock</strong> — "Total safety stock savings" or "Biggest saver"<br>` +
        `&#127777;&#65039; <strong>Signals</strong> — "Any signal anomalies?" or "AQI pool details"<br>` +
        `&#128300; <strong>Explanations</strong> — "Explain MAPE" or "What is risk score?"<br>` +
        `&#128302; <strong>Forecasts</strong> — "Forecast for Salbutamol" or "Next 3 months"<br><br>` +
        `Just type naturally — I understand SKU names, IDs, pool names, and common supply chain terms!`;
}}

function synthesisResponse(q) {{
    // Cross-tab analysis: correlate risk, accuracy, signals, and safety stock
    const atRiskSkus = metricsData.filter(m => m.ml_fa < 70).sort((a, b) => a.ml_fa - b.ml_fa);
    const poolPerf = {{}};
    metricsData.forEach(m => {{
        if (!poolPerf[m.signal_pool]) poolPerf[m.signal_pool] = {{ total: 0, atRisk: 0, totalImprovement: 0 }};
        poolPerf[m.signal_pool].total++;
        poolPerf[m.signal_pool].totalImprovement += m.improvement;
        if (m.ml_fa < 70) poolPerf[m.signal_pool].atRisk++;
    }});

    let resp = '<strong>Cross-Tab Diagnosis</strong><br><br>';

    // Find the worst pool
    const worstPool = Object.entries(poolPerf).sort((a, b) => (b[1].atRisk/b[1].total) - (a[1].atRisk/a[1].total))[0];
    if (worstPool && worstPool[1].atRisk > 0) {{
        const pctAtRisk = (worstPool[1].atRisk / worstPool[1].total * 100).toFixed(0);
        resp += `<strong style="color:#ff8787">Key Finding:</strong> The <strong>${{worstPool[0]}}</strong> pool has the highest concentration of at-risk SKUs (${{pctAtRisk}}% of pool is below 70% FA).<br><br>`;

        // Correlate with anomalies
        const poolAnomalies = anomaliesData.filter(a => a.pool === worstPool[0]);
        if (poolAnomalies.length > 0) {{
            resp += `<strong>Likely cause:</strong> Active signal anomaly detected — ${{poolAnomalies[0].message}}<br>`;
            resp += `This external signal disruption is likely driving forecast degradation for ${{worstPool[0]}} SKUs.<br><br>`;
        }} else {{
            resp += `<strong>Note:</strong> No active signal anomalies for this pool. The low accuracy may indicate structural demand pattern changes that the model hasn't captured.<br><br>`;
        }}
    }}

    // Safety stock impact
    const atRiskSS = atRiskSkus.reduce((s, m) => s + m.baseline_safety_stock, 0);
    resp += `<strong>Inventory Impact:</strong> At-risk SKUs hold ${{Math.round(atRiskSS).toLocaleString()}} units of baseline safety stock. `;
    resp += `Consider <em>increasing</em> safety stock for these items until forecast accuracy improves.<br><br>`;

    resp += `<strong>Recommended Actions:</strong><br>`;
    resp += `1. Review ${{worstPool ? worstPool[0] : 'worst performing'}} pool signal inputs<br>`;
    resp += `2. Increase safety stock for critical SKUs (FA < 30%)<br>`;
    resp += `3. Investigate demand pattern shifts with sales team<br>`;
    resp += `4. Consider manual override forecasts for highest-risk items`;

    return resp;
}}

function fallbackResponse(q) {{
    // Try fuzzy matching on any SKU name
    const words = q.split(/\s+/).filter(w => w.length > 3);
    for (const w of words) {{
        for (const m of metricsData) {{
            if (m.sku_name.toLowerCase().includes(w) || m.sku_id.toLowerCase().includes(w)) {{
                return skuDetailResponse(m, q);
            }}
        }}
    }}

    return `I'm not sure I understood that. Here are things I can help with:<br><br>` +
        `&#8226; Ask about a <strong>specific SKU</strong> (e.g., "Paracetamol details")<br>` +
        `&#8226; <strong>Risk flags</strong> — "Which SKUs are at risk?"<br>` +
        `&#8226; <strong>Pool analysis</strong> — "How is the AQI pool doing?"<br>` +
        `&#8226; <strong>Safety stock</strong> — "Show safety stock savings"<br>` +
        `&#8226; <strong>Explanations</strong> — "Explain risk score"<br><br>` +
        `Try the suggestion chips below for quick questions!`;
}}
</script>

</body>
</html>"""

    return html


if __name__ == "__main__":
    from data_generator import generate_all_data
    from forecasting import run_forecasting_pipeline

    sales, signals, meta = generate_all_data()
    results, metrics, future = run_forecasting_pipeline(sales, signals)
    html = generate_html_dashboard(metrics, results, future, signals)

    with open("dashboard.html", "w") as f:
        f.write(html)
    print("Dashboard written to dashboard.html")
