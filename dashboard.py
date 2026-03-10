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
    --navy: #0a1628;
    --navy-light: #132042;
    --navy-mid: #1a2d5a;
    --accent-green: #00d68f;
    --accent-green-dim: rgba(0,214,143,0.15);
    --accent-red: #ff6b6b;
    --accent-red-dim: rgba(255,107,107,0.12);
    --accent-amber: #ffc107;
    --accent-amber-dim: rgba(255,193,7,0.12);
    --accent-blue: #4dabf7;
    --text-primary: #e8edf5;
    --text-secondary: #8899b4;
    --card-bg: #0f1d36;
    --card-border: #1e3258;
    --input-bg: #132042;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--navy);
    color: var(--text-primary);
    min-height: 100vh;
}}
.header {{
    background: linear-gradient(135deg, var(--navy-light), var(--navy-mid));
    border-bottom: 1px solid var(--card-border);
    padding: 20px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.header h1 {{
    font-size: 22px;
    font-weight: 600;
    letter-spacing: 0.5px;
}}
.header h1 span {{ color: var(--accent-green); }}
.header .subtitle {{
    color: var(--text-secondary);
    font-size: 13px;
    margin-top: 2px;
}}
.header-right {{
    text-align: right;
    color: var(--text-secondary);
    font-size: 12px;
}}
.tab-bar {{
    display: flex;
    gap: 0;
    background: var(--navy-light);
    border-bottom: 2px solid var(--card-border);
    padding: 0 40px;
}}
.tab-btn {{
    padding: 14px 28px;
    border: none;
    background: none;
    color: var(--text-secondary);
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    border-bottom: 3px solid transparent;
    transition: all 0.2s;
}}
.tab-btn:hover {{ color: var(--text-primary); }}
.tab-btn.active {{
    color: var(--accent-green);
    border-bottom-color: var(--accent-green);
}}
.tab-content {{ display: none; padding: 30px 40px; }}
.tab-content.active {{ display: block; }}
.kpi-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}}
.kpi-card {{
    background: var(--card-bg);
    border: none;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2), 0 0 0 1px rgba(255,255,255,0.03);
    transition: transform 0.15s, box-shadow 0.15s;
}}
.kpi-card:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.3), 0 0 0 1px rgba(255,255,255,0.05);
}}
.kpi-card .label {{
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.3px;
    color: var(--text-secondary);
    margin-bottom: 8px;
}}
.kpi-card .value {{
    font-size: 36px;
    font-weight: 700;
}}
.kpi-card .value.green {{ color: var(--accent-green); }}
.kpi-card .value.blue {{ color: var(--accent-blue); }}
.kpi-card .value.amber {{ color: var(--accent-amber); }}
.kpi-card .sub {{
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 4px;
}}
/* Tier system: 1=hero decision signals, 2=diagnostic charts, 3=supporting data */
.chart-container {{
    background: var(--card-bg);
    border: none;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2), 0 0 0 1px rgba(255,255,255,0.03);
}}
.chart-container.tier-3 {{
    background: rgba(15,29,54,0.6);
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
    font-size: 24px;
    font-weight: 300;
    color: var(--text-secondary);
    margin-bottom: 28px;
    letter-spacing: -0.3px;
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
    padding: 12px 16px;
    background: transparent;
    color: var(--text-secondary);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 0.8px;
    border-bottom: 2px solid rgba(255,255,255,0.06);
}}
td {{
    padding: 12px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: var(--text-primary);
}}
tr:hover td {{ background: rgba(255,255,255,0.03); }}
.badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
}}
.badge-red {{ background: var(--accent-red-dim); color: var(--accent-red); }}
.badge-amber {{ background: var(--accent-amber-dim); color: var(--accent-amber); }}
.badge-green {{ background: var(--accent-green-dim); color: var(--accent-green); }}
.alert-box {{
    background: var(--accent-red-dim);
    border: 1px solid rgba(255,107,107,0.3);
    border-radius: 10px;
    padding: 16px 20px;
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
    border: 1px solid var(--card-border);
    color: var(--text-primary);
    padding: 10px 16px;
    border-radius: 8px;
    font-size: 14px;
    min-width: 280px;
    cursor: pointer;
}}
select:focus {{ outline: none; border-color: var(--accent-green); }}
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
    height: 8px;
    background: var(--navy);
    border-radius: 4px;
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
/* Workflow stepper */
.workflow-bar {{
    display: flex;
    align-items: center;
    gap: 0;
    padding: 12px 40px;
    background: var(--navy);
    border-bottom: 1px solid var(--card-border);
    font-size: 12px;
}}
.workflow-step {{
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--text-secondary);
    opacity: 0.5;
    transition: all 0.2s;
}}
.workflow-step.active {{
    color: var(--accent-green);
    opacity: 1;
    font-weight: 600;
}}
.workflow-step.completed {{
    color: var(--accent-green);
    opacity: 0.7;
}}
.workflow-step .step-num {{
    width: 22px;
    height: 22px;
    border-radius: 50%;
    border: 2px solid currentColor;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 700;
}}
.workflow-step.active .step-num {{
    background: var(--accent-green);
    color: var(--navy);
    border-color: var(--accent-green);
}}
.workflow-step.completed .step-num {{
    background: rgba(0,214,143,0.2);
    border-color: var(--accent-green);
}}
.workflow-arrow {{
    margin: 0 12px;
    color: var(--text-secondary);
    opacity: 0.3;
    font-size: 14px;
}}
/* Urgency-differentiated KPI cards */
.kpi-card.urgent {{
    background: linear-gradient(135deg, var(--card-bg), rgba(255,107,107,0.08));
    box-shadow: 0 1px 3px rgba(0,0,0,0.2), inset 0 0 0 1px rgba(255,107,107,0.3);
    animation: urgentPulse 3s ease-in-out infinite;
}}
@keyframes urgentPulse {{
    0%, 100% {{ box-shadow: 0 1px 3px rgba(0,0,0,0.2), inset 0 0 0 1px rgba(255,107,107,0.3); }}
    50% {{ box-shadow: 0 1px 8px rgba(255,107,107,0.15), inset 0 0 0 1px rgba(255,107,107,0.5); }}
}}
.kpi-card.healthy {{
    background: linear-gradient(135deg, var(--card-bg), rgba(0,214,143,0.04));
    box-shadow: 0 1px 3px rgba(0,0,0,0.2), inset 0 0 0 1px rgba(0,214,143,0.15);
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
    border: 1px solid var(--card-border);
    color: var(--text-primary);
    padding: 10px 16px;
    border-radius: 8px;
    font-size: 14px;
    outline: none;
}}
.sku-search-wrap input:focus {{ border-color: var(--accent-green); }}
.sku-dropdown {{
    display: none;
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    max-height: 320px;
    overflow-y: auto;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-top: none;
    border-radius: 0 0 8px 8px;
    z-index: 100;
}}
.sku-dropdown.open {{ display: block; }}
.sku-opt-group {{
    padding: 4px 12px;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-secondary);
    background: var(--navy-light);
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
.sku-opt:hover {{ background: rgba(255,255,255,0.04); }}
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
    background: var(--navy-mid);
    color: var(--text-primary);
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
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
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
    font-weight: 600;
    cursor: pointer;
    border: 1px solid var(--card-border);
    background: var(--card-bg);
    color: var(--text-secondary);
    transition: all 0.15s;
}}
.pool-pill:hover {{ border-color: var(--accent-green); color: var(--text-primary); }}
.pool-pill.active {{
    background: var(--accent-green);
    color: var(--navy);
    border-color: var(--accent-green);
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
    background: linear-gradient(135deg, rgba(0,214,143,0.08), rgba(77,171,247,0.06));
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 24px;
    box-shadow: inset 0 0 0 1px rgba(0,214,143,0.15);
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
    border-radius: 8px;
    text-align: center;
}}
.stat-block.baseline {{ background: rgba(77,171,247,0.1); border: 1px solid rgba(77,171,247,0.2); }}
.stat-block.ml {{ background: var(--accent-green-dim); border: 1px solid rgba(0,214,143,0.2); }}
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
    background: linear-gradient(90deg, var(--card-bg) 25%, rgba(255,255,255,0.05) 50%, var(--card-bg) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s ease-in-out infinite;
    border-radius: 8px;
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
    width: 62px;
    height: 62px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent-green), #00b377);
    border: none;
    cursor: pointer;
    box-shadow: 0 4px 20px rgba(0,214,143,0.4);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    transition: transform 0.2s, box-shadow 0.2s;
    animation: fabPulse 2s ease-in-out infinite;
}}
.chat-fab:hover {{ transform: scale(1.1); box-shadow: 0 6px 28px rgba(0,214,143,0.6); animation: none; }}
.chat-fab svg {{ width: 28px; height: 28px; fill: var(--navy); }}
@keyframes fabPulse {{
    0%, 100% {{ box-shadow: 0 4px 20px rgba(0,214,143,0.4); }}
    50% {{ box-shadow: 0 4px 32px rgba(0,214,143,0.7), 0 0 0 8px rgba(0,214,143,0.12); }}
}}
.chat-fab-label {{
    position: fixed;
    bottom: 98px;
    right: 22px;
    background: var(--card-bg);
    border: 1px solid var(--accent-green);
    color: var(--accent-green);
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 600;
    z-index: 9999;
    white-space: nowrap;
    animation: labelFade 8s ease-in-out forwards;
    pointer-events: none;
}}
.chat-fab-label::after {{
    content: '';
    position: absolute;
    bottom: -6px;
    right: 20px;
    width: 10px;
    height: 10px;
    background: var(--card-bg);
    border-right: 1px solid var(--accent-green);
    border-bottom: 1px solid var(--accent-green);
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
    bottom: 96px;
    right: 28px;
    width: 400px;
    max-height: 560px;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    box-shadow: 0 12px 48px rgba(0,0,0,0.5);
    z-index: 9998;
    display: none;
    flex-direction: column;
    overflow: hidden;
}}
.chat-panel.open {{ display: flex; }}

.chat-header {{
    padding: 16px 20px;
    background: linear-gradient(135deg, var(--navy-mid), var(--navy-light));
    border-bottom: 1px solid var(--card-border);
    display: flex;
    align-items: center;
    gap: 12px;
}}
.chat-header .chat-avatar {{
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: var(--accent-green);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 14px;
    color: var(--navy);
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
    max-height: 380px;
    min-height: 200px;
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
    background: var(--navy-light);
    border: 1px solid var(--card-border);
    color: var(--text-primary);
}}
.chat-msg.user {{
    align-self: flex-end;
    background: var(--accent-green);
    color: var(--navy);
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
    background: rgba(0,0,0,0.2);
    padding: 1px 5px;
    border-radius: 4px;
    font-size: 12px;
}}
.chat-msg strong {{ color: var(--accent-green); }}
.chat-msg.user strong {{ color: var(--navy); }}

.chat-suggestions {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding: 0 16px 12px;
}}
.chat-suggestion {{
    background: var(--navy);
    border: 1px solid var(--card-border);
    color: var(--text-secondary);
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.15s;
}}
.chat-suggestion:hover {{
    border-color: var(--accent-green);
    color: var(--accent-green);
}}

.chat-input-row {{
    display: flex;
    gap: 8px;
    padding: 12px 16px;
    border-top: 1px solid var(--card-border);
    background: var(--navy-light);
}}
.chat-input-row input {{
    flex: 1;
    background: var(--input-bg);
    border: 1px solid var(--card-border);
    color: var(--text-primary);
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 13px;
    outline: none;
}}
.chat-input-row input::placeholder {{ color: var(--text-secondary); opacity: 0.6; }}
.chat-input-row input:focus {{ border-color: var(--accent-green); }}
.chat-send {{
    background: var(--accent-green);
    border: none;
    color: var(--navy);
    width: 38px;
    height: 38px;
    border-radius: 8px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    font-weight: 700;
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
    .chat-panel {{ width: calc(100vw - 24px); right: 12px; bottom: 80px; }}
}}
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
    <button class="tab-btn" onclick="switchTab('risk')">Forecast Risk Flags</button>
    <button class="tab-btn" onclick="switchTab('recommend')">Model Recommendations</button>
    <button class="tab-btn" onclick="switchTab('safety')">Safety Stock Optimizer</button>
    <button class="tab-btn" onclick="toggleChat()" style="margin-left:auto; background:var(--accent-green); color:var(--navy); font-weight:700; border-radius:20px; padding:8px 18px;">💬 Chat Assistant</button>
</div>

<!-- WORKFLOW STEPPER -->
<div class="workflow-bar">
    <div class="workflow-step active" id="step-overview">
        <span class="step-num">1</span> Detect
    </div>
    <span class="workflow-arrow">&#8594;</span>
    <div class="workflow-step" id="step-risk">
        <span class="step-num">2</span> Diagnose
    </div>
    <span class="workflow-arrow">&#8594;</span>
    <div class="workflow-step" id="step-recommend">
        <span class="step-num">3</span> Decide
    </div>
    <span class="workflow-arrow">&#8594;</span>
    <div class="workflow-step" id="step-safety">
        <span class="step-num">4</span> Act
    </div>
</div>

<!-- TAB 1: OVERVIEW -->
<div id="tab-overview" class="tab-content active">
    <!-- Tab question -->
    <div class="tab-question">How well are your <strong>forecasts performing</strong>?</div>

    <!-- AI Briefing (proactive co-pilot) -->
    <div class="ai-briefing" id="ai-briefing-overview"></div>

    <!-- Hero Banner — Tier 1 Decision Signal -->
    <div style="text-align:center;margin-bottom:32px;padding:8px 0">
        <div style="font-size:13px;color:var(--text-secondary);margin-bottom:4px;letter-spacing:0.5px;text-transform:uppercase;font-weight:600" class="has-tooltip" data-tooltip="Difference between ML model and baseline (3-month SMA) forecast accuracy averaged across all SKUs">Forecast Accuracy Improvement</div>
        <div style="font-size:56px;font-weight:800;color:var(--accent-green);letter-spacing:-2px;line-height:1">+{avg_improvement}%</div>
        <div style="font-size:14px;color:var(--text-secondary);margin-top:8px">{skus_improved} of {total_skus} SKUs improved &mdash; Avg ML Forecast Accuracy: {avg_ml_fa}%</div>
    </div>

    <!-- Action Cards with urgency differentiation -->
    <div class="kpi-row" style="grid-template-columns:repeat(3,1fr)">
        <div class="kpi-card {'urgent' if num_at_risk > 0 else 'healthy'}" onclick="switchTab('risk')" style="cursor:pointer">
            <div class="label" style="font-weight:600;letter-spacing:0.3px;font-size:12px;color:{'var(--accent-red)' if num_at_risk > 0 else 'var(--accent-green)'}">{"&#9888; Needs Attention" if num_at_risk > 0 else "&#10003; All Clear"}</div>
            <div class="value" style="color:{'var(--accent-red)' if num_at_risk > 0 else 'var(--accent-green)'}">{str(num_at_risk) + ' SKUs at Risk' if num_at_risk > 0 else 'No At-Risk SKUs'}</div>
            <div class="sub">{'Click to review risk flags &#8594;' if num_at_risk > 0 else 'All above 70% FA threshold'}</div>
        </div>
        <div class="kpi-card {'urgent' if num_anomalies > 0 else 'healthy'}" style="{'padding:16px' if num_anomalies == 0 else ''}">
            <div class="label" style="font-size:12px">{"&#9888; Signal Anomalies" if num_anomalies > 0 else "&#10003; Signals Normal"}</div>
            <div class="value {'amber' if num_anomalies > 0 else ''}" style="{'font-size:28px;color:var(--accent-green)' if num_anomalies == 0 else ''}">{num_anomalies if num_anomalies > 0 else '&#10003;'}</div>
            <div class="sub">{'Active environmental alerts' if num_anomalies > 0 else 'All indicators in normal range'}</div>
        </div>
        <div class="kpi-card" onclick="switchTab('safety')" style="cursor:pointer;border-color:rgba(0,214,143,0.2)">
            <div class="label" style="font-size:12px">Safety Stock Optimization</div>
            <div class="value green" id="overview-stock-reduction">—</div>
            <div class="sub">Avg reduction via ML &mdash; Click for details \u2192</div>
        </div>
    </div>

    <!-- Cross-filter bar -->
    <div class="pool-filter-bar" id="pool-filter-bar">
        <span style="font-size:11px;color:var(--text-secondary);font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-right:4px">Filter by Signal:</span>
        <div class="pool-pill active" onclick="setPoolFilter('all')" data-pool="all">All Pools</div>
    </div>

    <!-- PROMOTED: SKU scatter (most analytically valuable) — Tier 2 Diagnostic -->
    <div class="chart-container">
        <div class="chart-title">SKU-Level Forecast Accuracy — Baseline vs ML</div>
        <div class="chart-subtitle">Each dot = one SKU. Above the diagonal = ML outperforms baseline.</div>
        <div id="chart-sku-scatter"></div>
    </div>

    <div class="two-col">
        <div class="chart-container">
            <div class="chart-title">Forecast Accuracy by Signal Pool</div>
            <div class="chart-subtitle">Grouped comparison of Baseline (SMA) vs ML (Gradient Boosting)</div>
            <div id="chart-pool-comparison"></div>
        </div>
        <div class="chart-container">
            <div class="chart-title">Forecast Accuracy Distribution</div>
            <div class="chart-subtitle">Overlaid histogram — right shift indicates ML improvement</div>
            <div id="chart-fa-distribution"></div>
        </div>
    </div>
</div>

<!-- TAB 2: RISK FLAGS -->
<div id="tab-risk" class="tab-content">
    <div class="tab-question">Which SKUs are <strong>dangerous</strong>?</div>
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
    <div class="tab-question">What forecasting model should we <strong>use</strong>?</div>
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
    <div class="tab-question">How much inventory should we <strong>hold</strong>?</div>
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
    navy: '#0a1628',
    navyLight: '#132042',
    green: '#00d68f',
    blue: '#4dabf7',
    red: '#ff6b6b',
    amber: '#ffc107',
    text: '#e8edf5',
    textSec: '#8899b4',
    gridColor: '#1e3258',
}};

// Consistent signal pool color mapping — used EVERYWHERE
const POOL_COLORS = {{
    'AQI': '#ff6b6b',
    'Temperature': '#ffa94d',
    'Monsoon': '#4dabf7',
    'Wedding': '#da77f2',
    'GoogleTrends': '#69db7c',
}};
const POOL_COLORS_DIM = {{
    'AQI': 'rgba(255,107,107,0.15)',
    'Temperature': 'rgba(255,169,77,0.15)',
    'Monsoon': 'rgba(77,171,247,0.15)',
    'Wedding': 'rgba(218,119,242,0.15)',
    'GoogleTrends': 'rgba(105,219,124,0.15)',
}};

// Active cross-filter state
let activePoolFilter = 'all';

const plotLayout = {{
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {{ family: 'Segoe UI, system-ui, sans-serif', color: COLORS.text, size: 12 }},
    margin: {{ l: 50, r: 30, t: 20, b: 50 }},
    xaxis: {{ gridcolor: COLORS.gridColor, zerolinecolor: COLORS.gridColor }},
    yaxis: {{ gridcolor: COLORS.gridColor, zerolinecolor: COLORS.gridColor }},
    legend: {{ bgcolor: 'rgba(0,0,0,0)', font: {{ size: 11 }} }},
}};

function switchTab(name) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    // Find the correct tab button
    const btns = document.querySelectorAll('.tab-btn');
    const tabMap = {{'overview':0, 'risk':1, 'recommend':2, 'safety':3}};
    if (tabMap[name] !== undefined) btns[tabMap[name]].classList.add('active');
    // Update workflow stepper
    updateWorkflowStep(name);
    // Trigger resize for plotly
    window.dispatchEvent(new Event('resize'));
}}

function navigateToSku(skuId) {{
    // Set the SKU in the Model Recommendations dropdown and switch tab
    const selector = document.getElementById('sku-selector');
    selector.value = skuId;
    updateRecommendation();
    switchTab('recommend');
}}

function updateWorkflowStep(tabName) {{
    const steps = ['overview', 'risk', 'recommend', 'safety'];
    steps.forEach((s, i) => {{
        const el = document.getElementById('step-' + s);
        if (!el) return;
        const idx = steps.indexOf(tabName);
        if (i < idx) el.className = 'workflow-step completed';
        else if (i === idx) el.className = 'workflow-step active';
        else el.className = 'workflow-step';
    }});
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
            type: 'bar', name: 'ML (Gradient Boosting)',
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

    // FA distribution
    Plotly.newPlot('chart-fa-distribution', [
        {{
            x: metricsData.map(m => m.baseline_fa),
            type: 'histogram', name: 'Baseline',
            marker: {{ color: COLORS.blue, opacity: 0.6 }},
            nbinsx: 15,
        }},
        {{
            x: metricsData.map(m => m.ml_fa),
            type: 'histogram', name: 'ML Model',
            marker: {{ color: COLORS.green, opacity: 0.6 }},
            nbinsx: 15,
        }}
    ], {{
        ...plotLayout,
        barmode: 'overlay',
        xaxis: {{ ...plotLayout.xaxis, title: 'Forecast Accuracy %' }},
        yaxis: {{ ...plotLayout.yaxis, title: 'Number of SKUs' }},
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
            mode: 'markers',
            type: 'scatter',
            name: pool,
            marker: {{ color: POOL_COLORS[pool] || COLORS.green, size: 11, opacity: 0.85,
                       line: {{ color: 'rgba(255,255,255,0.2)', width: 1 }} }},
        }});
    }});
    // Diagonal reference line
    scatterTraces.push({{
        x: [0, 105], y: [0, 105],
        mode: 'lines', name: 'No Change Line',
        line: {{ color: 'rgba(255,255,255,0.15)', dash: 'dash', width: 1 }},
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
        let html = '<table><thead><tr><th>SKU ID</th><th>SKU Name</th><th>Pool</th><th>Class</th><th>ML FA%</th><th title="MAPE Volatility: Standard deviation of monthly MAPE values — measures how erratic forecast errors are over time. Higher = less predictable.">MAPE Vol. <span style="cursor:help;opacity:0.6;font-size:10px">&#9432;</span></th><th>Risk Score</th><th>Severity</th></tr></thead><tbody>';
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

    // Risk scatter
    if (riskData.length > 0) {{
        Plotly.newPlot('chart-risk-scatter', [{{
            x: riskData.map(r => r.ml_fa),
            y: riskData.map(r => r.risk_score),
            text: riskData.map(r => r.sku_name),
            mode: 'markers+text',
            type: 'scatter',
            textposition: 'top center',
            textfont: {{ size: 10, color: COLORS.textSec }},
            marker: {{
                color: riskData.map(r => r.risk_score),
                colorscale: [[0, COLORS.amber], [1, COLORS.red]],
                size: 14,
                line: {{ color: 'rgba(255,255,255,0.2)', width: 1 }},
            }},
        }}], {{
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

    const recommended = m.ml_fa > m.baseline_fa ? 'ML (Gradient Boosting)' : 'Baseline (3M SMA)';
    const recFA = Math.max(m.ml_fa, m.baseline_fa);
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
            <div class="label has-tooltip" data-tooltip="The percentage point difference between ML model accuracy and baseline (3-month SMA) accuracy. Positive = ML is better.">Improvement vs Baseline</div>
            <div class="value" style="color:${{m.improvement > 0 ? COLORS.green : COLORS.red}}">${{m.improvement > 0 ? '+' : ''}}${{m.improvement.toFixed(1)}}%</div>
            <div class="sub">Accuracy delta</div>
        </div>
        <div class="kpi-card">
            <div class="label">Avg Monthly Demand</div>
            <div class="value blue">${{Math.round(m.avg_demand)}}</div>
            <div class="sub">units / month</div>
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
        </div>
    </div>`;
    document.getElementById('recommendation-content').innerHTML = html;

    // Time series chart
    if (ts) {{
        const traces = [
            {{ x: ts.dates, y: ts.actual, name: 'Actual', type: 'scatter', mode: 'lines+markers',
               line: {{ color: COLORS.text, width: 2 }}, marker: {{ size: 4 }} }},
            {{ x: ts.dates, y: ts.baseline, name: 'Baseline (SMA)', type: 'scatter', mode: 'lines',
               line: {{ color: COLORS.blue, width: 2, dash: 'dot' }} }},
        ];
        if (ts.ml) {{
            traces.push({{ x: ts.dates, y: ts.ml, name: 'ML Model', type: 'scatter', mode: 'lines',
               line: {{ color: COLORS.green, width: 2 }} }});
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
            ciWarning.innerHTML = '<div style="background:rgba(255,193,7,0.1);border:1px solid rgba(255,193,7,0.3);border-radius:8px;padding:10px 14px;margin-bottom:12px;font-size:12px;color:var(--accent-amber)"><strong>&#9888; Wide confidence interval</strong> — CI spans &plusmn;' + (ciPct/2).toFixed(0) + '% of forecast. Consider manual review or additional signal inputs for this SKU.</div>';
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
            <div class="label">Forecast Error Std (Baseline)</div>
            <div class="value" style="color:var(--accent-blue)">${{m.baseline_error_std.toFixed(0)}}</div>
            <div class="sub">units</div>
        </div>
        <div class="kpi-card">
            <div class="label">Forecast Error Std (ML)</div>
            <div class="value green">${{m.ml_error_std.toFixed(0)}}</div>
            <div class="sub">units</div>
        </div>
        <div class="kpi-card">
            <div class="label">Inventory Cost Saving</div>
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
            <div style="margin-top:20px;padding:16px;background:var(--navy);border-radius:8px;font-size:12px;color:var(--text-secondary)">
                <strong>Formula:</strong> Safety Stock = Z &times; &sigma;<sub>forecast error</sub> &times; &radic;Lead Time<br>
                <strong>Parameters:</strong> Z = 1.65 (95% SL) | Lead Time = 30 days<br>
                <span style="color:var(--accent-amber);font-size:11px;margin-top:4px;display:inline-block">&#9432; Note: Z and Lead Time shown are system defaults. In production, these would vary per SKU based on individual service level agreements and supplier lead times.</span>
            </div>
        </div>
        <div class="chart-container">
            <div class="chart-title">Safety Stock — All SKUs</div>
            <div id="chart-safety-all"></div>
        </div>
    </div>`;

    document.getElementById('safety-content').innerHTML = html;

    // All SKUs safety stock chart — sorted by reduction magnitude
    const sorted = [...metricsData].sort((a, b) => b.safety_stock_reduction - a.safety_stock_reduction);
    const top20 = sorted.slice(0, 20);
    const labels = top20.map(s => s.sku_id + ' (' + s.sku_name.substring(0, 15) + ')');
    const reductionAnnotations = top20.map((s, i) => ({{
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
            x: top20.map(s => s.baseline_safety_stock),
            type: 'bar', orientation: 'h', name: 'Baseline',
            marker: {{ color: '#74c0fc', opacity: 0.8 }},
        }},
        {{
            y: labels,
            x: top20.map(s => s.ml_safety_stock),
            type: 'bar', orientation: 'h', name: 'ML Optimized',
            marker: {{ color: '#63e6be', opacity: 0.9 }},
        }},
    ], {{
        ...plotLayout,
        barmode: 'group',
        height: 560,
        margin: {{ l: 160, r: 60, t: 10, b: 40 }},
        xaxis: {{ ...plotLayout.xaxis, title: 'Safety Stock (units)' }},
        yaxis: {{ ...plotLayout.yaxis, autorange: 'reversed' }},
        legend: {{ ...plotLayout.legend, orientation: 'h', y: 1.05, x: 0.5, xanchor: 'center' }},
        annotations: reductionAnnotations,
    }}, {{ responsive: true }});
}}

// ---- INIT ----
function init() {{
    populateSkuDropdowns();
    renderOverview();
    renderRisk();
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

function toggleChat() {{
    const panel = document.getElementById('chatPanel');
    panel.classList.toggle('open');
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
    const recommended = m.ml_fa > m.baseline_fa ? 'ML (Gradient Boosting)' : 'Baseline (3M SMA)';
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

    let resp = `<strong>Model Comparison</strong><br><br>`;
    resp += `<strong>Baseline:</strong> 3-Month Simple Moving Average (SMA)<br>`;
    resp += `<strong>ML Model:</strong> Gradient Boosting Regressor with external signals<br><br>`;
    resp += `<strong>Results:</strong><br>`;
    resp += `&#8226; ML wins on <strong>${{mlWins}}</strong> SKUs<br>`;
    resp += `&#8226; Baseline wins on <strong>${{baselineWins}}</strong> SKUs<br><br>`;
    resp += `The ML model uses baseline forecast + external signals (AQI, temperature, precipitation, wedding season, Google Trends) + lag features and seasonality encodings.<br><br>`;
    resp += `Check the <strong>Model Recommendations</strong> tab to see per-SKU recommendations.`;
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

    return `I can explain several concepts:<br><br>&#8226; <strong>Forecast Accuracy / MAPE</strong><br>&#8226; <strong>Safety Stock formula</strong><br>&#8226; <strong>Risk Score</strong><br>&#8226; <strong>ABC-XYZ classification</strong><br>&#8226; <strong>ML model (Gradient Boosting)</strong><br><br>Ask me about any of these!`;
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
