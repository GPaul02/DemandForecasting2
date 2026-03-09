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
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}}
.kpi-card .label {{
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
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
.chart-container {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 24px;
}}
.chart-title {{
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--card-border);
}}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}
th {{
    text-align: left;
    padding: 12px 16px;
    background: var(--navy-light);
    color: var(--text-secondary);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.8px;
    border-bottom: 2px solid var(--card-border);
}}
td {{
    padding: 12px 16px;
    border-bottom: 1px solid var(--card-border);
    color: var(--text-primary);
}}
tr:hover td {{ background: rgba(255,255,255,0.02); }}
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
    text-transform: uppercase;
    letter-spacing: 0.5px;
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
@media (max-width: 900px) {{
    .two-col, .detail-grid {{ grid-template-columns: 1fr; }}
    .header {{ padding: 16px 20px; }}
    .tab-content {{ padding: 20px; }}
    .tab-bar {{ padding: 0 20px; overflow-x: auto; }}
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
.stat-block .stat-label {{ font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }}
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
</div>

<!-- TAB 1: OVERVIEW -->
<div id="tab-overview" class="tab-content active">
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="label">Total SKUs Monitored</div>
            <div class="value blue">{total_skus}</div>
            <div class="sub">BZ / CY / CZ Segments</div>
        </div>
        <div class="kpi-card">
            <div class="label">Avg Baseline Accuracy</div>
            <div class="value" style="color:var(--accent-blue)">{avg_baseline_fa}%</div>
            <div class="sub">3-Month Moving Average</div>
        </div>
        <div class="kpi-card">
            <div class="label">Avg ML Accuracy</div>
            <div class="value green">{avg_ml_fa}%</div>
            <div class="sub">Gradient Boosting Model</div>
        </div>
        <div class="kpi-card">
            <div class="label">Avg Improvement</div>
            <div class="value green">+{avg_improvement}%</div>
            <div class="sub">{skus_improved}/{total_skus} SKUs improved</div>
        </div>
    </div>
    <div class="two-col">
        <div class="chart-container">
            <div class="chart-title">Baseline vs ML Forecast Accuracy by Signal Pool</div>
            <div id="chart-pool-comparison"></div>
        </div>
        <div class="chart-container">
            <div class="chart-title">Forecast Accuracy Distribution (All SKUs)</div>
            <div id="chart-fa-distribution"></div>
        </div>
    </div>
    <div class="chart-container">
        <div class="chart-title">SKU-Level Forecast Accuracy — Baseline vs ML</div>
        <div id="chart-sku-scatter"></div>
    </div>
</div>

<!-- TAB 2: RISK FLAGS -->
<div id="tab-risk" class="tab-content">
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="label">At-Risk SKUs (FA &lt; 70%)</div>
            <div class="value" style="color:var(--accent-red)">{num_at_risk}</div>
            <div class="sub">Require immediate attention</div>
        </div>
        <div class="kpi-card">
            <div class="label">Signal Anomalies</div>
            <div class="value amber">{num_anomalies}</div>
            <div class="sub">Active environmental alerts</div>
        </div>
        <div class="kpi-card">
            <div class="label">Highest Risk Score</div>
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
    <div class="selector">
        <label>Select SKU:</label>
        <select id="sku-selector" onchange="updateRecommendation()"></select>
    </div>
    <div id="recommendation-content"></div>
</div>

<!-- TAB 4: SAFETY STOCK -->
<div id="tab-safety" class="tab-content">
    <div class="selector">
        <label>Select SKU:</label>
        <select id="safety-sku-selector" onchange="updateSafetyStock()"></select>
    </div>
    <div id="safety-content"></div>
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
    event.target.classList.add('active');
    // Trigger resize for plotly
    window.dispatchEvent(new Event('resize'));
}}

// ---- TAB 1: OVERVIEW ----
function renderOverview() {{
    // Pool comparison bar chart
    const pools = poolData.map(p => p.signal_pool);
    Plotly.newPlot('chart-pool-comparison', [
        {{
            x: pools, y: poolData.map(p => p.baseline_fa),
            type: 'bar', name: 'Baseline (SMA)',
            marker: {{ color: COLORS.blue, opacity: 0.8 }},
            text: poolData.map(p => p.baseline_fa.toFixed(1) + '%'),
            textposition: 'outside', textfont: {{ size: 11 }},
        }},
        {{
            x: pools, y: poolData.map(p => p.ml_fa),
            type: 'bar', name: 'ML (GBR)',
            marker: {{ color: COLORS.green, opacity: 0.9 }},
            text: poolData.map(p => p.ml_fa.toFixed(1) + '%'),
            textposition: 'outside', textfont: {{ size: 11 }},
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

    // SKU scatter
    const poolColors = {{
        'AQI': '#ff6b6b', 'Monsoon': '#4dabf7', 'Temperature': '#ffc107',
        'Wedding': '#cc5de8', 'GoogleTrends': '#51cf66'
    }};
    const scatterTraces = [];
    const poolNames = [...new Set(metricsData.map(m => m.signal_pool))];
    poolNames.forEach(pool => {{
        const items = metricsData.filter(m => m.signal_pool === pool);
        scatterTraces.push({{
            x: items.map(i => i.baseline_fa),
            y: items.map(i => i.ml_fa),
            text: items.map(i => i.sku_name + ' (' + i.sku_id + ')'),
            mode: 'markers',
            type: 'scatter',
            name: pool,
            marker: {{ color: poolColors[pool] || COLORS.green, size: 10, opacity: 0.8 }},
        }});
    }});
    // Add diagonal reference line
    scatterTraces.push({{
        x: [0, 100], y: [0, 100],
        mode: 'lines', name: 'No Change Line',
        line: {{ color: COLORS.textSec, dash: 'dash', width: 1 }},
        showlegend: false,
    }});
    Plotly.newPlot('chart-sku-scatter', scatterTraces, {{
        ...plotLayout,
        xaxis: {{ ...plotLayout.xaxis, title: 'Baseline FA %', range: [0, 105] }},
        yaxis: {{ ...plotLayout.yaxis, title: 'ML FA %', range: [0, 105] }},
        legend: {{ ...plotLayout.legend, orientation: 'h', y: 1.15, x: 0.5, xanchor: 'center' }},
        annotations: [{{
            x: 30, y: 90, text: 'ML Better', showarrow: false,
            font: {{ color: COLORS.green, size: 12 }},
        }}, {{
            x: 90, y: 30, text: 'Baseline Better', showarrow: false,
            font: {{ color: COLORS.red, size: 12 }},
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
        tableContainer.innerHTML = '<p style="color:var(--accent-green);padding:20px;text-align:center">No SKUs below 70% forecast accuracy — all performing within acceptable range.</p>';
    }} else {{
        let html = '<table><thead><tr><th>SKU ID</th><th>SKU Name</th><th>Pool</th><th>Class</th><th>ML FA%</th><th>MAPE Vol.</th><th>Risk Score</th><th>Status</th></tr></thead><tbody>';
        riskData.forEach(r => {{
            const badge = r.risk_score > 60 ? 'badge-red' : (r.risk_score > 40 ? 'badge-amber' : 'badge-green');
            const status = r.risk_score > 60 ? 'CRITICAL' : (r.risk_score > 40 ? 'WARNING' : 'WATCH');
            html += `<tr>
                <td style="font-weight:600">${{r.sku_id}}</td>
                <td>${{r.sku_name}}</td>
                <td>${{r.signal_pool}}</td>
                <td><span class="badge badge-amber">${{r.abc_xyz}}</span></td>
                <td style="color:${{r.ml_fa < 50 ? COLORS.red : COLORS.amber}}">${{r.ml_fa.toFixed(1)}}%</td>
                <td>${{r.mape_volatility.toFixed(1)}}</td>
                <td style="font-weight:700;color:${{r.risk_score > 60 ? COLORS.red : COLORS.amber}}">${{r.risk_score.toFixed(0)}}</td>
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
}}

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
            <div class="sub">Best performing model for this SKU</div>
        </div>
        <div class="kpi-card">
            <div class="label">Current FA%</div>
            <div class="value green">${{recFA.toFixed(1)}}%</div>
            <div class="sub">On test period (6 months)</div>
        </div>
        <div class="kpi-card">
            <div class="label">Improvement over Baseline</div>
            <div class="value" style="color:${{m.improvement > 0 ? COLORS.green : COLORS.red}}">${{m.improvement > 0 ? '+' : ''}}${{m.improvement.toFixed(1)}}%</div>
            <div class="sub">ML vs Baseline accuracy delta</div>
        </div>
        <div class="kpi-card">
            <div class="label">Avg Monthly Demand</div>
            <div class="value blue">${{Math.round(m.avg_demand)}}</div>
            <div class="sub">units/month</div>
        </div>
    </div>
    <div class="two-col">
        <div class="chart-container">
            <div class="chart-title">Historical Demand vs Forecasts — ${{m.sku_name}}</div>
            <div id="chart-ts-detail"></div>
        </div>
        <div class="chart-container">
            <div class="chart-title">Next 3-Month Forecast with Confidence Interval</div>
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

    // Future forecast
    if (future.length > 0) {{
        const futureDates = future.map(f => f.date.substring(0, 10));
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
                <strong>Parameters:</strong> Z = 1.65 (95% SL) | Lead Time = 30 days
            </div>
        </div>
        <div class="chart-container">
            <div class="chart-title">Safety Stock — All SKUs</div>
            <div id="chart-safety-all"></div>
        </div>
    </div>`;

    document.getElementById('safety-content').innerHTML = html;

    // All SKUs safety stock chart
    const sorted = [...metricsData].sort((a, b) => b.safety_stock_reduction - a.safety_stock_reduction);
    const top20 = sorted.slice(0, 20);
    Plotly.newPlot('chart-safety-all', [
        {{
            y: top20.map(s => s.sku_id),
            x: top20.map(s => s.baseline_safety_stock),
            type: 'bar', orientation: 'h', name: 'Baseline',
            marker: {{ color: COLORS.blue, opacity: 0.7 }},
        }},
        {{
            y: top20.map(s => s.sku_id),
            x: top20.map(s => s.ml_safety_stock),
            type: 'bar', orientation: 'h', name: 'ML Optimized',
            marker: {{ color: COLORS.green, opacity: 0.8 }},
        }},
    ], {{
        ...plotLayout,
        barmode: 'group',
        height: 500,
        margin: {{ l: 80, r: 30, t: 10, b: 40 }},
        xaxis: {{ ...plotLayout.xaxis, title: 'Safety Stock (units)' }},
        yaxis: {{ ...plotLayout.yaxis, autorange: 'reversed' }},
        legend: {{ ...plotLayout.legend, orientation: 'h', y: 1.05, x: 0.5, xanchor: 'center' }},
    }}, {{ responsive: true }});
}}

// ---- INIT ----
function init() {{
    populateSkuDropdowns();
    renderOverview();
    renderRisk();
    updateRecommendation();
    updateSafetyStock();
}}

document.addEventListener('DOMContentLoaded', init);
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
