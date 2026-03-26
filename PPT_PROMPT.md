# Prompt for generating a presentation on PharmaCast

Copy everything below this line and paste it into any LLM (ChatGPT, Claude, Gemini, etc.) to generate a professional presentation.

---

You are a presentation design expert. Create a detailed, visually-described, slide-by-slide PowerPoint presentation for the project described below. For each slide, provide: the slide title, the layout description, all text content (including speaker notes), and describe any visuals, diagrams, charts, or icons that should be placed on the slide. The presentation should be professional, clean, and suitable for an academic/corporate audience. Use a modern color scheme anchored on forest green (#2d8a4e), warm off-white (#f5f5f0), and dark charcoal (#1e1e1e). Total slides: 20-25.

The audience is a mix of professors, supply chain professionals, and technical teammates. The tone should be confident, clear, and insight-driven — not just describing what we built, but why each design decision matters and what impact it creates.

---

## PROJECT: PharmaCast — Pharmaceutical Demand Forecasting Decision Support System

### What this project is

PharmaCast is an end-to-end Decision Support System for pharmaceutical demand forecasting. It targets the hardest-to-forecast SKU segments in pharmaceutical supply chains — specifically BZ, CY, and CZ classes in the ABC-XYZ inventory classification — and uses a combination of baseline statistical methods, machine learning models, and real-world external signals to dramatically improve forecast accuracy and optimize safety stock levels.

The entire system runs as a Python pipeline that generates a self-contained HTML dashboard (no server, no installation, just open in a browser) with interactive Plotly.js charts, risk analysis, model recommendations, safety stock optimization, and a built-in chatbot assistant.

---

### THE PROBLEM (Motivation)

Pharmaceutical supply chains face a critical forecasting challenge:

1. **High-value A-class SKUs** get all the attention — sophisticated models, dedicated analysts, regular reviews. But they represent only 10-20% of total SKU count.

2. **B-class and C-class SKUs** (medium-to-low revenue items) are the bulk of the catalog (60-80% of SKUs). They still have strict service-level requirements (patients need medication regardless of its revenue class), but they receive minimal forecasting effort.

3. **Y and Z variability classes** have erratic or intermittent demand patterns. Simple moving averages and naive forecasts perform poorly on these items because the demand is inherently noisy, sporadic, or event-driven.

4. **BZ, CY, CZ segments** sit at the intersection of these two challenges: low attention AND high variability. These are the SKUs most likely to cause stockouts, emergency orders, and patient impact.

5. **External signals matter but are ignored**: Respiratory drug demand surges when AQI spikes. Anti-malarial demand rises during monsoon. Electrolyte demand tracks temperature. Supplement demand spikes during wedding season. These correlations exist in the real world but traditional forecasting systems ignore them entirely.

PharmaCast addresses all of these problems simultaneously.

---

### THE SOLUTION (Architecture)

PharmaCast is a 3-step Python pipeline:

**Step 1: Data Generation** (`data_generator.py`)
- Generates 36 months of synthetic sales data (May 2022 – April 2025) for 40 pharmaceutical SKUs
- 40 SKUs organized into 5 signal pools (8 SKUs each):
  - **AQI Pool**: Respiratory and antihistamine drugs (Salbutamol Inhaler, Cetirizine 10mg, Montelukast 10mg, Budesonide Inhaler, Levocetrizine 5mg, Fluticasone Spray, Theophylline 300mg, Chlorpheniramine 4mg). Demand correlates with Air Quality Index — when AQI spikes (winter smog, pollution events), respiratory drug demand surges.
  - **Monsoon Pool**: Anti-malarials and gastrointestinal drugs (ORS Sachets, Loperamide 2mg, Chloroquine 500mg, Artemether-Lumefantrine, Zinc Dispersible 20mg, Metronidazole 400mg, Doxycycline 100mg, Racecadotril 100mg). Demand correlates with precipitation — heavy monsoon rainfall drives malaria and waterborne illness outbreaks.
  - **Temperature Pool**: Electrolytes, dermatological, and heat-related products (Electrolyte Powder, Calamine Lotion, Sunscreen SPF50, Glucose-D Powder, Hydrocortisone Cream, Prickly Heat Powder, Isotonic Drink Mix, Aloe Vera Gel). Demand correlates with temperature — extreme heat drives dehydration and skin conditions.
  - **Wedding Pool**: Vitamins, nutraceuticals, and supplements (Multivitamin Tablets, Biotin 10000mcg, Omega-3 Capsules, Vitamin C 1000mg, Collagen Powder, Iron + Folic Acid, Calcium + D3, Protein Supplement). Demand spikes during Indian wedding season (November–December and April–May) as people purchase health and beauty supplements.
  - **Google Trends Pool**: Symptom-search-linked drugs (Paracetamol 500mg, Azithromycin 500mg, Ivermectin 12mg, Dolo 650mg, Vitamin D3 60K, Zinc 50mg, Cough Syrup 100ml, Throat Lozenges). Demand correlates with Google search trends for symptoms — when people search for "cough," "fever," or "cold," demand for these OTC medications rises.

- External signals generated with realistic Indian seasonality:
  - AQI: Range 30–500, peaks in winter (Oct–Feb)
  - Temperature: Range 15–45°C, Indian climate pattern (peak May–Jun, low Dec–Jan)
  - Precipitation: Range 0–300mm, monsoon peaks Jul–Sep
  - Wedding Flag: Binary (0/1), active in Nov–Dec and Apr–May
  - Google Trends: Range 0–100, baseline + seasonal + 15% random spike probability

- Signal correlation factors (how much each signal affects demand):
  - AQI: ±15% of base demand (factor: 0.15 × (aqi − 120) / 100)
  - Monsoon: ±20% of base demand (factor: 0.20 × (precipitation − 50) / 200)
  - Temperature: ±18% of base demand (factor: 0.18 × (temp − 28) / 10)
  - Wedding: ±25% of base demand (factor: 0.25 × wedding_flag)
  - Google Trends: ±12% of base demand (factor: 0.12 × (trends − 40) / 30)

- ABC-XYZ classification: Only BZ, CY, CZ segments. B-class base demand: 300–800 units/month. C-class base demand: 80–350 units/month. Distribution: 15 BZ, 15 CY, 10 CZ across the 40 SKUs.

- Three demand patterns: intermittent (85% normal, 15% near-zero), erratic (high CoV, σ = 0.3 × base), seasonal (sinusoidal with random peak month).

- Total data: 1,440 records (40 SKUs × 36 months), fully deterministic with seed=42.

**Step 2: Forecasting Pipeline** (`forecasting.py`)

This is the core intelligence layer with 3 sub-layers:

*Layer 1 — Baseline Statistical Forecasts:*
- 3-Month Simple Moving Average (SMA): forecast[t] = mean(demand[t-3 : t])
- Exponential Smoothing (α = 0.3): level[t] = 0.3 × demand[t] + 0.7 × level[t-1], forecast[t] = level[t-1]
- These serve dual purpose: (a) benchmark to beat, and (b) features for ML models.

*Layer 2 — ML Feature Engineering (12 features):*
1. baseline_forecast (3-month SMA output)
2. exp_smoothing_forecast (exponential smoothing output)
3. aqi (external signal)
4. temperature (external signal)
5. precipitation (external signal)
6. wedding_flag (external signal, binary)
7. google_trends (external signal)
8. month_sin = sin(2π × month / 12) — cyclical month encoding
9. month_cos = cos(2π × month / 12) — cyclical month encoding
10. demand_lag1 (demand at t-1)
11. demand_lag2 (demand at t-2)
12. demand_rolling_std (3-month rolling standard deviation of demand)

*Layer 3 — Multi-Model ML Pipeline with Per-Pool Selection:*

4 candidate models compete:
- Gradient Boosting: GradientBoostingRegressor (150 trees, max_depth=4, learning_rate=0.1, subsample=0.8)
- Random Forest: RandomForestRegressor (200 trees, max_depth=6, min_samples_leaf=3)
- Extra Trees: ExtraTreesRegressor (200 trees, max_depth=6, min_samples_leaf=3)
- Ridge Regression: Ridge (alpha=1.0)

Critical design decision: **Model selection is per signal pool, not global.** Each pool has fundamentally different demand dynamics (weather-driven vs. event-driven vs. search-driven), so a one-size-fits-all model would be suboptimal.

Selection process per pool:
1. Split: Train (months 1–24), Validation (months 25–30), Test (months 31–36, held out)
2. Train all 4 models on months 1–24
3. Evaluate all 4 on validation set (months 25–30) using MAPE
4. Rank by validation MAPE
5. Retrain ALL models on months 1–30 (train + validation combined)
6. Build weighted ensemble of top 3 models: weight[i] = (1/MAPE[i]) / Σ(1/MAPE[j])
7. Compare ensemble vs best single model on validation: use whichever has lower MAPE
8. Generate final test predictions (months 31–36)

Per-SKU metrics computed (test period):
- MAPE = mean(|actual − forecast| / actual) × 100
- Forecast Accuracy (FA) = max(0, 100 − MAPE)
- Improvement = ML_FA − Baseline_FA
- Error Std = std(actual − forecast)
- MAPE Volatility = std(individual APE values)
- Risk Score = (100 − ML_FA) + MAPE_Volatility

Safety stock calculation:
- Formula: SS = Z × σ_forecast_error × √(Lead_Time_days)
- Z = 1.65 (95% service level), Lead Time = 30 days
- Therefore: SS ≈ 9.04 × σ
- Reduction% = (Baseline_SS − ML_SS) / Baseline_SS × 100
- Better ML accuracy → lower σ → lower safety stock → freed working capital

Future forecasting (3-month ahead, months 37–39):
- Point forecast = mean(last 6 months) + small noise
- 95% CI = point ± 1.96 × forecast_std

**Step 3: Dashboard Generation** (`dashboard.py`, ~3000 lines)

Generates a single self-contained HTML file (~300KB) with embedded Plotly.js charts, CSS, and JavaScript. No server needed.

Design system (Apple-inspired, premium feel):
- Warm off-white background (#f5f5f0)
- Borderless white cards with soft shadows
- Muted color palette: Green #2d8a4e, Red #bf4342, Amber #b86e00, Blue #2c6fce
- 96px hero metric, clear typography hierarchy
- Sentence case, no emojis, human-readable language
- Plotly toolbar hidden for clean presentation
- 20px border radius on all cards

Dashboard has 4 tabs + floating chat assistant:

**Tab 1 — Overview:** "How well are my forecasts performing?"
- 96px hero metric showing overall improvement (e.g., "+8.9%")
- 8 KPI cards (Total SKUs, Avg Baseline FA, Avg ML FA, Avg Improvement, SKUs Improved, At-Risk SKUs, Highest Risk Score, Signal Anomalies)
- Intelligence briefing with amber left-border and action pill buttons
- Pool filter pills for cross-filtering
- Scatter plot: Baseline FA vs ML FA per SKU (above diagonal = ML wins)
- Bar chart: accuracy by signal pool
- Donut chart: model selection distribution
- Accuracy distribution histogram

**Tab 2 — Risk Flags:** "Which SKUs need attention?"
- Headline: "N SKUs need attention" (FA < 70%)
- Top 3 risk insight cards with colored severity stripes (red/amber)
- Sortable risk data table with all metrics
- Risk score tooltip explaining the formula
- Signal anomaly alerts (AQI > 250, Temp > 40°C, Precipitation > 200mm, Google Trends > 75)

**Tab 3 — Model Recommendations:** "Which model should I use for each SKU?"
- Searchable SKU dropdown with risk indicator dots
- Model comparison bar chart (test MAPE for all 4 models)
- Actual vs Baseline vs ML time series
- Residual analysis
- Future forecast with confidence bands
- 7-step expandable methodology section

**Tab 4 — Safety Stock:** "How much safety stock do I need?"
- Comparison cards: Total Baseline SS vs ML SS vs Reduction
- What-if simulator with interactive sliders (service level: 90/95/99%, lead time: 15/30/45/60 days, error adjustment)
- Top 10 reduction bar chart

**Chat Assistant:** Floating panel with keyword-based NLP intent routing
- Can answer queries about any SKU, pool, risk status, safety stock, formulas, and cross-tab diagnosis
- Suggestion chips for quick-start: "Diagnose root causes", "Which SKUs are at risk?", "Best performing pool?", "Safety stock savings"
- No external API calls — all logic runs client-side in JavaScript

---

### TECHNOLOGY STACK

| Component | Technology |
|-----------|-----------|
| Language | Python 3.x |
| Data Processing | NumPy, Pandas |
| Machine Learning | Scikit-learn |
| Visualization | Plotly.js 2.27.0 |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Deployment | Self-contained HTML file |

Dependencies: numpy>=1.24.0, pandas>=2.0.0, scikit-learn>=1.3.0

---

### KEY NUMBERS FOR THE PRESENTATION

| Metric | Value |
|--------|-------|
| Total SKUs | 40 (8 per pool) |
| Signal Pools | 5 |
| Data Period | 36 months |
| Data Records | 1,440 |
| ML Features | 12 |
| Candidate Models | 4 |
| Training Period | 24 months (months 1–24) |
| Validation Period | 6 months (months 25–30) |
| Test Period | 6 months (months 31–36) |
| Ensemble Size | Top 3 (inverse-MAPE weighted) |
| Service Level | 95% (Z = 1.65) |
| Lead Time | 30 days |
| At-Risk Threshold | FA < 70% |
| Future Horizon | 3 months |
| Dashboard Tabs | 4 + chatbot |
| Dashboard Size | ~300KB single HTML file |

---

### PRESENTATION STRUCTURE (Suggested)

Please create slides following this flow:

1. **Title Slide** — PharmaCast: Pharmaceutical Demand Forecasting DSS
2. **The Problem** — Why BZ/CY/CZ segments are the blind spot in pharma supply chains (2-3 slides with real-world context about stockouts, patient impact, wasted working capital)
3. **Our Insight** — External signals (AQI, weather, cultural events, search trends) can predict demand shifts that traditional methods miss
4. **Solution Overview** — High-level architecture diagram (3-step pipeline)
5. **Data Universe** — 40 SKUs across 5 signal pools with realistic pharmaceutical context (show the pool-to-drug mapping, explain why each signal drives demand for its drug category)
6. **Signal Correlations** — How each external signal affects demand (visual: signal time series overlaid with demand curves)
7. **ABC-XYZ Classification** — What BZ/CY/CZ means and why it matters (visual: 2x2 matrix with quadrant highlighting)
8. **Forecasting Pipeline** — Layer 1: Baselines, Layer 2: Feature Engineering, Layer 3: ML Models (3 slides, one per layer)
9. **Per-Pool Model Selection** — The critical design decision (flowchart showing the selection process)
10. **Ensemble Strategy** — Inverse-MAPE weighting (visual: weight distribution example)
11. **Results & Impact** — Key metrics: FA improvement, safety stock reduction, at-risk identification (2-3 slides with charts)
12. **Safety Stock Optimization** — Formula, before vs after, working capital freed
13. **Dashboard Walkthrough** — Screenshots/descriptions of each tab (4 slides)
14. **Chat Assistant** — Built-in NLP query engine
15. **Design Philosophy** — Apple-inspired, human-readable, premium feel (1 slide on design decisions)
16. **Technology Stack** — Clean tech stack slide
17. **Key Innovations** — What makes PharmaCast different (per-pool selection, signal integration, self-contained deployment, what-if simulator)
18. **Future Scope** — Real data integration, API deployment, more signals, deep learning models
19. **Thank You / Q&A**

For each slide, provide:
- Exact title
- All bullet points or text content
- Description of any visual/chart/diagram to include
- Speaker notes (what to say when presenting this slide)
- Any animation suggestions

Make the presentation tell a story: Problem → Insight → Solution → Evidence → Impact.
