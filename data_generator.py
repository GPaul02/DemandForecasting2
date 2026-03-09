"""
Synthetic Data Generator for Pharmaceutical Demand Forecasting MVP.

Generates 36 months of synthetic sales data for 40 SKUs (8 per signal pool)
with corresponding external signal data (AQI, Temperature, Precipitation,
Wedding Season, Google Trends).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SIGNAL_POOLS = {
    "AQI": {
        "description": "Respiratory, antihistamine, bronchodilator SKUs",
        "sku_prefix": "AQI",
        "categories": [
            "Salbutamol Inhaler", "Cetirizine 10mg", "Montelukast 10mg",
            "Budesonide Inhaler", "Levocetrizine 5mg", "Fluticasone Spray",
            "Theophylline 300mg", "Chlorpheniramine 4mg"
        ],
    },
    "Monsoon": {
        "description": "Anti-malarials, ORS, anti-diarrhoeals",
        "sku_prefix": "MON",
        "categories": [
            "ORS Sachets", "Loperamide 2mg", "Chloroquine 500mg",
            "Artemether-Lumefantrine", "Zinc Dispersible 20mg", "Metronidazole 400mg",
            "Doxycycline 100mg", "Racecadotril 100mg"
        ],
    },
    "Temperature": {
        "description": "Electrolytes, heat-stroke, dermatological SKUs",
        "sku_prefix": "TMP",
        "categories": [
            "Electrolyte Powder", "Calamine Lotion", "Sunscreen SPF50",
            "Glucose-D Powder", "Hydrocortisone Cream", "Prickly Heat Powder",
            "Isotonic Drink Mix", "Aloe Vera Gel"
        ],
    },
    "Wedding": {
        "description": "Vitamins, nutraceuticals, supplements",
        "sku_prefix": "WED",
        "categories": [
            "Multivitamin Tablets", "Biotin 10000mcg", "Omega-3 Capsules",
            "Vitamin C 1000mg", "Collagen Powder", "Iron + Folic Acid",
            "Calcium + D3", "Protein Supplement"
        ],
    },
    "GoogleTrends": {
        "description": "Symptom-search spike linked SKUs",
        "sku_prefix": "GTR",
        "categories": [
            "Paracetamol 500mg", "Azithromycin 500mg", "Ivermectin 12mg",
            "Dolo 650mg", "Vitamin D3 60K", "Zinc 50mg",
            "Cough Syrup 100ml", "Throat Lozenges"
        ],
    },
}

# ABC-XYZ classifications for our target segments
TARGET_CLASSIFICATIONS = ["BZ", "CY", "CZ"]


def generate_date_range():
    """Generate 36 monthly periods from May 2022 to June 2025 (inclusive of Apr 2025)."""
    dates = pd.date_range(start="2022-05-01", periods=36, freq="MS")
    return dates


def generate_external_signals(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Generate synthetic external signal data for each month."""
    n = len(dates)
    months = dates.month.values

    # AQI: higher in winter (Oct-Feb) due to pollution + crop burning
    aqi_base = 120 + 80 * np.sin(2 * np.pi * (months - 11) / 12)
    aqi = np.clip(aqi_base + rng.normal(0, 40, n), 30, 500).round(1)

    # Temperature (°C): peaks in May-June, low in Dec-Jan (Indian climate)
    temp_base = 30 + 10 * np.sin(2 * np.pi * (months - 6) / 12)
    temperature = np.clip(temp_base + rng.normal(0, 2, n), 15, 45).round(1)

    # Precipitation (mm): peaks Jul-Sep (monsoon)
    precip_base = np.where((months >= 6) & (months <= 9), 180 + 60 * rng.random(n), 20 + 15 * rng.random(n))
    precipitation = np.clip(precip_base + rng.normal(0, 20, n), 0, 300).round(1)

    # Wedding season: peaks in Nov-Dec and Apr-May (Indian wedding seasons)
    wedding_flag = np.where((months == 11) | (months == 12) | (months == 4) | (months == 5), 1, 0)

    # Google Trends: random spikes with some seasonality
    gtrends_base = 40 + 15 * np.sin(2 * np.pi * months / 12) + rng.normal(0, 10, n)
    # Add occasional spikes
    spike_mask = rng.random(n) > 0.85
    gtrends_base[spike_mask] += rng.uniform(20, 40, spike_mask.sum())
    google_trends = np.clip(gtrends_base, 0, 100).round(1)

    return pd.DataFrame({
        "date": dates,
        "aqi": aqi,
        "temperature": temperature,
        "precipitation": precipitation,
        "wedding_flag": wedding_flag,
        "google_trends": google_trends,
    })


def _demand_pattern_intermittent(n, base, rng):
    """Intermittent demand: occasional low periods."""
    demand = np.ones(n) * base
    for i in range(n):
        if rng.random() > 0.85:
            demand[i] = base * rng.uniform(0.2, 0.5)
        else:
            demand[i] = base * rng.uniform(0.8, 1.3)
    return demand


def _demand_pattern_erratic(n, base, rng):
    """Erratic demand: moderate coefficient of variation."""
    return base * np.abs(rng.normal(1.0, 0.3, n))


def _demand_pattern_seasonal(n, base, rng, peak_month=7):
    """Seasonal demand with noise."""
    months = np.arange(1, n + 1) % 12 + 1
    seasonal = 1 + 0.4 * np.sin(2 * np.pi * (months - peak_month) / 12)
    return base * seasonal * rng.uniform(0.85, 1.15, n)


def generate_sku_sales(dates, signals_df, rng):
    """Generate sales data for all 40 SKUs with signal-correlated demand."""
    n = len(dates)
    all_records = []
    sku_metadata = []

    patterns = [_demand_pattern_intermittent, _demand_pattern_erratic, _demand_pattern_seasonal]

    for pool_name, pool_info in SIGNAL_POOLS.items():
        for i, category in enumerate(pool_info["categories"]):
            sku_id = f"{pool_info['sku_prefix']}-{i+1:03d}"
            abc_xyz = TARGET_CLASSIFICATIONS[i % len(TARGET_CLASSIFICATIONS)]
            abc = abc_xyz[0]
            xyz = abc_xyz[1]

            # Base demand level: B-class higher than C-class
            base_demand = rng.uniform(300, 800) if abc == "B" else rng.uniform(80, 350)

            # Choose demand pattern
            pattern_fn = patterns[i % len(patterns)]
            peak = rng.integers(1, 12)
            if pattern_fn == _demand_pattern_seasonal:
                raw_demand = pattern_fn(n, base_demand, rng, peak_month=peak)
            else:
                raw_demand = pattern_fn(n, base_demand, rng)

            # Add signal-driven correlation
            signal_effect = np.zeros(n)
            if pool_name == "AQI":
                signal_effect = 0.15 * (signals_df["aqi"].values - 120) / 100 * base_demand
            elif pool_name == "Monsoon":
                signal_effect = 0.2 * (signals_df["precipitation"].values - 50) / 200 * base_demand
            elif pool_name == "Temperature":
                signal_effect = 0.18 * (signals_df["temperature"].values - 28) / 10 * base_demand
            elif pool_name == "Wedding":
                signal_effect = 0.25 * signals_df["wedding_flag"].values * base_demand
            elif pool_name == "GoogleTrends":
                signal_effect = 0.12 * (signals_df["google_trends"].values - 40) / 30 * base_demand

            demand = np.maximum(raw_demand + signal_effect, 0).round(0).astype(int)

            for j in range(n):
                all_records.append({
                    "date": dates[j],
                    "sku_id": sku_id,
                    "sku_name": category,
                    "signal_pool": pool_name,
                    "abc_class": abc,
                    "xyz_class": xyz,
                    "abc_xyz": abc_xyz,
                    "demand": int(demand[j]),
                })

            sku_metadata.append({
                "sku_id": sku_id,
                "sku_name": category,
                "signal_pool": pool_name,
                "abc_class": abc,
                "xyz_class": xyz,
                "abc_xyz": abc_xyz,
            })

    sales_df = pd.DataFrame(all_records)
    metadata_df = pd.DataFrame(sku_metadata)
    return sales_df, metadata_df


def generate_all_data(seed=42):
    """Main entry point: generates all synthetic data and returns DataFrames."""
    rng = np.random.default_rng(seed)
    dates = generate_date_range()
    signals_df = generate_external_signals(dates, rng)
    sales_df, metadata_df = generate_sku_sales(dates, signals_df, rng)
    return sales_df, signals_df, metadata_df


if __name__ == "__main__":
    sales, signals, meta = generate_all_data()
    print(f"Sales data: {sales.shape}")
    print(f"Signals data: {signals.shape}")
    print(f"SKU metadata: {meta.shape}")
    print(f"\nSample sales:\n{sales.head(10)}")
    print(f"\nSignal pools: {meta['signal_pool'].value_counts().to_dict()}")
    print(f"ABC-XYZ classes: {meta['abc_xyz'].value_counts().to_dict()}")
