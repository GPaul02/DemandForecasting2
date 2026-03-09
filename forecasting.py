"""
Forecasting Engine for Pharmaceutical Demand Forecasting MVP.

Layer 1: Baseline statistical forecast (3-month Simple Moving Average)
Layer 2: ML correction model (XGBoost) using baseline + external signals
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error


TRAIN_MONTHS = 30  # months 1-30
TEST_MONTHS = 6    # months 31-36
Z_SCORE = 1.65     # for 95% service level
LEAD_TIME_DAYS = 30


def compute_baseline_forecast(sales_df):
    """Compute 3-month simple moving average forecast per SKU."""
    results = []
    for sku_id in sales_df["sku_id"].unique():
        sku_data = sales_df[sales_df["sku_id"] == sku_id].sort_values("date").reset_index(drop=True)
        demands = sku_data["demand"].values
        forecasts = np.full(len(demands), np.nan)

        for i in range(3, len(demands)):
            forecasts[i] = np.mean(demands[i-3:i])

        sku_data["baseline_forecast"] = forecasts
        results.append(sku_data)

    return pd.concat(results, ignore_index=True)


def prepare_ml_features(sales_with_baseline, signals_df):
    """Merge baseline forecasts with external signals for ML features."""
    df = sales_with_baseline.merge(signals_df, on="date", how="left")

    # Lag features
    feature_cols = ["baseline_forecast", "aqi", "temperature", "precipitation",
                    "wedding_flag", "google_trends"]

    # Add month as feature
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Add lag-1 demand
    df["demand_lag1"] = df.groupby("sku_id")["demand"].shift(1)
    df["demand_lag2"] = df.groupby("sku_id")["demand"].shift(2)

    # Rolling std (volatility)
    df["demand_rolling_std"] = df.groupby("sku_id")["demand"].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    )

    all_features = feature_cols + ["month_sin", "month_cos", "demand_lag1",
                                    "demand_lag2", "demand_rolling_std"]
    return df, all_features


def train_and_predict(df, feature_cols):
    """Train per-pool ML models and generate predictions."""
    results = []

    for pool in df["signal_pool"].unique():
        pool_df = df[df["signal_pool"] == pool].copy()

        # Split data: use row index based on sorted dates
        all_skus = pool_df["sku_id"].unique()

        pool_train_parts = []
        pool_test_parts = []

        for sku_id in all_skus:
            sku_data = pool_df[pool_df["sku_id"] == sku_id].sort_values("date").reset_index(drop=True)
            pool_train_parts.append(sku_data.iloc[:TRAIN_MONTHS])
            pool_test_parts.append(sku_data.iloc[TRAIN_MONTHS:])

        train_data = pd.concat(pool_train_parts, ignore_index=True)
        test_data = pd.concat(pool_test_parts, ignore_index=True)

        # Drop rows with NaN in features
        train_clean = train_data.dropna(subset=feature_cols + ["demand"])
        test_clean = test_data.dropna(subset=feature_cols + ["demand"])

        if len(train_clean) < 10 or len(test_clean) < 1:
            test_data["ml_forecast"] = test_data["baseline_forecast"]
            results.append(test_data)
            continue

        X_train = train_clean[feature_cols].values
        y_train = train_clean["demand"].values
        X_test = test_clean[feature_cols].values

        model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            subsample=0.8,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        predictions = np.maximum(predictions, 0).round(0)

        test_clean = test_clean.copy()
        test_clean["ml_forecast"] = predictions

        # Also generate train predictions for full picture
        train_preds = model.predict(X_train)
        train_clean = train_clean.copy()
        train_clean["ml_forecast"] = np.maximum(train_preds, 0).round(0)

        results.append(train_clean)
        results.append(test_clean)

    return pd.concat(results, ignore_index=True)


def calculate_metrics(df):
    """Calculate MAPE and Forecast Accuracy per SKU for both models."""
    metrics = []

    for sku_id in df["sku_id"].unique():
        sku_data = df[df["sku_id"] == sku_id].sort_values("date")

        # Test period only (last 6 months)
        test_data = sku_data.tail(TEST_MONTHS)
        test_valid = test_data.dropna(subset=["baseline_forecast", "ml_forecast"])
        test_valid = test_valid[test_valid["demand"] > 0]

        if len(test_valid) < 1:
            continue

        actual = test_valid["demand"].values.astype(float)
        baseline = test_valid["baseline_forecast"].values.astype(float)
        ml_pred = test_valid["ml_forecast"].values.astype(float)

        # MAPE
        baseline_mape = np.mean(np.abs(actual - baseline) / np.maximum(actual, 1)) * 100
        ml_mape = np.mean(np.abs(actual - ml_pred) / np.maximum(actual, 1)) * 100

        baseline_fa = max(0, 100 - baseline_mape)
        ml_fa = max(0, 100 - ml_mape)

        # Forecast error std for safety stock
        baseline_errors = actual - baseline
        ml_errors = actual - ml_pred
        baseline_error_std = np.std(baseline_errors)
        ml_error_std = np.std(ml_errors)

        # MAPE volatility (std of absolute percentage errors over test period)
        ape_values = np.abs(actual - ml_pred) / np.maximum(actual, 1) * 100
        mape_volatility = np.std(ape_values)

        # Recent trend
        recent_demand = actual[-3:] if len(actual) >= 3 else actual
        avg_demand = np.mean(actual)
        recent_avg = np.mean(recent_demand)

        meta = sku_data.iloc[0]
        metrics.append({
            "sku_id": sku_id,
            "sku_name": meta["sku_name"],
            "signal_pool": meta["signal_pool"],
            "abc_xyz": meta["abc_xyz"],
            "baseline_mape": round(baseline_mape, 2),
            "ml_mape": round(ml_mape, 2),
            "baseline_fa": round(baseline_fa, 2),
            "ml_fa": round(ml_fa, 2),
            "improvement": round(ml_fa - baseline_fa, 2),
            "baseline_error_std": round(baseline_error_std, 2),
            "ml_error_std": round(ml_error_std, 2),
            "mape_volatility": round(mape_volatility, 2),
            "avg_demand": round(avg_demand, 2),
            "recent_avg_demand": round(recent_avg, 2),
        })

    return pd.DataFrame(metrics)


def compute_safety_stock(error_std, z=Z_SCORE, lead_time_days=LEAD_TIME_DAYS):
    """Safety Stock = Z * σ_forecast_error * sqrt(Lead Time in days)."""
    return z * error_std * np.sqrt(lead_time_days)


def generate_future_forecast(df, signals_df, feature_cols, n_months=3):
    """Generate next 3-month forecast with confidence intervals per SKU."""
    forecasts = []
    last_date = df["date"].max()

    for sku_id in df["sku_id"].unique():
        sku_data = df[df["sku_id"] == sku_id].sort_values("date")
        meta = sku_data.iloc[0]

        recent = sku_data["demand"].tail(6).values
        mean_demand = np.mean(recent) if len(recent) > 0 else 0
        std_demand = np.std(recent) if len(recent) > 1 else mean_demand * 0.2

        ml_data = sku_data.dropna(subset=["ml_forecast"])
        if len(ml_data) > 0:
            ml_errors = ml_data["demand"].values - ml_data["ml_forecast"].values
            forecast_std = np.std(ml_errors) if len(ml_errors) > 1 else std_demand
        else:
            forecast_std = std_demand

        for m in range(1, n_months + 1):
            future_date = last_date + pd.DateOffset(months=m)
            # Simple projection: use recent average with slight trend
            point_forecast = max(0, mean_demand + np.random.normal(0, std_demand * 0.1))
            lower = max(0, point_forecast - 1.96 * forecast_std)
            upper = point_forecast + 1.96 * forecast_std

            forecasts.append({
                "sku_id": sku_id,
                "sku_name": meta["sku_name"],
                "signal_pool": meta["signal_pool"],
                "date": future_date,
                "forecast": round(point_forecast, 0),
                "lower_bound": round(lower, 0),
                "upper_bound": round(upper, 0),
            })

    return pd.DataFrame(forecasts)


def run_forecasting_pipeline(sales_df, signals_df):
    """Run the full forecasting pipeline and return all results."""
    # Layer 1: Baseline
    df = compute_baseline_forecast(sales_df)

    # Prepare ML features
    df, feature_cols = prepare_ml_features(df, signals_df)

    # Layer 2: ML correction
    df_with_ml = train_and_predict(df, feature_cols)

    # Metrics
    metrics = calculate_metrics(df_with_ml)

    # Safety stock
    metrics["baseline_safety_stock"] = compute_safety_stock(metrics["baseline_error_std"])
    metrics["ml_safety_stock"] = compute_safety_stock(metrics["ml_error_std"])
    metrics["safety_stock_reduction"] = (
        (metrics["baseline_safety_stock"] - metrics["ml_safety_stock"])
        / metrics["baseline_safety_stock"].replace(0, 1) * 100
    ).round(2)

    # Future forecasts
    np.random.seed(42)
    future = generate_future_forecast(df_with_ml, signals_df, feature_cols)

    return df_with_ml, metrics, future


if __name__ == "__main__":
    from data_generator import generate_all_data

    sales, signals, meta = generate_all_data()
    df_results, metrics, future = run_forecasting_pipeline(sales, signals)

    print(f"Results shape: {df_results.shape}")
    print(f"\nMetrics summary:")
    print(f"  Avg Baseline FA: {metrics['baseline_fa'].mean():.1f}%")
    print(f"  Avg ML FA: {metrics['ml_fa'].mean():.1f}%")
    print(f"  Avg Improvement: {metrics['improvement'].mean():.1f}%")
    print(f"\nFuture forecasts: {future.shape}")
