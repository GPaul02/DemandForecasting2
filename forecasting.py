"""
Forecasting Engine for Pharmaceutical Demand Forecasting MVP.

Layer 1: Baseline statistical forecasts (3-month SMA + Exponential Smoothing)
Layer 2: Multi-model ML pipeline (Gradient Boosting, Random Forest, Ridge, Extra Trees)
Layer 3: Per-pool model selection with validation + weighted ensemble of top models
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Ridge


TRAIN_MONTHS = 30  # months 1-30
TEST_MONTHS = 6    # months 31-36
Z_SCORE = 1.65     # for 95% service level
LEAD_TIME_DAYS = 30

# Validation split: last 6 of training months used for model selection
VAL_MONTHS = 6

# Model registry — all candidate models with their configs
MODEL_REGISTRY = {
    "Gradient Boosting": lambda: GradientBoostingRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.1,
        random_state=42, subsample=0.8,
    ),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    ),
    "Extra Trees": lambda: ExtraTreesRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    ),
    "Ridge Regression": lambda: Ridge(alpha=1.0),
}


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


def compute_exp_smoothing_forecast(sales_df, alpha=0.3):
    """Compute Simple Exponential Smoothing forecast per SKU."""
    results = []
    for sku_id in sales_df["sku_id"].unique():
        sku_data = sales_df[sales_df["sku_id"] == sku_id].sort_values("date").reset_index(drop=True)
        demands = sku_data["demand"].values
        forecasts = np.full(len(demands), np.nan)

        if len(demands) > 1:
            level = float(demands[0])
            for i in range(1, len(demands)):
                forecasts[i] = level
                level = alpha * demands[i] + (1 - alpha) * level

        sku_data["exp_smoothing_forecast"] = forecasts
        results.append(sku_data)

    return pd.concat(results, ignore_index=True)


def prepare_ml_features(sales_with_baseline, signals_df):
    """Merge baseline forecasts with external signals for ML features."""
    df = sales_with_baseline.merge(signals_df, on="date", how="left")

    feature_cols = ["baseline_forecast", "aqi", "temperature", "precipitation",
                    "wedding_flag", "google_trends"]

    # Add month as feature
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Add lag-1/2 demand
    df["demand_lag1"] = df.groupby("sku_id")["demand"].shift(1)
    df["demand_lag2"] = df.groupby("sku_id")["demand"].shift(2)

    # Rolling std (volatility)
    df["demand_rolling_std"] = df.groupby("sku_id")["demand"].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    )

    # Add exponential smoothing as feature too
    if "exp_smoothing_forecast" in df.columns:
        feature_cols.append("exp_smoothing_forecast")

    all_features = feature_cols + ["month_sin", "month_cos", "demand_lag1",
                                    "demand_lag2", "demand_rolling_std"]
    return df, all_features


def _compute_mape(actual, predicted):
    """Compute MAPE avoiding division by zero."""
    mask = actual > 0
    if mask.sum() == 0:
        return 100.0
    return np.mean(np.abs(actual[mask] - predicted[mask]) / actual[mask]) * 100


def train_and_predict(df, feature_cols):
    """Train multiple ML models per pool, select best via validation, ensemble top models."""
    results = []
    model_selection_log = {}  # pool -> {model_name: val_mape, ...}

    for pool in df["signal_pool"].unique():
        pool_df = df[df["signal_pool"] == pool].copy()
        all_skus = pool_df["sku_id"].unique()

        pool_train_parts = []
        pool_val_parts = []
        pool_test_parts = []

        for sku_id in all_skus:
            sku_data = pool_df[pool_df["sku_id"] == sku_id].sort_values("date").reset_index(drop=True)
            train_end = TRAIN_MONTHS - VAL_MONTHS
            pool_train_parts.append(sku_data.iloc[:train_end])
            pool_val_parts.append(sku_data.iloc[train_end:TRAIN_MONTHS])
            pool_test_parts.append(sku_data.iloc[TRAIN_MONTHS:])

        train_data = pd.concat(pool_train_parts, ignore_index=True)
        val_data = pd.concat(pool_val_parts, ignore_index=True)
        test_data = pd.concat(pool_test_parts, ignore_index=True)

        train_clean = train_data.dropna(subset=feature_cols + ["demand"])
        val_clean = val_data.dropna(subset=feature_cols + ["demand"])
        test_clean = test_data.dropna(subset=feature_cols + ["demand"])

        if len(train_clean) < 10 or len(test_clean) < 1:
            test_data["ml_forecast"] = test_data["baseline_forecast"]
            test_data["best_model"] = "Baseline (3M SMA)"
            test_data["model_scores"] = "{}"
            results.append(test_data)
            model_selection_log[pool] = {"Baseline (3M SMA)": 0}
            continue

        X_train = train_clean[feature_cols].values
        y_train = train_clean["demand"].values
        X_val = val_clean[feature_cols].values if len(val_clean) > 0 else None
        y_val = val_clean["demand"].values if len(val_clean) > 0 else None
        X_test = test_clean[feature_cols].values

        # Retrain on full training set (train + val) for final predictions
        full_train = pd.concat([train_clean, val_clean], ignore_index=True)
        X_full = full_train[feature_cols].values
        y_full = full_train["demand"].values

        # ── Train all candidate models ──
        trained_models = {}
        val_scores = {}  # model_name -> val_mape

        for model_name, model_fn in MODEL_REGISTRY.items():
            try:
                model = model_fn()
                model.fit(X_train, y_train)

                # Validate
                if X_val is not None and len(X_val) > 0:
                    val_preds = np.maximum(model.predict(X_val), 0)
                    val_mape = _compute_mape(y_val, val_preds)
                else:
                    # Fallback: use training MAPE (less ideal)
                    train_preds = np.maximum(model.predict(X_train), 0)
                    val_mape = _compute_mape(y_train, train_preds)

                val_scores[model_name] = round(val_mape, 2)
                trained_models[model_name] = model
            except Exception:
                # Skip models that fail (e.g., singular matrix for Ridge)
                continue

        if not trained_models:
            test_data["ml_forecast"] = test_data["baseline_forecast"]
            test_data["best_model"] = "Baseline (3M SMA)"
            test_data["model_scores"] = "{}"
            results.append(test_data)
            model_selection_log[pool] = {"Baseline (3M SMA)": 0}
            continue

        # ── Model selection: pick best + build ensemble ──
        sorted_models = sorted(val_scores.items(), key=lambda x: x[1])
        best_model_name = sorted_models[0][0]
        best_val_mape = sorted_models[0][1]

        # Retrain ALL models on full training data (train + val)
        final_models = {}
        for model_name in trained_models:
            model = MODEL_REGISTRY[model_name]()
            model.fit(X_full, y_full)
            final_models[model_name] = model

        # ── Generate predictions ──
        # Weighted ensemble: inverse-MAPE weighting of top models
        top_entries = sorted_models[:3]
        # Convert MAPE to weights (lower MAPE = higher weight)
        mapes = np.array([max(entry[1], 0.1) for entry in top_entries])
        inv_mapes = 1.0 / mapes
        weights = inv_mapes / inv_mapes.sum()

        ensemble_test_preds = np.zeros(len(X_test))
        ensemble_train_preds = np.zeros(len(X_full))

        for i, (model_name, _) in enumerate(top_entries):
            model = final_models[model_name]
            ensemble_test_preds += weights[i] * model.predict(X_test)
            ensemble_train_preds += weights[i] * model.predict(X_full)

        ensemble_test_preds = np.maximum(ensemble_test_preds, 0).round(0)
        ensemble_train_preds = np.maximum(ensemble_train_preds, 0).round(0)

        # Also get solo best-model predictions for comparison
        best_model = final_models[best_model_name]
        solo_test_preds = np.maximum(best_model.predict(X_test), 0).round(0)
        solo_train_preds = np.maximum(best_model.predict(X_full), 0).round(0)

        # Pick ensemble vs solo: whichever has lower test-set MAPE
        # (Use validation set actual for this decision since test is hold-out)
        if y_val is not None and len(y_val) > 0:
            # Re-predict val with ensemble for comparison
            ens_val_preds = np.zeros(len(X_val))
            for i, (model_name, _) in enumerate(top_entries):
                # Use the models trained on train-only for fair comparison
                try:
                    m = MODEL_REGISTRY[model_name]()
                    m.fit(X_train, y_train)
                    ens_val_preds += weights[i] * m.predict(X_val)
                except Exception:
                    ens_val_preds += weights[i] * np.full(len(X_val), np.mean(y_train))

            ens_val_mape = _compute_mape(y_val, np.maximum(ens_val_preds, 0))
            use_ensemble = ens_val_mape < best_val_mape
        else:
            use_ensemble = len(top_entries) > 1

        if use_ensemble and len(top_entries) > 1:
            final_test_preds = ensemble_test_preds
            final_train_preds = ensemble_train_preds
            chosen_name = f"Ensemble ({', '.join(n for n, _ in top_entries)})"
        else:
            final_test_preds = solo_test_preds
            final_train_preds = solo_train_preds
            chosen_name = best_model_name

        # Compute per-SKU test-period MAPEs for ALL models (consistent with baseline_fa/ml_fa)
        sku_test_scores = {}
        for sku_id in all_skus:
            sku_test = test_clean[test_clean["sku_id"] == sku_id]
            if len(sku_test) == 0:
                continue
            sku_feats = sku_test.dropna(subset=feature_cols + ["demand"])
            if len(sku_feats) == 0:
                continue
            X_sku = sku_feats[feature_cols].values
            y_sku = sku_feats["demand"].values.astype(float)
            scores_for_sku = {}
            for mname, mobj in final_models.items():
                preds = np.maximum(mobj.predict(X_sku), 0)
                mape = _compute_mape(y_sku, preds)
                scores_for_sku[mname] = round(mape, 2)
            sku_test_scores[sku_id] = scores_for_sku

        test_clean = test_clean.copy()
        test_clean["ml_forecast"] = final_test_preds
        test_clean["best_model"] = chosen_name
        # Per-SKU test-period MAPEs (not pool-level validation MAPEs)
        test_clean["model_scores"] = test_clean["sku_id"].map(
            lambda sid: str(sku_test_scores.get(sid, val_scores))
        )

        full_train_out = full_train.copy()
        full_train_out["ml_forecast"] = final_train_preds
        full_train_out["best_model"] = chosen_name
        full_train_out["model_scores"] = str(val_scores)

        results.append(full_train_out)
        results.append(test_clean)

        model_selection_log[pool] = {
            "val_scores": val_scores,
            "best_single": best_model_name,
            "chosen": chosen_name,
            "weights": {n: round(float(w), 3) for (n, _), w in zip(top_entries, weights)},
        }

    combined = pd.concat(results, ignore_index=True)
    return combined, model_selection_log


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

        # MAPE volatility
        ape_values = np.abs(actual - ml_pred) / np.maximum(actual, 1) * 100
        mape_volatility = np.std(ape_values)

        # Recent trend
        recent_demand = actual[-3:] if len(actual) >= 3 else actual
        avg_demand = np.mean(actual)
        recent_avg = np.mean(recent_demand)

        # Exp smoothing accuracy (if available)
        exp_sm_fa = None
        if "exp_smoothing_forecast" in test_valid.columns:
            exp_sm = test_valid["exp_smoothing_forecast"].values.astype(float)
            exp_sm_valid = ~np.isnan(exp_sm)
            if exp_sm_valid.sum() > 0:
                exp_sm_mape = np.mean(np.abs(actual[exp_sm_valid] - exp_sm[exp_sm_valid]) / np.maximum(actual[exp_sm_valid], 1)) * 100
                exp_sm_fa = round(max(0, 100 - exp_sm_mape), 2)

        # Model info
        best_model = "Unknown"
        model_scores = {}
        if "best_model" in sku_data.columns:
            bm = sku_data["best_model"].dropna()
            if len(bm) > 0:
                best_model = bm.iloc[-1]
        if "model_scores" in sku_data.columns:
            ms = sku_data["model_scores"].dropna()
            if len(ms) > 0:
                try:
                    raw = eval(ms.iloc[-1])
                    model_scores = {k: round(float(v), 2) for k, v in raw.items()}
                except Exception:
                    model_scores = {}

        meta = sku_data.iloc[0]
        entry = {
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
            "best_model": best_model,
            "model_scores": model_scores,
        }
        if exp_sm_fa is not None:
            entry["exp_smoothing_fa"] = exp_sm_fa

        metrics.append(entry)

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
    # Layer 1: Baseline forecasts
    df = compute_baseline_forecast(sales_df)

    # Exponential Smoothing baseline
    es_df = compute_exp_smoothing_forecast(sales_df, alpha=0.3)
    df = df.merge(
        es_df[["sku_id", "date", "exp_smoothing_forecast"]],
        on=["sku_id", "date"], how="left"
    )

    # Prepare ML features
    df, feature_cols = prepare_ml_features(df, signals_df)

    # Layer 2: Multi-model ML with selection + ensemble
    df_with_ml, model_log = train_and_predict(df, feature_cols)

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

    return df_with_ml, metrics, future, model_log


if __name__ == "__main__":
    from data_generator import generate_all_data

    sales, signals, meta = generate_all_data()
    df_results, metrics, future, model_log = run_forecasting_pipeline(sales, signals)

    print(f"Results shape: {df_results.shape}")
    print(f"\nMetrics summary:")
    print(f"  Avg Baseline FA: {metrics['baseline_fa'].mean():.1f}%")
    print(f"  Avg ML FA: {metrics['ml_fa'].mean():.1f}%")
    print(f"  Avg Improvement: {metrics['improvement'].mean():.1f}%")
    print(f"\nModel selection per pool:")
    for pool, info in model_log.items():
        print(f"  {pool}: {info['chosen']} (best single: {info['best_single']})")
        print(f"    Val scores: {info.get('val_scores', {})}")
    print(f"\nBest model distribution:")
    print(metrics["best_model"].value_counts().to_string())
    print(f"\nFuture forecasts: {future.shape}")
