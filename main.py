"""
Main orchestrator for the Pharmaceutical Demand Forecasting MVP.

Generates synthetic data, runs forecasting pipeline, and builds
a self-contained HTML dashboard.
"""

from data_generator import generate_all_data
from forecasting import run_forecasting_pipeline
from dashboard import generate_html_dashboard


def main():
    print("=" * 60)
    print("  PharmaCast — Demand Forecasting Decision Support System")
    print("=" * 60)

    # Step 1: Generate synthetic data
    print("\n[1/3] Generating synthetic data for 40 SKUs (36 months)...")
    sales_df, signals_df, metadata_df = generate_all_data(seed=42)
    print(f"  -> Sales records: {len(sales_df):,}")
    print(f"  -> SKUs: {len(metadata_df)}")
    print(f"  -> Signal pools: {', '.join(metadata_df['signal_pool'].unique())}")
    print(f"  -> ABC-XYZ classes: {metadata_df['abc_xyz'].value_counts().to_dict()}")

    # Step 2: Run forecasting pipeline
    print("\n[2/3] Running forecasting pipeline...")
    print("  -> Computing baseline forecasts (3-month SMA)...")
    print("  -> Training ML models (Gradient Boosting per pool)...")
    results_df, metrics_df, future_df = run_forecasting_pipeline(sales_df, signals_df)
    print(f"  -> Avg Baseline FA: {metrics_df['baseline_fa'].mean():.1f}%")
    print(f"  -> Avg ML FA: {metrics_df['ml_fa'].mean():.1f}%")
    print(f"  -> Avg Improvement: {(metrics_df['ml_fa'].mean() - metrics_df['baseline_fa'].mean()):.1f}%")
    at_risk = metrics_df[metrics_df['ml_fa'] < 70]
    print(f"  -> At-risk SKUs (FA < 70%): {len(at_risk)}")

    # Step 3: Generate dashboard
    print("\n[3/3] Generating HTML dashboard...")
    html = generate_html_dashboard(metrics_df, results_df, future_df, signals_df)
    output_path = "dashboard.html"
    with open(output_path, "w") as f:
        f.write(html)
    print(f"  -> Dashboard saved to: {output_path}")
    print(f"  -> File size: {len(html):,} bytes")

    print("\n" + "=" * 60)
    print("  Done! Open dashboard.html in your browser.")
    print("=" * 60)


if __name__ == "__main__":
    main()
