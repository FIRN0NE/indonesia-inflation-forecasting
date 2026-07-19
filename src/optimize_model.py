"""
optimize_model.py

Optimasi model Prophet untuk peramalan inflasi bulanan (m-to-m) nasional
(kolom 'INDONESIA') dengan menambahkan efek hari libur nasional Indonesia
dan sedikit tuning hyperparameter, lalu evaluasi ulang via cross-validation.

Baseline sebelumnya (evaluate_model.py):
    RMSE = 0.4248 %
    MAE  = 0.3197 %
"""

from pathlib import Path

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# --- Path setup: root project = parent dari folder /src ---
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "cleaned_inflasi.csv"

TARGET = "INDONESIA"

# Baseline dari evaluate_model.py untuk perbandingan eksplisit.
BASELINE_RMSE = 0.4248
BASELINE_MAE = 0.3197


def load_prophet_df() -> pd.DataFrame:
    """Load data bersih dan bentuk ke format Prophet (ds, y)."""
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])

    if TARGET not in df.columns:
        raise KeyError(
            f"Kolom target '{TARGET}' tidak ada di {DATA_PATH.name}. "
            f"Kolom tersedia (contoh): {list(df.columns[:5])}..."
        )

    prophet_df = (
        df[["datetime", TARGET]]
        .rename(columns={"datetime": "ds", TARGET: "y"})
        .sort_values("ds")
        .reset_index(drop=True)
    )
    return prophet_df


def main() -> None:
    prophet_df = load_prophet_df()
    print(f"Kolom target : {TARGET}")
    print(
        f"Data latih   : {len(prophet_df)} baris "
        f"({prophet_df['ds'].min():%Y-%m-%d} s/d {prophet_df['ds'].max():%Y-%m-%d})"
    )

    # --- Inisialisasi model dengan tuning + efek libur nasional ---
    model = Prophet(
        yearly_seasonality=True,
        changepoint_prior_scale=0.1,   # tren lebih fleksibel thd guncangan jangka pendek
        seasonality_prior_scale=10.0,  # beri ruang lebih pada komponen musiman
    )
    # Efek libur nasional Indonesia (mis. Lebaran) ditambahkan sebelum fit.
    model.add_country_holidays(country_name="ID")

    model.fit(prophet_df)

    # --- Cross-validation, parameter identik dengan baseline ---
    print("\nMenjalankan cross-validation (ini butuh beberapa saat)...")
    cv_results = cross_validation(
        model,
        initial="3650 days",
        period="180 days",
        horizon="365 days",
    )

    metrics = performance_metrics(cv_results)
    mean_rmse = metrics["rmse"].mean()
    mean_mae = metrics["mae"].mean()

    # --- Perbandingan eksplisit dengan baseline ---
    d_rmse = mean_rmse - BASELINE_RMSE
    d_mae = mean_mae - BASELINE_MAE
    pct_rmse = d_rmse / BASELINE_RMSE * 100
    pct_mae = d_mae / BASELINE_MAE * 100

    print("\n=== Perbandingan metrik ===")
    print(f"{'Metrik':<8}{'Baseline':>12}{'Optimized':>12}{'Selisih':>12}{'Perubahan':>12}")
    print(f"{'RMSE':<8}{BASELINE_RMSE:>11.4f}%{mean_rmse:>11.4f}%{d_rmse:>+11.4f}%{pct_rmse:>+11.2f}%")
    print(f"{'MAE':<8}{BASELINE_MAE:>11.4f}%{mean_mae:>11.4f}%{d_mae:>+11.4f}%{pct_mae:>+11.2f}%")

    print("\n=== Kesimpulan ===")
    if mean_mae < BASELINE_MAE:
        print(f"MAE turun {abs(pct_mae):.2f}% -> optimasi BERHASIL menurunkan error.")
    else:
        print(f"MAE naik {abs(pct_mae):.2f}% -> optimasi TIDAK menurunkan error.")


if __name__ == "__main__":
    main()
