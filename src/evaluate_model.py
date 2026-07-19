"""
evaluate_model.py

Evaluasi akurasi model Prophet untuk peramalan inflasi bulanan (m-to-m)
nasional (kolom 'INDONESIA') menggunakan time-series cross-validation.

Alur:
1. Load data/cleaned_inflasi.csv, ambil kolom target 'INDONESIA'.
2. Format ke DataFrame Prophet (ds, y).
3. Latih ulang model Prophet.
4. Cross-validation (initial=3650d, period=180d, horizon=365d).
5. Hitung rata-rata RMSE & MAE.
"""

from pathlib import Path

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# --- Path setup: root project = parent dari folder /src ---
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "cleaned_inflasi.csv"

TARGET = "INDONESIA"


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

    # --- Latih ulang model (konsisten dengan model_forecast.py) ---
    model = Prophet(yearly_seasonality=True)
    model.fit(prophet_df)

    # --- Time-series cross-validation ---
    # initial : 3650 hari (~10 thn) dipakai sebagai training awal mutlak
    # period  : cutoff bergeser tiap 180 hari (~6 bulan)
    # horizon : akurasi dievaluasi untuk prediksi 365 hari (~12 bulan) ke depan
    print("\nMenjalankan cross-validation (ini butuh beberapa saat)...")
    cv_results = cross_validation(
        model,
        initial="3650 days",
        period="180 days",
        horizon="365 days",
    )

    # --- Metrik performa ---
    metrics = performance_metrics(cv_results)
    mean_rmse = metrics["rmse"].mean()
    mean_mae = metrics["mae"].mean()

    print("\n=== Metrik akurasi (rata-rata seluruh horizon) ===")
    print(f"RMSE : {mean_rmse:.4f} persen")
    print(f"MAE  : {mean_mae:.4f} persen")


if __name__ == "__main__":
    main()
