"""
optimize_model_v2.py

Dua pendekatan untuk menekan MAE forecast inflasi INDONESIA:

  Pendekatan 1 - Grid Search M-to-M
      Iterasi changepoint_prior_scale (0.01, 0.02, 0.05) pada target m-to-m
      TANPA efek holiday. Pilih parameter dengan MAE cross-validation terendah.

  Pendekatan 2 - Pivot ke target Y-o-Y
      Data kita adalah inflasi m-to-m (persen), BUKAN IHK mentah. Maka Y-o-Y
      tidak bisa dihitung dengan pct_change(12) langsung. Cara yang benar:
        1. Rekonstruksi indeks harga  : IHK_t = cumprod(1 + mtm_t/100)
        2. Y-o-Y_t (persen)           : (IHK_t / IHK_{t-12} - 1) * 100
      Lalu latih Prophet parameter default pada target Y-o-Y dan jalankan
      cross-validation yang sama (initial/period/horizon identik).

Membandingkan MAE terbaik m-to-m vs MAE y-o-y di akhir.
"""

import warnings
from pathlib import Path

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

warnings.simplefilter("ignore", category=FutureWarning)

# --- Parameter cross-validation (identik dengan evaluate_model.py) ---
CV_INITIAL = "3650 days"   # 10 tahun pertama sebagai training mutlak
CV_PERIOD = "180 days"     # geser tiap 6 bulan
CV_HORIZON = "365 days"    # evaluasi prediksi 12 bulan ke depan

TARGET = "INDONESIA"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_inflasi.csv"


def load_series():
    """Baca data bersih, kembalikan Series inflasi m-to-m (persen) ber-index datetime."""
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"], index_col="datetime")
    if TARGET not in df.columns:
        raise KeyError(f"Kolom '{TARGET}' tidak ditemukan di {DATA_PATH}")
    return df[TARGET].sort_index()


def to_prophet_frame(series):
    """Ubah Series ber-index datetime menjadi DataFrame ds/y untuk Prophet."""
    return pd.DataFrame({"ds": series.index, "y": series.values}).dropna()


def run_cv(model, df):
    """Fit model lalu jalankan cross-validation, kembalikan (rmse, mae) rata-rata."""
    model.fit(df)
    cv = cross_validation(
        model,
        initial=CV_INITIAL,
        period=CV_PERIOD,
        horizon=CV_HORIZON,
        disable_tqdm=True,
    )
    metrics = performance_metrics(cv)
    return metrics["rmse"].mean(), metrics["mae"].mean()


# ---------------------------------------------------------------------------
# Pendekatan 1: Grid Search M-to-M
# ---------------------------------------------------------------------------
def grid_search_mtm(series):
    df = to_prophet_frame(series)
    grid = [0.01, 0.02, 0.05]
    results = []

    print("=== Pendekatan 1: Grid Search M-to-M (tanpa holiday) ===")
    for cps in grid:
        model = Prophet(
            changepoint_prior_scale=cps,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        rmse, mae = run_cv(model, df)
        results.append({"changepoint_prior_scale": cps, "rmse": rmse, "mae": mae})
        print(f"  cps={cps:<5} -> RMSE={rmse:.4f}%  MAE={mae:.4f}%")

    best = min(results, key=lambda r: r["mae"])
    print(f"  -> Terbaik: cps={best['changepoint_prior_scale']} "
          f"(MAE={best['mae']:.4f}%)\n")
    return best


# ---------------------------------------------------------------------------
# Pendekatan 2: Target Y-o-Y
# ---------------------------------------------------------------------------
def compute_yoy(mtm_series):
    """
    Konversi inflasi m-to-m (persen) menjadi inflasi Y-o-Y (persen).

    Karena data adalah laju bulanan (bukan IHK), indeks harga direkonstruksi
    dengan cumulative product, baru diambil perubahan 12-bulan.
    """
    price_index = (1 + mtm_series / 100.0).cumprod()
    yoy = (price_index / price_index.shift(12) - 1) * 100.0
    return yoy.dropna()  # 12 bulan pertama hilang (tak ada pembanding tahun lalu)


def evaluate_yoy(series):
    yoy = compute_yoy(series)
    df = to_prophet_frame(yoy)

    print("=== Pendekatan 2: Target Y-o-Y (Prophet default) ===")
    print(f"  Rentang y-o-y : {df['ds'].min():%Y-%m} s/d {df['ds'].max():%Y-%m} "
          f"({len(df)} baris)")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    rmse, mae = run_cv(model, df)
    print(f"  -> RMSE={rmse:.4f}%  MAE={mae:.4f}%\n")
    return {"rmse": rmse, "mae": mae}


def main():
    series = load_series()
    print(f"Kolom target : {TARGET}")
    print(f"Data m-to-m  : {len(series)} baris "
          f"({series.index.min():%Y-%m} s/d {series.index.max():%Y-%m})\n")

    best_mtm = grid_search_mtm(series)
    yoy_res = evaluate_yoy(series)

    # --- Perbandingan akhir ---
    print("=== Perbandingan akhir MAE ===")
    print(f"{'Pendekatan':<38}{'RMSE':>10}{'MAE':>10}")
    print(f"{'Baseline m-to-m (cps=0.05, awal)':<38}{'0.4248%':>10}{'0.3197%':>10}")
    print(f"{'M-to-M terbaik (cps=' + str(best_mtm['changepoint_prior_scale']) + ')':<38}"
          f"{best_mtm['rmse']:>9.4f}%{best_mtm['mae']:>9.4f}%")
    print(f"{'Y-o-Y (default)':<38}{yoy_res['rmse']:>9.4f}%{yoy_res['mae']:>9.4f}%")

    return best_mtm, yoy_res


if __name__ == "__main__":
    main()
