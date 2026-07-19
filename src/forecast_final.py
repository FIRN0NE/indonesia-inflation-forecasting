"""
forecast_final.py

Forecast final 12 bulan ke depan untuk inflasi INDONESIA menggunakan target
Y-o-Y (Year-on-Year) yang error relatifnya jauh lebih rendah dibanding m-to-m
(lihat optimize_model_v2.py).

Alur:
  1. Load cleaned_inflasi.csv (kolom target INDONESIA, inflasi m-to-m persen).
  2. Rekonstruksi indeks harga (cumprod) -> hitung inflasi Y-o-Y.
     Y-o-Y_t = (IHK_t / IHK_{t-12} - 1) * 100,  IHK_t = cumprod(1 + mtm_t/100)
  3. Latih Prophet (default) pada target Y-o-Y.
  4. Prediksi 12 bulan ke depan (freq='MS').
  5. Plot historis aktual + garis forecast + interval (upper/lower).
  6. Simpan ke forecast_yoy_plot.png di root, cetak estimasi bulan ke-12.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend non-interaktif, aman untuk simpan file tanpa display
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet

warnings.simplefilter("ignore", category=FutureWarning)

TARGET = "INDONESIA"
FORECAST_MONTHS = 12
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_inflasi.csv"
PLOT_PATH = BASE_DIR / "forecast_yoy_plot.png"


def load_series():
    """Baca data bersih, kembalikan Series inflasi m-to-m (persen) ber-index datetime."""
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"], index_col="datetime")
    if TARGET not in df.columns:
        raise KeyError(f"Kolom '{TARGET}' tidak ditemukan di {DATA_PATH}")
    return df[TARGET].sort_index()


def compute_yoy(mtm_series):
    """
    Konversi inflasi m-to-m (persen) menjadi inflasi Y-o-Y (persen).

    Data adalah laju bulanan (bukan IHK), jadi indeks harga direkonstruksi
    dengan cumulative product lalu diambil perubahan 12-bulan.
    """
    price_index = (1 + mtm_series / 100.0).cumprod()
    yoy = (price_index / price_index.shift(12) - 1) * 100.0
    return yoy.dropna()  # 12 bulan pertama hilang (tak ada pembanding tahun lalu)


def main():
    series = load_series()
    yoy = compute_yoy(series)
    df = pd.DataFrame({"ds": yoy.index, "y": yoy.values})

    print(f"Kolom target : {TARGET} (dikonversi ke Y-o-Y)")
    print(f"Data latih   : {len(df)} baris "
          f"({df['ds'].min():%Y-%m} s/d {df['ds'].max():%Y-%m})")

    # --- Latih Prophet (default) ---
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    model.fit(df)

    # --- Forecast 12 bulan ke depan ---
    future = model.make_future_dataframe(periods=FORECAST_MONTHS, freq="MS")
    forecast = model.predict(future)

    # --- Visualisasi: historis aktual + forecast + interval ---
    fig, ax = plt.subplots(figsize=(13, 6))

    # Data historis aktual
    ax.plot(df["ds"], df["y"], "k.", markersize=5, label="Aktual (Y-o-Y historis)")

    # Garis prediksi
    ax.plot(forecast["ds"], forecast["yhat"], color="#2b6cb0",
            linewidth=2, label="Prediksi (yhat)")

    # Interval ketidakpastian (upper/lower)
    ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                    color="#2b6cb0", alpha=0.2, label="Interval prediksi")

    # Garis pemisah awal periode forecast
    last_actual = df["ds"].max()
    ax.axvline(last_actual, color="red", linestyle="--", linewidth=1,
               label="Awal forecast")

    ax.set_title("Forecast Inflasi Indonesia (Y-o-Y) — 12 Bulan ke Depan")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Inflasi Y-o-Y (%)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    print(f"Plot tersimpan: {PLOT_PATH}")

    # --- Estimasi bulan ke-12 ---
    last = forecast.iloc[-1]
    print(f"\n=== Estimasi inflasi Y-o-Y bulan ke-{FORECAST_MONTHS} ke depan ===")
    print(f"Periode : {last['ds']:%Y-%m-%d}")
    print(f"yhat    : {last['yhat']:.4f} %")
    print(f"rentang : {last['yhat_lower']:.4f} % s/d {last['yhat_upper']:.4f} %")


if __name__ == "__main__":
    main()
