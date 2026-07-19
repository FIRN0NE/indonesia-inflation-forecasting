"""
model_forecast.py

Forecast inflasi bulanan nasional Indonesia menggunakan Prophet.
Baca data bersih hasil data_prep.py, latih model, prediksi 12 bulan ke depan,
lalu simpan visualisasi historis + interval prediksi ke forecast_plot.png.
"""

import os

import matplotlib

matplotlib.use("Agg")  # backend non-interaktif, aman tanpa display
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet

# --- Path ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_inflasi.csv")
PLOT_PATH = os.path.join(BASE_DIR, "forecast_plot.png")

# Kandidat kolom target agregat nasional, diurutkan berdasar prioritas.
TARGET_CANDIDATES = ["NASIONAL", "INDONESIA", "DKI JAKARTA"]


def pilih_kolom_target(df):
    """Cari kolom agregat nasional. Fallback: DKI Jakarta, lalu kolom pertama."""
    kolom_upper = {c.upper(): c for c in df.columns}
    for kandidat in TARGET_CANDIDATES:
        if kandidat in kolom_upper:
            return kolom_upper[kandidat]
    return df.columns[0]


def main():
    # 1. Baca data bersih, parse index datetime.
    df = pd.read_csv(DATA_PATH, index_col="datetime", parse_dates=True)

    target = pilih_kolom_target(df)
    print(f"Kolom target : {target}")

    # 2. Siapkan DataFrame Prophet: reset index -> kolom 'ds' & 'y'.
    dfp = df[[target]].reset_index()
    dfp = dfp.rename(columns={"datetime": "ds", target: "y"})
    dfp = dfp.dropna(subset=["y"])
    print(f"Data latih   : {len(dfp)} baris ({dfp['ds'].min().date()} s/d {dfp['ds'].max().date()})")

    # 3. Inisialisasi & latih model. Data bulanan -> aktifkan seasonality tahunan.
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(dfp)

    # 4. Dataframe masa depan: 12 bulan ke depan, awal bulan (Month Start).
    future = model.make_future_dataframe(periods=12, freq="MS")

    # 5. Prediksi.
    forecast = model.predict(future)

    # --- Visualisasi matplotlib (plotly tidak tersedia) ---
    fig, ax = plt.subplots(figsize=(13, 6))

    # Data historis aktual.
    ax.plot(dfp["ds"], dfp["y"], "k.", markersize=4, label="Aktual (historis)")

    # Garis prediksi (tren utama).
    ax.plot(forecast["ds"], forecast["yhat"], color="#2563eb", linewidth=1.8, label="Prediksi (yhat)")

    # Batas atas/bawah interval prediksi.
    ax.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        color="#2563eb",
        alpha=0.2,
        label="Interval prediksi (lower/upper)",
    )

    # Tandai awal periode forecast.
    batas = dfp["ds"].max()
    ax.axvline(batas, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Awal forecast")

    ax.set_title(f"Prediksi Inflasi Bulanan {target} - 12 Bulan ke Depan (Prophet)")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Inflasi Bulanan (M-to-M, %)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # 6. Simpan plot ke root directory.
    fig.savefig(PLOT_PATH, dpi=120)
    print(f"Plot tersimpan: {PLOT_PATH}")

    # 7. Estimasi bulan ke-12 dari hasil prediksi.
    prediksi_masa_depan = forecast.tail(12)
    baris_ke12 = prediksi_masa_depan.iloc[-1]
    print("\n=== Estimasi inflasi bulan ke-12 ke depan ===")
    print(f"Periode : {baris_ke12['ds'].date()}")
    print(f"yhat    : {baris_ke12['yhat']:.4f} %")
    print(f"rentang : {baris_ke12['yhat_lower']:.4f} % s/d {baris_ke12['yhat_upper']:.4f} %")


if __name__ == "__main__":
    main()
