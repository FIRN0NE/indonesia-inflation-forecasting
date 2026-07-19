"""Gabungkan & bersihkan data Inflasi Bulanan (M-to-M) BPS 2010-2025.

Output: data/cleaned_inflasi.csv
- Index  : datetime (YYYY-MM-01), terurut lama -> baru
- Kolom  : nama kota/wilayah (termasuk INDONESIA), nilai inflasi bulanan (persen)
- Missing values diisi dengan interpolasi time-series.
"""

import re
import glob
from pathlib import Path

import pandas as pd

# Lokasi folder relatif terhadap root project (parent dari /src)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT = DATA_DIR / "cleaned_inflasi.csv"

# Mapping nama bulan Bahasa Indonesia -> angka
BULAN_ID = {
    "Januari": 1,
    "Februari": 2,
    "Maret": 3,
    "April": 4,
    "Mei": 5,
    "Juni": 6,
    "Juli": 7,
    "Agustus": 8,
    "September": 9,
    "Oktober": 10,
    "November": 11,
    "Desember": 12,
}


def baca_satu_file(path: Path) -> pd.DataFrame:
    """Baca satu CSV BPS -> long format: kolom [datetime, kota, inflasi]."""
    # Tahun diambil dari nama file (mis. "... 2010.csv")
    match = re.search(r"(\d{4})", path.name)
    if not match:
        raise ValueError(f"Tidak menemukan tahun pada nama file: {path.name}")
    tahun = int(match.group(1))

    # 3 baris pertama adalah metadata; baris ke-4 (skiprows=3) jadi header bulan.
    df = pd.read_csv(path, skiprows=3)
    df = df.rename(columns={df.columns[0]: "kota"})

    # Buang kolom rekap tahunan, sisakan hanya bulan.
    df = df.drop(columns=["Tahunan"], errors="ignore")

    # Wide -> long
    df_long = df.melt(id_vars="kota", var_name="bulan", value_name="inflasi")

    # "-" (dan sejenis) -> NaN, lalu paksa numerik
    df_long["inflasi"] = pd.to_numeric(
        df_long["inflasi"].replace("-", pd.NA), errors="coerce"
    )

    # Susun datetime YYYY-MM-01
    df_long["bulan_num"] = df_long["bulan"].map(BULAN_ID)
    df_long = df_long.dropna(subset=["bulan_num"])  # jaga2 kolom non-bulan
    df_long["datetime"] = pd.to_datetime(
        dict(
            year=tahun,
            month=df_long["bulan_num"].astype(int),
            day=1,
        )
    )

    return df_long[["datetime", "kota", "inflasi"]]


def main() -> None:
    files = sorted(glob.glob(str(DATA_DIR / "Inflasi Bulanan*.csv")))
    if not files:
        raise FileNotFoundError(f"Tidak ada CSV inflasi di {DATA_DIR}")

    print(f"Menemukan {len(files)} file CSV.")
    frames = [baca_satu_file(Path(f)) for f in files]
    long_df = pd.concat(frames, ignore_index=True)

    # Long -> wide: index datetime, kolom = kota
    wide = long_df.pivot_table(
        index="datetime", columns="kota", values="inflasi", aggfunc="first"
    )
    wide = wide.sort_index()  # terlama -> terbaru
    wide.columns.name = None

    # Cek missing values
    total_missing = int(wide.isna().sum().sum())
    print(f"Rentang data : {wide.index.min():%Y-%m} s/d {wide.index.max():%Y-%m}")
    print(f"Dimensi      : {wide.shape[0]} bulan x {wide.shape[1]} wilayah")
    print(f"Missing values: {total_missing}")

    if total_missing > 0:
        # Interpolasi berbasis waktu (hanya mengisi di antara titik valid),
        # lalu tutup sisa NaN di ujung dengan ffill/bfill.
        wide = wide.interpolate(method="time", limit_direction="both")
        wide = wide.ffill().bfill()
        sisa = int(wide.isna().sum().sum())
        print(f"Setelah interpolasi, sisa missing: {sisa}")

    wide.to_csv(OUTPUT, index_label="datetime")
    print(f"Tersimpan: {OUTPUT}")


if __name__ == "__main__":
    main()
