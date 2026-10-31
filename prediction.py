
# =============================================================================
# PREDICTION ENGINE - APOTEK STOCK FORECASTING
# Berdasarkan kode asli Google Colab
# =============================================================================

import io
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Tampilan tabel
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


# =============================================================================
# LANGKAH 1 — MEMBACA FILE CSV
# =============================================================================

def baca_csv(sumber):
    """
    Membaca CSV dengan mencoba beberapa kombinasi separator dan encoding.
    Mendukung file yang diupload melalui Streamlit.
    """

    kombinasi = [
        {"sep": None, "engine": "python", "encoding": "utf-8"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8"},
        {"sep": "\t", "encoding": "utf-8"},
        {"sep": None, "engine": "python", "encoding": "latin-1"},
    ]

    error_terakhir = None

    for opsi in kombinasi:
        try:
            if isinstance(sumber, pd.DataFrame):
                return sumber.copy()

            if isinstance(sumber, (bytes, bytearray)):
                buf = io.BytesIO(sumber)

            elif hasattr(sumber, "getvalue"):
                buf = io.BytesIO(sumber.getvalue())

            else:
                sumber.seek(0)
                buf = sumber

            if hasattr(buf, "seek"):
                buf.seek(0)

            df = pd.read_csv(buf, **opsi)

            if df.shape[1] < 2:
                continue

            return df

        except Exception as err:
            error_terakhir = err

    raise ValueError(
        f"Gagal membaca file CSV. Error terakhir: {error_terakhir}"
    )


# =============================================================================
# FUNGSI UTAMA — RUN FORECAST
# =============================================================================

def run_forecast(sumber):

    # =========================================================================
    # LANGKAH 0 — PERSIAPAN LIBRARY
    # =========================================================================

    NAMA_FILE = getattr(sumber, "name", "sales.csv")

    # =========================================================================
    # LANGKAH 1 — MEMBACA FILE CSV
    # =========================================================================

    df_asli = baca_csv(sumber)

    df_asli.columns = [
        str(k).strip() for k in df_asli.columns
    ]

    # =========================================================================
    # LANGKAH 2 — INSPEKSI DATA MENTAH
    # =========================================================================

    def _adalah_kolom_bulan(nama):
        """Cek apakah nama kolom berformat YYYY-MM."""

        return pd.notna(
            pd.to_datetime(
                str(nama),
                format="%Y-%m",
                errors="coerce"
            )
        )

    kolom_kandidat_bulan = [
        k for k in df_asli.columns
        if _adalah_kolom_bulan(k)
    ]

    # =========================================================================
    # LANGKAH 3 — KONFIGURASI
    # =========================================================================

    ITEM_COL = None
    KOLOM_BULAN = None

    N_LAG = 4
    N_BULAN_FORECAST = 12
    MIN_BULAN_TERISI = 4
    RANDOM_STATE = 42

    # Parameter XGBoost asli
    XGB_PARAMS = dict(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )

    # =========================================================================
    # LANGKAH 4 — DETEKSI KOLOM BULAN & UBAH WIDE → LONG
    # =========================================================================

    kolom_bulan = (
        KOLOM_BULAN
        or [
            k for k in df_asli.columns
            if _adalah_kolom_bulan(k)
        ]
    )

    if not kolom_bulan:
        raise ValueError(
            "Kolom bulan tidak terdeteksi. "
            "Nama kolom bulan harus berformat YYYY-MM."
        )

    kolom_non_bulan = [
        k for k in df_asli.columns
        if k not in kolom_bulan
    ]

    item_col = ITEM_COL or (
        kolom_non_bulan[0]
        if kolom_non_bulan else None
    )

    if item_col is None or item_col not in df_asli.columns:
        raise KeyError(
            f"Kolom kode obat tidak ditemukan. "
            f"Kolom tersedia: {list(df_asli.columns)}"
        )

    df_pakai = df_asli[
        [item_col] + kolom_bulan
    ].copy()

    df_pakai[item_col] = (
        df_pakai[item_col]
        .astype(str)
        .str.strip()
    )

    df_long = df_pakai.melt(
        id_vars=item_col,
        var_name="bulan",
        value_name="sales"
    )

    df_long["bulan"] = pd.to_datetime(
        df_long["bulan"],
        format="%Y-%m"
    )

    df_long = df_long.rename(
        columns={item_col: "kode_obat"}
    )

    df_long = (
        df_long
        .sort_values(["kode_obat", "bulan"])
        .reset_index(drop=True)
    )

    # =========================================================================
    # LANGKAH 5 — PENANGANAN DATA KOSONG
    # =========================================================================

    jumlah_kosong = df_long["sales"].isna().sum()

    persen_kosong = (
        jumlah_kosong / len(df_long) * 100
    )

    df_long["tercatat"] = df_long["sales"].notna()

    df_long["sales"] = (
        pd.to_numeric(
            df_long["sales"],
            errors="coerce"
        ).fillna(0)
    )

    # =========================================================================
    # LANGKAH 6 — DETEKSI & PENANGANAN OUTLIER (CAPPING)
    # =========================================================================

    data_tercatat = df_long[df_long["tercatat"]]

    q1 = (
        data_tercatat
        .groupby("kode_obat")["sales"]
        .quantile(0.25)
    )

    q3 = (
        data_tercatat
        .groupby("kode_obat")["sales"]
        .quantile(0.75)
    )

    iqr = q3 - q1

    batas_bawah = (
        q1 - 1.5 * iqr
    ).clip(lower=0)

    batas_atas = q3 + 1.5 * iqr

    batas_bawah_baris = (
        df_long["kode_obat"].map(batas_bawah)
    )

    batas_atas_baris = (
        df_long["kode_obat"].map(batas_atas)
    )

    sales_asli = df_long["sales"].copy()

    perlu_dipangkas = df_long["tercatat"]

    df_long.loc[perlu_dipangkas, "sales"] = (
        df_long.loc[perlu_dipangkas, "sales"].clip(
            lower=batas_bawah_baris[perlu_dipangkas],
            upper=batas_atas_baris[perlu_dipangkas]
        )
    )

    jumlah_dipangkas = int(
        (sales_asli != df_long["sales"]).sum()
    )

    # =========================================================================
    # LANGKAH 7 — MEMBUAT FITUR LAG
    # =========================================================================

    kolom_lag = [
        f"lag{i}"
        for i in range(1, N_LAG + 1)
    ]

    for i, nama in enumerate(kolom_lag, start=1):

        df_long[nama] = (
            df_long
            .groupby("kode_obat")["sales"]
            .shift(i)
        )

    df_fitur = (
        df_long
        .dropna(subset=kolom_lag)
        .reset_index(drop=True)
    )

    # =========================================================================
    # LANGKAH 8 — CEK KECUKUPAN DATA PER OBAT
    # =========================================================================

    jumlah_bulan_terisi = (
        df_long
        .groupby("kode_obat")["tercatat"]
        .sum()
        .astype(int)
    )

    jumlah_baris_latih = (
        df_fitur
        .groupby("kode_obat")
        .size()
    )

    ringkasan_obat = pd.DataFrame({
        "jumlah_bulan_terisi": jumlah_bulan_terisi,
        "jumlah_baris_latih": jumlah_baris_latih,
    }).fillna(0)

    ringkasan_obat["jumlah_baris_latih"] = (
        ringkasan_obat["jumlah_baris_latih"]
        .astype(int)
    )

    ringkasan_obat["cukup_data"] = (
        (
            ringkasan_obat["jumlah_bulan_terisi"]
            >= MIN_BULAN_TERISI
        )
        &
        (
            ringkasan_obat["jumlah_baris_latih"] >= 2
        )
    )

    n_cukup = int(
        ringkasan_obat["cukup_data"].sum()
    )

    n_minim = (
        len(ringkasan_obat) - n_cukup
    )

    # =========================================================================
    # LANGKAH 9 — MELATIH MODEL PER OBAT
    # =========================================================================

    grup_fitur = dict(
        tuple(df_fitur.groupby("kode_obat"))
    )

    rata_rata_terisi = (
        df_long[df_long["tercatat"]]
        .groupby("kode_obat")["sales"]
        .mean()
    )

    models = {}
    metode_forecast = {}
    lag_terakhir = {}
    nilai_fallback = {}

    total_obat = len(ringkasan_obat)

    for i, (kode, baris) in enumerate(
        ringkasan_obat.iterrows(),
        start=1
    ):

        if baris["cukup_data"]:

            grup = (
                grup_fitur[kode]
                .sort_values("bulan")
            )

            model = XGBRegressor(**XGB_PARAMS)

            model.fit(
                grup[kolom_lag],
                grup["sales"]
            )

            models[kode] = model

            metode_forecast[kode] = "XGBoost"

            lag_terakhir[kode] = (
                grup.tail(1)[kolom_lag]
                .iloc[0]
                .to_dict()
            )

        else:

            nilai_fallback[kode] = float(
                rata_rata_terisi.get(kode, 0.0)
            )

            metode_forecast[kode] = (
                "Rata-rata (Data Minim)"
            )

    # =========================================================================
    # LANGKAH 10 — FORECAST 12 BULAN KE DEPAN
    # =========================================================================

    future_predictions = {}

    for kode, model in models.items():

        lag_sekarang = [
            lag_terakhir[kode][f"lag{i}"]
            for i in range(1, N_LAG + 1)
        ]

        preds = []

        for _ in range(N_BULAN_FORECAST):

            X_input = pd.DataFrame(
                [lag_sekarang],
                columns=kolom_lag
            )

            pred = max(
                float(model.predict(X_input)[0]),
                0.0
            )

            preds.append(pred)

            lag_sekarang = (
                [pred] + lag_sekarang[:-1]
            )

        future_predictions[kode] = preds

    for kode, nilai in nilai_fallback.items():

        future_predictions[kode] = (
            [nilai] * N_BULAN_FORECAST
        )

    # =========================================================================
    # LANGKAH 11 — EVALUASI SEDERHANA (SANITY CHECK)
    # =========================================================================

    N_HOLDOUT = 2

    selisih_semua = []
    jumlah_obat_diuji = 0

    for kode, grup in grup_fitur.items():

        if metode_forecast.get(kode) != "XGBoost":
            continue

        if len(grup) <= N_HOLDOUT:
            continue

        grup = grup.sort_values("bulan")

        latih = grup.iloc[:-N_HOLDOUT]
        uji = grup.iloc[-N_HOLDOUT:]

        model_uji = XGBRegressor(**XGB_PARAMS)

        model_uji.fit(
            latih[kolom_lag],
            latih["sales"]
        )

        prediksi_uji = model_uji.predict(
            uji[kolom_lag]
        )

        selisih_semua.extend(
            np.abs(
                prediksi_uji - uji["sales"].values
            )
        )

        jumlah_obat_diuji += 1

    if selisih_semua:

        mae = float(np.mean(selisih_semua))

    else:

        mae = None

    metrics = {
        "mae": mae,
        "jumlah_obat_diuji": jumlah_obat_diuji,
        "jumlah_bulan_holdout": N_HOLDOUT,
    }

    # =========================================================================
    # LANGKAH 12 — VISUALISASI
    # =========================================================================

    N_CONTOH = 3

    top_obat = (
        df_long
        .groupby("kode_obat")["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(N_CONTOH)
        .index
    )

    bulan_terakhir = df_long["bulan"].max()

    bulan_forecast = pd.date_range(
        bulan_terakhir + pd.DateOffset(months=1),
        periods=N_BULAN_FORECAST,
        freq="MS"
    )

    # Grafik disiapkan di app.py agar dapat ditampilkan secara interaktif.
    # Data historis dan forecast dikembalikan melalui result.

    # =========================================================================
    # LANGKAH 13 — FORMAT HASIL & EKSPOR
    # =========================================================================

    label_bulan = [
        b.strftime("%Y-%m")
        for b in bulan_forecast
    ]

    forecast_df = pd.DataFrame.from_dict(
        future_predictions,
        orient="index",
        columns=label_bulan
    )

    forecast_df.index.name = "kode_obat"

    forecast_df = forecast_df.round(1)

    forecast_df["metode_forecast"] = pd.Series(
        metode_forecast
    )

    forecast_df["jumlah_bulan_terisi"] = (
        ringkasan_obat["jumlah_bulan_terisi"]
    )

    forecast_df["total_prediksi_12bulan"] = (
        forecast_df[label_bulan]
        .sum(axis=1)
        .round(1)
    )

    forecast_df = (
        forecast_df
        .sort_values(
            "total_prediksi_12bulan",
            ascending=False
        )
        .reset_index()
    )

    # =========================================================================
    # RETURN HASIL KE APP.PY
    # =========================================================================

    summary = {
        "total_obat": len(ringkasan_obat),
        "jumlah_model_xgboost": len(models),
        "jumlah_fallback": len(nilai_fallback),
        "jumlah_bulan_data": len(kolom_bulan),
        "jumlah_data_kosong": int(jumlah_kosong),
        "jumlah_outlier_dipangkas": jumlah_dipangkas,
        "bulan_terakhir": bulan_terakhir.strftime("%Y-%m"),
        "bulan_forecast_awal": bulan_forecast[0].strftime("%Y-%m"),
        "bulan_forecast_akhir": bulan_forecast[-1].strftime("%Y-%m"),
        "n_lag": N_LAG,
        "n_bulan_forecast": N_BULAN_FORECAST,
    }

    return {
        "forecast_df": forecast_df,
        "df_long": df_long,
        "df_fitur": df_fitur,
        "ringkasan_obat": ringkasan_obat,
        "future_predictions": future_predictions,
        "metode_forecast": metode_forecast,
        "metrics": metrics,
        "summary": summary,
        "bulan_forecast": bulan_forecast,
        "kolom_bulan": kolom_bulan,
        "item_col": item_col,
    }
