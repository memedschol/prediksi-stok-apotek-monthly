
# =============================================================================
# APOTEK STOCK FORECASTING
# Streamlit Web Application
# =============================================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from prediction import run_forecast


# =============================================================================
# 1. PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Apotek Stock Forecasting",
    page_icon="💊",
    layout="wide"
)


# =============================================================================
# 2. CUSTOM STYLE
# =============================================================================

st.markdown("""
<style>
    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #176B50;
        margin-bottom: 0px;
    }

    .sub-title {
        color: #777777;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .stButton > button {
        background-color: #176B50;
        color: white;
        border-radius: 8px;
        height: 45px;
        font-weight: 600;
        width: 100%;
    }

    .stDownloadButton > button {
        width: 100%;
        border-radius: 8px;
        height: 45px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 3. HEADER
# =============================================================================

st.markdown(
    '<div class="main-title">💊 Apotek Stock Forecasting</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Monthly Sales Forecasting System | XGBoost Model'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =============================================================================
# 4. FILE UPLOAD
# =============================================================================

st.subheader("Upload Data Penjualan")

st.write(
    "Upload file CSV berisi data historis penjualan obat "
    "dengan format kolom bulan YYYY-MM."
)

uploaded_file = st.file_uploader(
    "Pilih file CSV",
    type=["csv"],
    help="Upload data penjualan obat dalam format CSV."
)


# =============================================================================
# 5. PROCESS FORECAST
# =============================================================================

if uploaded_file is not None:

    st.success(
        f"File berhasil dipilih: {uploaded_file.name}"
    )

    # Preview data
    with st.expander("Lihat preview data CSV"):
        try:
            preview = pd.read_csv(
                uploaded_file,
                sep=None,
                engine="python"
            )

            st.dataframe(
                preview.head(10),
                use_container_width=True
            )

            st.caption(
                f"Jumlah baris: {len(preview):,} | "
                f"Jumlah kolom: {len(preview.columns):,}"
            )

        except Exception as e:
            st.warning(
                f"Preview tidak dapat ditampilkan: {e}"
            )

    if st.button("🚀 PROSES PREDIKSI STOK"):

        try:

            with st.spinner(
                "Sedang memproses data dan melatih model XGBoost..."
            ):

                uploaded_file.seek(0)

                result = run_forecast(uploaded_file)

                st.session_state["forecast_result"] = result
                st.session_state["uploaded_filename"] = (
                    uploaded_file.name
                )

            st.success(
                "Prediksi stok berhasil diselesaikan!"
            )

        except Exception as e:

            st.error(
                f"Terjadi kesalahan saat memproses data: {e}"
            )

            st.session_state.pop(
                "forecast_result", None
            )


# =============================================================================
# 6. DISPLAY FORECAST RESULTS
# =============================================================================

if "forecast_result" in st.session_state:

    result = st.session_state["forecast_result"]

    forecast_df = result["forecast_df"]
    df_long = result["df_long"]
    metrics = result["metrics"]
    summary = result["summary"]

    st.divider()

    st.header("📊 Hasil Forecasting")

    # -------------------------------------------------------------------------
    # SUMMARY METRICS
    # -------------------------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Obat",
        f"{summary['total_obat']:,}"
    )

    col2.metric(
        "Model XGBoost",
        f"{summary['jumlah_model_xgboost']:,}"
    )

    col3.metric(
        "Model Fallback",
        f"{summary['jumlah_fallback']:,}"
    )

    col4.metric(
        "Forecast Period",
        f"{summary['n_bulan_forecast']} Bulan"
    )

    st.caption(
        f"Periode data terakhir: {summary['bulan_terakhir']} | "
        f"Periode forecast: {summary['bulan_forecast_awal']} "
        f"sampai {summary['bulan_forecast_akhir']}"
    )

    # -------------------------------------------------------------------------
    # MODEL EVALUATION
    # -------------------------------------------------------------------------

    st.subheader("Evaluasi Model")

    col1, col2 = st.columns(2)

    with col1:

        if metrics["mae"] is not None:

            st.metric(
                "Mean Absolute Error (MAE)",
                f"{metrics['mae']:,.2f} unit"
            )

        else:

            st.info(
                "Data belum mencukupi untuk evaluasi MAE."
            )

    with col2:

        st.metric(
            "Jumlah Obat yang Diuji",
            f"{metrics['jumlah_obat_diuji']:,}"
        )

    # -------------------------------------------------------------------------
    # FORECAST TABLE
    # -------------------------------------------------------------------------

    st.subheader("Tabel Hasil Prediksi")

    st.write(
        "Berikut hasil prediksi kebutuhan penjualan "
        "untuk masing-masing obat selama 12 bulan ke depan."
    )

    st.dataframe(
        forecast_df,
        use_container_width=True,
        hide_index=True
    )

    # -------------------------------------------------------------------------
    # FORECAST VISUALIZATION
    # -------------------------------------------------------------------------

    st.divider()

    st.subheader("Grafik Historis dan Forecast")

    daftar_obat = sorted(
        df_long["kode_obat"].unique().tolist()
    )

    selected_obat = st.selectbox(
        "Pilih obat untuk melihat grafik:",
        daftar_obat
    )

    if selected_obat:

        riwayat = (
            df_long[
                df_long["kode_obat"] == selected_obat
            ]
            .sort_values("bulan")
        )

        baris_forecast = forecast_df[
            forecast_df["kode_obat"] == selected_obat
        ]

        if not baris_forecast.empty:

            kolom_forecast = [
                b.strftime("%Y-%m")
                for b in result["bulan_forecast"]
            ]

            nilai_forecast = (
                baris_forecast[kolom_forecast]
                .iloc[0]
                .astype(float)
                .values
            )

            fig, ax = plt.subplots(figsize=(12, 5))

            ax.plot(
                riwayat["bulan"],
                riwayat["sales"],
                marker="o",
                label="Penjualan Historis"
            )

            ax.plot(
                result["bulan_forecast"],
                nilai_forecast,
                marker="o",
                linestyle="--",
                label="Forecast XGBoost / Fallback"
            )

            ax.axvline(
                riwayat["bulan"].max(),
                linestyle=":",
                alpha=0.7
            )

            ax.set_title(
                f"Forecast Penjualan Obat {selected_obat}"
            )

            ax.set_xlabel("Bulan")
            ax.set_ylabel("Jumlah Penjualan")
            ax.legend()
            ax.grid(alpha=0.3)

            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

            metode = baris_forecast[
                "metode_forecast"
            ].iloc[0]

            st.caption(
                f"Metode prediksi untuk obat ini: {metode}"
            )

    # -------------------------------------------------------------------------
    # DOWNLOAD FORECAST CSV
    # -------------------------------------------------------------------------

    st.divider()

    st.subheader("Download Hasil Prediksi")

    csv_data = forecast_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="⬇️ DOWNLOAD HASIL FORECAST CSV",
        data=csv_data,
        file_name="forecast_xgboost_result.csv",
        mime="text/csv"
    )

    st.caption(
        "File hasil berisi kode obat, prediksi bulanan, "
        "metode forecasting, dan total prediksi 12 bulan."
    )


# =============================================================================
# 7. FOOTER
# =============================================================================

st.divider()

st.caption(
    "Apotek Stock Forecasting | Powered by Python, Streamlit & XGBoost"
)
