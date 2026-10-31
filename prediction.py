# =============================================================================
# PREDICTION ENGINE - APOTEK STOCK FORECASTING
# Model: XGBoost Regressor + Mean Fallback
# Forecast: 12 Months
# =============================================================================
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
