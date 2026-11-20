export interface KpisResponse {
  annual_mean_pm25_ugm3: Record<string, number>;
  peak_pm25_ugm3: number;
  peak_date: string;
  best_model_by_val: string;
  best_model_rmse_test: number;
  best_model_skill_test: number;
  persistence_rmse_test: number;
  data_date_range_start: string;
  data_date_range_end: string;
  total_rows: number;
  train_rows: number;
  val_rows: number;
  test_rows: number;
}
export interface TodayResponse {
  date: string;
  pm25_ugm3: number;
  aqi_category: string;
  temperature: number;
  humidity: number;
  wind_speed: number;
  rainfall: number;
}
export interface SeriesPoint { date: string; pm25_ugm3: number; aqi_category: string; }
export interface SeriesResponse { start: string; end: string; count: number; series: SeriesPoint[]; }
export interface HistoryPoint { date: string; pm25_ugm3: number; aqi_category: string; }
export interface ForecastResponse {
  query_date: string;
  forecast_date: string;
  model: string;
  pm25_forecast: number;
  aqi_category: string;
  advisory: string;
  disclaimer: string;
  confidence_low: number;
  confidence_high: number;
  confidence_method: string;
  history_7d: HistoryPoint[];
  weather_inputs: {
    temperature_2m_mean: number;
    relative_humidity_2m_mean: number;
    wind_speed_10m_max: number;
    precipitation_sum: number;
  };
}
export interface EdaResponse {
  monthly_mean_by_month: Record<string, number>;
  peak_month: number;
  peak_month_mean_ugm3: number;
  weekday_means_ugm3: Record<string, number>;
  smog_season_mean_ugm3: number;
  off_season_mean_ugm3: number;
  smog_vs_offseason_ratio: number;
  autocorr_lag1: number;
  autocorr_lag7: number;
  weather_correlations: Record<string, number>;
  strongest_weather_corr_var: string;
  strongest_weather_corr_value: number;
}
export interface ModelMetrics {
  model: string; rmse: number; rmse_std: number;
  mae: number; mae_std: number;
  f1: number; f1_std: number; skill: number | null;
}
export interface ModelsResponse {
  comparison: ModelMetrics[];
  predictions_test: Array<{
    date: string; actual: number; persistence: number;
    moving_avg7: number; ridge: number; rf: number; rf_kalman: number | null;
  }>;
  confusion_matrix: { model: string; labels: string[]; matrix: number[][]; };
  feature_importance: { model: string; importances: Record<string, number>; };
  seed_stability: {
    rmse_mean: number; rmse_std: number;
    mae_mean: number; mae_std: number;
    f1_mean: number; f1_std: number;
  };
}