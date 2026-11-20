import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Thermometer, Droplets, Wind, CloudRain } from 'lucide-react';
import ReactECharts from 'echarts-for-react';
import { useForecast } from '../hooks/useApi';
import { AQIBadge }   from '../components/AQIBadge';
import { Skeleton }   from '../components/Skeleton';
import { ErrorState } from '../components/ErrorState';
import { aqiColor } from '../theme';
import type { HistoryPoint } from '../types/api';

const DATA_START = '2019-05-16';
const DATA_END   = '2025-02-17';

export function Forecast() {
  const [date,  setDate]       = useState('');
  const [model, setModel]      = useState<'ridge' | 'random_forest'>('random_forest');
  const [query, setQuery]      = useState<string | null>(null);
  const [qModel, setQModel]    = useState<string>('random_forest');

  const fc = useForecast(query, qModel);

  function handleFetch() {
    if (!date) return;
    setQuery(date);
    setQModel(model);
  }

  function buildHistOption(hist: HistoryPoint[], forecast_date: string, pm25_forecast: number, aqi_cat: string) {
    const labels = [...hist.map(h => h.date), forecast_date];
    const vals: Array<number | null> = [...hist.map(h => h.pm25_ugm3), null];
    return {
      backgroundColor: 'transparent',
      grid: { top: 10, right: 10, bottom: 36, left: 50 },
      tooltip: { trigger: 'axis', backgroundColor: '#111A2E', borderColor: 'rgba(34,211,238,0.2)', textStyle: { color: '#E2E8F0' } },
      xAxis: { type: 'category', data: labels, axisLabel: { color: '#64748B', fontSize: 9, rotate: 30 } },
      yAxis: { type: 'value', axisLabel: { color: '#64748B', fontSize: 9 }, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } } },
      series: [
        {
          type: 'bar', data: vals,
          itemStyle: {
            color: (p: { dataIndex: number }) =>
              p.dataIndex < hist.length ? aqiColor(hist[p.dataIndex]?.aqi_category ?? '') : '#22D3EE',
            borderRadius: [4, 4, 0, 0],
          },
        },
        {
          type: 'scatter', symbol: 'circle', symbolSize: 14,
          data: [{ value: [forecast_date, pm25_forecast], itemStyle: { color: aqiColor(aqi_cat) } }],
        },
      ],
    };
  }

  return (
    <div className="flex flex-col gap-6 p-6 overflow-y-auto h-full max-w-3xl mx-auto w-full">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h2 className="text-xl font-bold text-white mb-1">PM2.5 Forecast</h2>
        <p className="text-sm text-slate-400">Enter a date to predict next-day air quality</p>
      </motion.div>

      {/* Controls */}
      <div className="glass rounded-2xl p-5 flex flex-col sm:flex-row gap-3 items-end">
        <div className="flex flex-col gap-1 flex-1">
          <label className="text-xs text-slate-400">Date (data: {DATA_START} to {DATA_END})</label>
          <input type="date" value={date} onChange={e => setDate(e.target.value)}
            min={DATA_START} max={DATA_END}
            className="rounded-xl bg-white/5 border border-white/10 text-white px-3 py-2 text-sm focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent/30 transition" />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-slate-400">Model</label>
          <select value={model} onChange={e => setModel(e.target.value as 'ridge' | 'random_forest')}
            className="rounded-xl bg-white/5 border border-white/10 text-white px-3 py-2 text-sm focus:outline-none focus:border-accent transition">
            <option value="random_forest">Random Forest</option>
            <option value="ridge">Ridge Regression</option>
          </select>
        </div>
        <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
          onClick={handleFetch} disabled={!date || fc.isFetching}
          className="flex items-center gap-2 px-5 py-2 rounded-xl font-semibold text-sm transition-all disabled:opacity-50 text-white"
          style={{ background: 'linear-gradient(135deg,#22D3EE,#8B5CF6)' }}>
          {fc.isFetching
            ? <span className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" />
            : <Search className="w-4 h-4" />}
          Get Forecast
        </motion.button>
      </div>

      {/* Empty state */}
      {!query && !fc.data && (
        <div className="glass rounded-2xl p-12 text-center">
          <div className="text-6xl mb-4">🌤</div>
          <p className="text-slate-400">Enter a date above to get tomorrow's PM2.5 forecast</p>
        </div>
      )}

      {fc.isFetching && <Skeleton className="h-96" />}
      {fc.error && !fc.isFetching && <ErrorState message={fc.error.message} />}

      <AnimatePresence>
        {fc.data && !fc.isFetching && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="flex flex-col gap-4">
            {/* Main result */}
            <div className="glass rounded-2xl p-6">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <p className="text-xs text-slate-400 mb-1">Forecast for{' '}
                    <span className="text-accent font-mono">{fc.data.forecast_date}</span>
                  </p>
                  <div className="flex items-baseline gap-2">
                    <span className="text-5xl font-bold font-mono" style={{ color: aqiColor(fc.data.aqi_category) }}>
                      {fc.data.pm25_forecast.toFixed(1)}
                    </span>
                    <span className="text-slate-400 text-sm">µg/m³</span>
                  </div>
                </div>
                <AQIBadge category={fc.data.aqi_category} pulse size="lg" />
              </div>
              <p className="text-sm text-slate-300 mb-4">{fc.data.advisory}</p>
              {/* Confidence range */}
              <div className="mb-4">
                <p className="text-xs text-slate-500 mb-2">90% Confidence Range</p>
                <div className="relative h-6 rounded-full bg-white/5 overflow-hidden">
                  <motion.div className="absolute h-full rounded-full"
                    style={{ background: `linear-gradient(90deg,${aqiColor(fc.data.aqi_category)}44,${aqiColor(fc.data.aqi_category)}88)` }}
                    initial={{ width: 0, left: '0%' }}
                    animate={{
                      left:  `${(fc.data.confidence_low  / 500) * 100}%`,
                      width: `${((fc.data.confidence_high - fc.data.confidence_low) / 500) * 100}%`,
                    }}
                    transition={{ duration: 0.8, ease: 'easeOut' }} />
                  <div className="absolute top-0 bottom-0 w-0.5 bg-white"
                    style={{ left: `${(fc.data.pm25_forecast / 500) * 100}%` }} />
                </div>
                <div className="flex justify-between text-xs text-slate-500 mt-1">
                  <span>{fc.data.confidence_low.toFixed(1)}</span>
                  <span className="text-white font-mono">{fc.data.pm25_forecast.toFixed(1)}</span>
                  <span>{fc.data.confidence_high.toFixed(1)}</span>
                </div>
                <p className="text-xs text-slate-600 mt-1 italic">{fc.data.confidence_method}</p>
              </div>
              <p className="text-xs text-slate-600 italic border-t border-white/5 pt-3">{fc.data.disclaimer}</p>
            </div>

            {/* 7-day history chart */}
            <div className="glass rounded-2xl p-5">
              <p className="text-sm font-medium text-slate-300 mb-3">Last 7 Days + Forecast</p>
              <ReactECharts
                option={buildHistOption(fc.data.history_7d, fc.data.forecast_date, fc.data.pm25_forecast, fc.data.aqi_category)}
                style={{ height: 180 }} theme="dark" />
            </div>

            {/* Weather inputs */}
            <div className="glass rounded-2xl p-5">
              <p className="text-sm font-medium text-slate-300 mb-4">Weather Inputs Used</p>
              <div className="grid grid-cols-2 gap-3">
                {([
                  { icon: Thermometer, label: 'Temperature', value: `${fc.data.weather_inputs.temperature_2m_mean.toFixed(1)} °C`,  color: '#FF7E00' },
                  { icon: Droplets,    label: 'Humidity',    value: `${fc.data.weather_inputs.relative_humidity_2m_mean.toFixed(0)} %`,  color: '#22D3EE' },
                  { icon: Wind,        label: 'Wind Speed',  value: `${fc.data.weather_inputs.wind_speed_10m_max.toFixed(1)} km/h`,       color: '#8B5CF6' },
                  { icon: CloudRain,   label: 'Rainfall',    value: `${fc.data.weather_inputs.precipitation_sum.toFixed(1)} mm`,           color: '#60A5FA' },
                ] as const).map(({ icon: Icon, label, value, color }) => (
                  <div key={label} className="flex items-center gap-3 p-3 rounded-xl bg-white/5">
                    <Icon className="w-4 h-4 flex-shrink-0" style={{ color }} />
                    <div>
                      <p className="text-xs text-slate-500">{label}</p>
                      <p className="text-sm font-mono font-semibold text-white">{value}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}