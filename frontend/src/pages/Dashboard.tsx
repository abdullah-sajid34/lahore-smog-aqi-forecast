import { lazy, Suspense } from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Award, Target, Zap, Calendar } from 'lucide-react';
import ReactECharts from 'echarts-for-react';
import { useKpis, useToday, useSeries, useForecast } from '../hooks/useApi';
import { Gauge }      from '../components/Gauge';
import { KpiCard }    from '../components/KpiCard';
import { AQIBadge }   from '../components/AQIBadge';
import { ChartCard }  from '../components/ChartCard';
import { Skeleton }   from '../components/Skeleton';
import { ErrorState } from '../components/ErrorState';
import { COLORS, aqiColor } from '../theme';
import type { SeriesPoint } from '../types/api';

const SmogScene = lazy(() =>
  import('../components/SmogScene').then(m => ({ default: m.SmogScene }))
);

const FORECAST_DATE  = '2025-02-17';
const SERIES_END     = '2025-02-18';
const SERIES_START   = '2024-11-20';

function SceneFallback() {
  return <div className="absolute inset-0 rounded-3xl"
    style={{ background: 'radial-gradient(ellipse at 50% 60%,rgba(139,92,246,0.18) 0%,transparent 70%)' }} />;
}

export function Dashboard() {
  const kpis   = useKpis();
  const today  = useToday();
  const series = useSeries(SERIES_START, SERIES_END);
  const fc     = useForecast(FORECAST_DATE, 'random_forest');

  const pm25     = today.data?.pm25_ugm3   ?? null;
  const category = today.data?.aqi_category ?? 'Unknown';

  function buildChartOption(pts: SeriesPoint[]) {
    const dates  = pts.map(p => p.date);
    const values = pts.map(p => p.pm25_ugm3);
    const bands: Array<[object, object]> = [
      [{ yAxis: 0,     itemStyle: { color: 'rgba(0,228,0,0.07)'   } }, { yAxis: 12    }],
      [{ yAxis: 12,    itemStyle: { color: 'rgba(255,214,10,0.07)' } }, { yAxis: 35.4  }],
      [{ yAxis: 35.4,  itemStyle: { color: 'rgba(255,126,0,0.07)' }  }, { yAxis: 55.4  }],
      [{ yAxis: 55.4,  itemStyle: { color: 'rgba(255,45,45,0.07)' }  }, { yAxis: 150.4 }],
      [{ yAxis: 150.4, itemStyle: { color: 'rgba(143,63,151,0.07)'}  }, { yAxis: 250.4 }],
      [{ yAxis: 250.4, itemStyle: { color: 'rgba(126,0,35,0.07)'  }  }, { yAxis: 500   }],
    ];
    return {
      backgroundColor: 'transparent',
      grid: { top: 20, right: 20, bottom: 60, left: 55 },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#111A2E',
        borderColor: 'rgba(34,211,238,0.2)',
        textStyle: { color: '#E2E8F0', fontSize: 12 },
        formatter: (params: Array<{ axisValue: string; value: number }>) => {
          const p   = params[0];
          const cat = pts.find(d => d.date === p.axisValue)?.aqi_category ?? '';
          return `${p.axisValue}<br/><b style="color:#22D3EE">${p.value.toFixed(1)} µg/m³</b><br/>${cat}`;
        },
      },
      xAxis: {
        type: 'category', data: dates,
        axisLabel: { color: '#64748B', fontSize: 10 },
        axisLine:  { lineStyle: { color: '#1E293B' } },
      },
      yAxis: {
        type: 'value', name: 'µg/m³',
        nameTextStyle: { color: '#64748B', fontSize: 10 },
        axisLabel: { color: '#64748B', fontSize: 10 },
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } },
      },
      dataZoom: [
        { type: 'slider', bottom: 8, height: 20, fillerColor: 'rgba(34,211,238,0.1)', borderColor: 'rgba(34,211,238,0.2)', textStyle: { color: '#64748B' } },
        { type: 'inside' },
      ],
      series: [{
        type: 'line', data: values, smooth: true, symbol: 'none',
        lineStyle: { color: '#22D3EE', width: 2 },
        areaStyle: {
          color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [{ offset: 0, color: 'rgba(34,211,238,0.35)' }, { offset: 1, color: 'rgba(34,211,238,0.02)' }] },
        },
        markLine: {
          silent: true,
          lineStyle: { color: '#FFD60A', type: 'dashed', width: 1 },
          data: [{ yAxis: 35, label: { formatter: 'NAAQS 35', color: '#FFD60A', fontSize: 10 } }],
        },
        markArea: { data: bands, silent: true },
      }],
    };
  }

  return (
    <div className="flex flex-col gap-6 p-6 overflow-y-auto h-full">

      {/* Hero */}
      <section className="relative rounded-3xl overflow-hidden" style={{ minHeight: 320, background: 'linear-gradient(135deg,#0B1220,#111A2E)' }}>
        <Suspense fallback={<SceneFallback />}>
          {pm25 !== null && <SmogScene pm25={pm25} />}
        </Suspense>
        <div className="relative z-10 flex flex-col md:flex-row items-center justify-around p-8 gap-6">
          {/* Gauge */}
          <div>
            {today.isLoading && <Skeleton className="w-72 h-56" />}
            {today.error    && <ErrorState message="Could not load today's data" />}
            {today.data     && <Gauge value={pm25} category={category} />}
          </div>
          {/* Forecast card */}
          <motion.div initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}
            className="glass rounded-2xl p-6 max-w-xs w-full">
            <p className="text-xs text-slate-400 uppercase tracking-wider mb-3 font-medium">Tomorrow's Forecast</p>
            {fc.isLoading && <Skeleton className="h-24" />}
            {fc.error     && <ErrorState message={fc.error.message} />}
            {fc.data && (
              <>
                <div className="flex items-baseline gap-2 mb-2">
                  <span className="text-4xl font-bold font-mono" style={{ color: aqiColor(fc.data.aqi_category) }}>
                    {fc.data.pm25_forecast.toFixed(1)}
                  </span>
                  <span className="text-sm text-slate-400">µg/m³</span>
                  {pm25 !== null && (
                    <span className={`ml-2 flex items-center text-sm font-medium ${fc.data.pm25_forecast > pm25 ? 'text-red-400' : 'text-green-400'}`}>
                      {fc.data.pm25_forecast > pm25 ? <TrendingUp className="w-4 h-4 mr-0.5" /> : <TrendingDown className="w-4 h-4 mr-0.5" />}
                      {Math.abs(fc.data.pm25_forecast - pm25).toFixed(1)}
                    </span>
                  )}
                </div>
                <AQIBadge category={fc.data.aqi_category} pulse size="sm" />
                <p className="text-xs text-slate-400 mt-3 leading-relaxed">{fc.data.advisory}</p>
                <p className="text-xs text-slate-600 mt-2 italic">{fc.data.disclaimer}</p>
              </>
            )}
          </motion.div>
        </div>
      </section>

      {/* KPI Cards */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard title="2024 Annual Average" delay={0}
          value={kpis.data?.annual_mean_pm25_ugm3['2024'] ?? null} unit="µg/m³"
          color={COLORS.accent} icon={<Calendar className="w-4 h-4" />} subtitle="All-year mean PM2.5" />
        <KpiCard title="All-time Peak" delay={1}
          value={kpis.data?.peak_pm25_ugm3 ?? null} unit="µg/m³"
          color="#FF2D2D" icon={<Zap className="w-4 h-4" />}
          subtitle={kpis.data ? `on ${kpis.data.peak_date}` : ''} />
        <KpiCard title="Best Model RMSE" delay={2}
          value={kpis.data?.best_model_rmse_test ?? null} unit="µg/m³"
          color={COLORS.violet} icon={<Target className="w-4 h-4" />}
          subtitle={kpis.data?.best_model_by_val} format={v => v.toFixed(2)} />
        <KpiCard title="Skill vs Persistence" delay={3}
          value={kpis.data ? kpis.data.best_model_skill_test * 100 : null} unit="%"
          color="#00E400" icon={<Award className="w-4 h-4" />}
          subtitle="Higher = better than naive" format={v => v.toFixed(1)} />
      </section>

      {/* 90-day chart */}
      <ChartCard title="PM2.5 — Last 90 Days" subtitle="Gradient fill · AQI bands · NAAQS marker · Zoom enabled">
        {series.isLoading && <Skeleton className="h-64" />}
        {series.error     && <ErrorState message="Could not load series data" />}
        {series.data      && (
          <ReactECharts option={buildChartOption(series.data.series)} style={{ height: 280 }} theme="dark" />
        )}
      </ChartCard>

      {/* Smog season banner */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }}
        className="glass rounded-2xl p-5 flex items-center gap-4"
        style={{ borderColor: 'rgba(255,125,0,0.3)', background: 'rgba(255,125,0,0.06)' }}>
        <div className="w-3 h-3 rounded-full bg-orange-400 animate-pulse2 flex-shrink-0" />
        <div>
          <p className="text-sm font-semibold text-orange-300">🌫  Smog Season: October – January</p>
          <p className="text-xs text-slate-400 mt-0.5">
            PM2.5 averages <span className="text-orange-300 font-mono font-bold">2.77×</span> higher
            than off-season (199.7 vs 72.2 µg/m³) · Crop burning + winter thermal inversions
          </p>
        </div>
      </motion.div>
    </div>
  );
}