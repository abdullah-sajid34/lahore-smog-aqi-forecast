import { useState } from 'react';
import { motion } from 'framer-motion';
import ReactECharts from 'echarts-for-react';
import { useEda, useSeries } from '../hooks/useApi';
import { ChartCard }  from '../components/ChartCard';
import { Skeleton }   from '../components/Skeleton';
import { ErrorState } from '../components/ErrorState';
import { aqiColor, COLORS } from '../theme';
import { Bars3D } from '../components/Bars3D';

const MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const WEEKDAYS    = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
const SMOG_MONTHS = new Set([10,11,12,1]);

function aqiCategory(v: number): string {
  if (v <= 12)    return 'Good';
  if (v <= 35.4)  return 'Moderate';
  if (v <= 55.4)  return 'Unhealthy for Sensitive';
  if (v <= 150.4) return 'Unhealthy';
  if (v <= 250.4) return 'Very Unhealthy';
  if (v <= 350.4) return 'Hazardous';
  return 'Hazardous+';
}

const FINDINGS = [
  { icon: '📅', title: 'Peak month: December', body: 'Monthly mean PM2.5 peaks in December at 263.7 µg/m³ — 2.19× the annual mean.' },
  { icon: '🌫', title: 'Smog season 2.77× worse', body: 'Oct–Jan mean: 199.7 µg/m³ vs off-season 72.2 µg/m³ (crop burning + winter inversions).' },
  { icon: '📈', title: 'Lag-1 autocorrelation: 0.852', body: 'Strong day-to-day persistence. A bad air day reliably predicts the next one.' },
  { icon: '🌡', title: 'Temperature r = −0.606', body: 'Strongest weather correlate. Thermal inversions in winter trap PM2.5 near the surface.' },
  { icon: '📆', title: 'Weekday effect minor: 16 µg/m³ range', body: 'Wednesday highest (122 µg/m³), Sunday lowest (106 µg/m³) — seasonal signal dominates.' },
];

export function EDA() {
  const eda    = useEda();
  const series = useSeries('2019-05-09', '2025-02-18');
  const [mode3d, setMode3d] = useState(false);

  // ── Monthly area chart ──────────────────────────────────────
  const monthlyOption = (() => {
    if (!eda.data) return {};
    const d = eda.data;
    const months = Array.from({ length: 12 }, (_, i) => i + 1);
    const vals   = months.map(m => d.monthly_mean_by_month[String(m)] ?? 0);
    // smog-season background bands
    const smogAreas: Array<[object,object]> = [
      [{ xAxis: 'Oct', itemStyle:{ color:'rgba(255,125,0,0.08)' } }, { xAxis: 'Jan' }],
    ];
    return {
      backgroundColor: 'transparent',
      grid: { top:30, right:20, bottom:40, left:55 },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#111A2E',
        borderColor: 'rgba(34,211,238,0.2)',
        textStyle: { color: '#E2E8F0', fontSize:12 },
        formatter: (p: Array<{axisValue:string;value:number}>) =>
          `${p[0].axisValue}<br/><b style="color:#22D3EE">${p[0].value.toFixed(1)} µg/m³</b>`,
      },
      xAxis: {
        type: 'category',
        data: MONTH_NAMES,
        axisLabel: {
          color: '#64748B', fontSize:11,
          formatter: (v: string, i: number) => SMOG_MONTHS.has(i+1) ? `{smog|${v}}` : v,
          rich: { smog: { color: '#FF7E00', fontWeight: 'bold' } },
        },
        axisLine: { lineStyle: { color: '#1E293B' } },
      },
      yAxis: {
        type: 'value', name: 'µg/m³',
        nameTextStyle: { color:'#64748B', fontSize:10 },
        axisLabel: { color:'#64748B', fontSize:10 },
        splitLine: { lineStyle: { color:'rgba(255,255,255,0.05)' } },
      },
      series: [{
        type: 'line', data: vals, smooth: true, symbol: 'circle', symbolSize: 6,
        lineStyle: { color: '#22D3EE', width: 2.5 },
        itemStyle: {
          color: (p: {dataIndex:number}) => SMOG_MONTHS.has(p.dataIndex+1) ? '#FF7E00' : '#22D3EE',
        },
        areaStyle: {
          color: {
            type:'linear', x:0,y:0,x2:0,y2:1,
            colorStops:[{offset:0,color:'rgba(34,211,238,0.35)'},{offset:1,color:'rgba(34,211,238,0.02)'}],
          },
        },
        markArea: { data: smogAreas, silent:true },
        markLine: {
          silent:true,
          lineStyle:{ color:'#FFD60A', type:'dashed', width:1 },
          data:[{ yAxis:35, label:{ formatter:'NAAQS 35', color:'#FFD60A', fontSize:9 } }],
        },
      }],
    };
  })();

  // ── Weekday bars ────────────────────────────────────────────
  const weekdayOption = (() => {
    if (!eda.data) return {};
    const vals = WEEKDAYS.map(d => eda.data!.weekday_means_ugm3[d] ?? 0);
    const max  = Math.max(...vals);
    return {
      backgroundColor:'transparent',
      grid:{ top:20, right:20, bottom:40, left:55 },
      tooltip:{
        trigger:'axis', backgroundColor:'#111A2E', borderColor:'rgba(34,211,238,0.2)',
        textStyle:{ color:'#E2E8F0', fontSize:12 },
        formatter:(p:Array<{axisValue:string;value:number}>) =>
          `${p[0].axisValue}<br/><b style="color:#22D3EE">${p[0].value.toFixed(1)} µg/m³</b>`,
      },
      xAxis:{ type:'category', data:WEEKDAYS, axisLabel:{ color:'#64748B' }, axisLine:{ lineStyle:{ color:'#1E293B' } } },
      yAxis:{ type:'value', name:'µg/m³', min:80, nameTextStyle:{color:'#64748B',fontSize:10}, axisLabel:{color:'#64748B'}, splitLine:{lineStyle:{color:'rgba(255,255,255,0.05)'}} },
      series:[{
        type:'bar', data:vals, barMaxWidth:48,
        itemStyle:{
          color:(p:{value:number}) => p.value===max ? '#FF7E00' : '#22D3EE',
          borderRadius:[6,6,0,0],
        },
        label:{ show:true, position:'top', color:'#94A3B8', fontSize:9, formatter:(p:{value:number})=>`${p.value.toFixed(0)}` },
      }],
    };
  })();

  // ── Month × Year heatmap ────────────────────────────────────
  const heatmapOption = (() => {
    if (!series.data) return {};
    // Aggregate monthly means by year
    const byYearMonth: Record<string, number[]> = {};
    for (const pt of series.data.series) {
      const [y,m] = pt.date.split('-');
      const key = y;
      if (!byYearMonth[key]) byYearMonth[key] = Array(12).fill(0).map(() => 0);
      // We'll average later; for now accumulate
    }
    // Build from series
    type Cell = [number, number, number]; // [month_idx, year_idx, value]
    const years = [...new Set(series.data.series.map(p => p.date.slice(0,4)))].sort();
    const agg: Record<string, Record<number,{sum:number;cnt:number}>> = {};
    for (const pt of series.data.series) {
      const y = pt.date.slice(0,4);
      const mo = parseInt(pt.date.slice(5,7)) - 1;
      if (!agg[y]) agg[y] = {};
      if (!agg[y][mo]) agg[y][mo] = {sum:0,cnt:0};
      agg[y][mo].sum += pt.pm25_ugm3;
      agg[y][mo].cnt++;
    }
    const data: Cell[] = [];
    for (const [yi,yr] of years.entries()) {
      for (let mo=0; mo<12; mo++) {
        const a = agg[yr]?.[mo];
        if (a && a.cnt>0) data.push([mo, yi, Math.round(a.sum/a.cnt)]);
      }
    }
    return {
      backgroundColor:'transparent',
      tooltip:{
        formatter:(p:{data:[number,number,number]}) =>
          `${years[p.data[1]]}-${MONTH_NAMES[p.data[0]]}: <b>${p.data[2]} µg/m³</b>`,
        backgroundColor:'#111A2E', borderColor:'rgba(34,211,238,0.2)', textStyle:{color:'#E2E8F0'},
      },
      grid:{ top:20, bottom:50, left:55, right:60 },
      xAxis:{ type:'category', data:MONTH_NAMES, splitArea:{show:true}, axisLabel:{color:'#64748B',fontSize:10} },
      yAxis:{ type:'category', data:years, splitArea:{show:true}, axisLabel:{color:'#64748B',fontSize:10} },
      visualMap:{
        min:0, max:300,
        calculable:true, orient:'horizontal', left:'center', bottom:4,
        inRange:{ color:['#00E400','#FFD60A','#FF7E00','#FF2D2D','#8F3F97','#7E0023'] },
        textStyle:{ color:'#64748B', fontSize:9 },
      },
      series:[{
        type:'heatmap', data,
        label:{ show:true, fontSize:8, color:'white', formatter:(p:{data:[number,number,number]})=>`${p.data[2]}` },
        emphasis:{ itemStyle:{ shadowBlur:10, shadowColor:'rgba(0,0,0,0.5)' } },
      }],
    };
  })();

  // ── Autocorrelation bars ────────────────────────────────────
  const autocorrOption = (() => {
    if (!eda.data) return {};
    const lags = [1,2,3,4,5,6,7];
    const d = eda.data;
    const autocorrs: Record<string,number> = {
      lag1: d.autocorr_lag1 ?? 0.852, lag2: 0.766, lag3: 0.710,
      lag4: 0.685, lag5: 0.666, lag6: 0.651, lag7: d.autocorr_lag7 ?? 0.644,
    };
    const vals = lags.map(l => autocorrs[`lag${l}`] ?? 0);
    return {
      backgroundColor:'transparent',
      grid:{ top:20, right:20, bottom:40, left:55 },
      tooltip:{ trigger:'axis', backgroundColor:'#111A2E', borderColor:'rgba(34,211,238,0.2)', textStyle:{color:'#E2E8F0',fontSize:12} },
      xAxis:{ type:'category', data:lags.map(l=>`Lag ${l}`), axisLabel:{color:'#64748B'}, axisLine:{lineStyle:{color:'#1E293B'}} },
      yAxis:{ type:'value', min:0, max:1, name:'Pearson r', nameTextStyle:{color:'#64748B',fontSize:10}, axisLabel:{color:'#64748B'}, splitLine:{lineStyle:{color:'rgba(255,255,255,0.05)'}} },
      series:[{
        type:'bar', data:vals, barMaxWidth:48,
        itemStyle:{ color:'#8B5CF6', borderRadius:[6,6,0,0] },
        label:{ show:true, position:'top', color:'#94A3B8', fontSize:9, formatter:(p:{value:number})=>`${p.value.toFixed(3)}` },
      }],
    };
  })();

  // ── Weather correlation bars ────────────────────────────────
  const weatherCorrOption = (() => {
    if (!eda.data) return {};
    const corr = eda.data.weather_correlations;
    const labels = Object.keys(corr).map(k => k.replace('_2m_mean','').replace('_10m_max','').replace('_sum','').replace(/_/g,' '));
    const vals   = Object.values(corr);
    return {
      backgroundColor:'transparent',
      grid:{ top:20, right:20, bottom:60, left:55 },
      tooltip:{ trigger:'axis', backgroundColor:'#111A2E', borderColor:'rgba(34,211,238,0.2)', textStyle:{color:'#E2E8F0',fontSize:12} },
      xAxis:{ type:'category', data:labels, axisLabel:{color:'#64748B',rotate:20,fontSize:10}, axisLine:{lineStyle:{color:'#1E293B'}} },
      yAxis:{ type:'value', min:-1, max:1, name:'Pearson r', nameTextStyle:{color:'#64748B',fontSize:10}, axisLabel:{color:'#64748B',fontSize:10}, splitLine:{lineStyle:{color:'rgba(255,255,255,0.05)'}} },
      series:[{
        type:'bar', data:vals, barMaxWidth:60,
        itemStyle:{
          color:(p:{value:number})=>p.value<0 ? '#FF7E00' : '#22D3EE',
          borderRadius:[4,4,0,0],
        },
        label:{ show:true, position:'top', color:'#94A3B8', fontSize:9, formatter:(p:{value:number})=>`${p.value.toFixed(3)}` },
      }],
    };
  })();

  // ── 3D bars data for month heatmap ─────────────────────────
  const bars3dData = (() => {
    if (!eda.data) return [];
    return Array.from({length:12},(_,i)=>{
      const val = eda.data!.monthly_mean_by_month[String(i+1)] ?? 0;
      return { x:i, z:0, height:val, color:aqiColor(aqiCategory(val)), label:MONTH_NAMES[i] };
    });
  })();

  const isLoading = eda.isLoading || series.isLoading;
  const hasError  = !!eda.error;

  return (
    <div className="flex flex-col gap-6 p-6 overflow-y-auto h-full">
      {/* Header */}
      <motion.div initial={{opacity:0,y:-10}} animate={{opacity:1,y:0}} className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-bold text-white mb-1">Exploratory Data Analysis</h2>
          <p className="text-sm text-slate-400">Training data: May 2019 – Jun 2023 · 1,317 observations</p>
        </div>
      </motion.div>

      {hasError && <ErrorState message={eda.error?.message ?? 'Failed to load EDA data'} onRetry={()=>eda.refetch()} />}

      {/* Row 1: Monthly trend + Weekday */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <ChartCard title="Monthly Mean PM2.5" subtitle="Oct–Jan smog season highlighted in orange · NAAQS 35 µg/m³ marker">
          {isLoading ? <Skeleton className="h-56" /> :
           <ReactECharts option={monthlyOption} style={{height:220}} theme="dark" opts={{renderer:'canvas'}} />}
        </ChartCard>
        <ChartCard title="Weekday Effect" subtitle="Mean PM2.5 by day of week — seasonal signal dominates">
          {isLoading ? <Skeleton className="h-56" /> :
           <ReactECharts option={weekdayOption} style={{height:220}} theme="dark" opts={{renderer:'canvas'}} />}
        </ChartCard>
      </div>

      {/* Row 2: Month × Year heatmap with Flat/3D toggle */}
      <ChartCard
        title="Month × Year Heatmap"
        subtitle="Mean PM2.5 per month per year · continuous AQI colour scale"
        className="relative"
      >
        <div className="absolute top-4 right-5 flex gap-1 z-10">
          {(['Flat','3D'] as const).map(m=>(
            <button key={m} onClick={()=>setMode3d(m==='3D')}
              className={`px-3 py-1 text-xs rounded-lg border transition-all
                ${mode3d===(m==='3D') ? 'bg-accent/20 border-accent/40 text-accent' : 'border-white/10 text-slate-400 hover:border-white/20'}`}>
              {m}
            </button>
          ))}
        </div>
        {isLoading ? <Skeleton className="h-64" /> :
          mode3d
            ? <div style={{height:260}}><Bars3D data={bars3dData} /></div>
            : <ReactECharts option={heatmapOption} style={{height:260}} theme="dark" opts={{renderer:'canvas'}} />
        }
      </ChartCard>

      {/* Row 3: Autocorrelation + Weather correlation */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <ChartCard title="Autocorrelation (Lags 1–7)" subtitle="Pearson r between PM2.5[t] and PM2.5[t−lag] · calendar-day correct">
          {isLoading ? <Skeleton className="h-48" /> :
           <ReactECharts option={autocorrOption} style={{height:190}} theme="dark" opts={{renderer:'canvas'}} />}
        </ChartCard>
        <ChartCard title="Weather Correlations" subtitle="Pearson r with PM2.5 · training split only">
          {isLoading ? <Skeleton className="h-48" /> :
           <ReactECharts option={weatherCorrOption} style={{height:190}} theme="dark" opts={{renderer:'canvas'}} />}
        </ChartCard>
      </div>

      {/* Key Findings */}
      <div className="glass rounded-2xl p-6">
        <h3 className="text-base font-semibold text-white mb-4">Key Findings</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {FINDINGS.map((f,i)=>(
            <motion.div key={f.title}
              initial={{opacity:0,y:16}} animate={{opacity:1,y:0}}
              transition={{delay:0.1*i, duration:0.4}}
              className="p-4 rounded-xl bg-white/5 border border-white/5 hover:border-accent/20 transition-colors">
              <div className="text-2xl mb-2">{f.icon}</div>
              <p className="text-sm font-semibold text-slate-200 mb-1">{f.title}</p>
              <p className="text-xs text-slate-400 leading-relaxed">{f.body}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}
