import { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { ChevronUp, ChevronDown, ArrowUpDown } from 'lucide-react';
import ReactECharts from 'echarts-for-react';
import { useModels } from '../hooks/useApi';
import { ChartCard }  from '../components/ChartCard';
import { Skeleton }   from '../components/Skeleton';
import { ErrorState } from '../components/ErrorState';
import { aqiColor } from '../theme';

type SortKey  = 'model' | 'rmse' | 'mae' | 'f1' | 'skill';
type SortDir  = 'asc' | 'desc';

const MODEL_DISPLAY: Record<string, string> = {
  Persistence:         'Persistence',
  MovingAverage7:      '7-day MA',
  Ridge:               'Ridge',
  RandomForest:        'Random Forest',
  'RandomForest+Kalman':'RF + Kalman',
};

function aqiCategory(v: number): string {
  if (v <= 12)    return 'Good';
  if (v <= 35.4)  return 'Moderate';
  if (v <= 55.4)  return 'Unhealthy for Sensitive';
  if (v <= 150.4) return 'Unhealthy';
  if (v <= 250.4) return 'Very Unhealthy';
  if (v <= 350.4) return 'Hazardous';
  return 'Hazardous+';
}

export function Models() {
  const models = useModels();
  const [sort, setSort]     = useState<SortKey>('rmse');
  const [dir, setDir]       = useState<SortDir>('asc');
  const [tab, setTab]       = useState<'rf' | 'ridge'>('rf');
  const [playing, setPlaying] = useState(false);
  const [playIdx, setPlayIdx] = useState(0);
  const playRef  = useRef<ReturnType<typeof setInterval> | null>(null);

  // Sort comparison table
  const sorted = (() => {
    if (!models.data) return [];
    const rows = [...models.data.comparison];
    rows.sort((a, b) => {
      let va: number | string = a[sort as keyof typeof a] ?? 0;
      let vb: number | string = b[sort as keyof typeof b] ?? 0;
      if (typeof va === 'string') va = va.toLowerCase();
      if (typeof vb === 'string') vb = vb.toLowerCase();
      if (va < vb) return dir === 'asc' ? -1 : 1;
      if (va > vb) return dir === 'asc' ?  1 : -1;
      return 0;
    });
    return rows;
  })();

  function toggleSort(key: SortKey) {
    if (sort === key) setDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSort(key); setDir('asc'); }
  }

  // Best model by lowest RMSE (excluding Persistence)
  const bestModel = (() => {
    if (!models.data) return '';
    const non = models.data.comparison.filter(r => r.model !== 'Persistence');
    return non.reduce((a,b) => a.rmse < b.rmse ? a : b).model;
  })();

  // Animated play for predictions chart
  const preds = models.data?.predictions_test ?? [];
  useEffect(() => {
    if (!playing) { if (playRef.current) clearInterval(playRef.current); return; }
    setPlayIdx(0);
    playRef.current = setInterval(() => {
      setPlayIdx(i => {
        if (i >= preds.length - 1) {
          setPlaying(false);
          return preds.length - 1;
        }
        return i + 1;
      });
    }, 18);
    return () => { if (playRef.current) clearInterval(playRef.current); };
  }, [playing, preds.length]);

  // ── RMSE/MAE grouped bars ────────────────────────────────────
  const rmseMaeOption = (() => {
    if (!models.data) return {};
    const rows = models.data.comparison;
    const names = rows.map(r => MODEL_DISPLAY[r.model] ?? r.model);
    return {
      backgroundColor: 'transparent',
      grid: { top:30, right:20, bottom:60, left:55 },
      legend: { data:['RMSE','MAE'], top:4, textStyle:{ color:'#94A3B8', fontSize:10 }, inactiveColor:'#2D3748' },
      tooltip: {
        trigger:'axis', backgroundColor:'#111A2E', borderColor:'rgba(34,211,238,0.2)',
        textStyle:{ color:'#E2E8F0', fontSize:11 },
      },
      xAxis: { type:'category', data:names, axisLabel:{ color:'#64748B', rotate:15, fontSize:9 }, axisLine:{ lineStyle:{ color:'#1E293B' } } },
      yAxis: { type:'value', name:'µg/m³', nameTextStyle:{ color:'#64748B',fontSize:9 }, axisLabel:{ color:'#64748B',fontSize:9 }, splitLine:{ lineStyle:{ color:'rgba(255,255,255,0.05)' } } },
      series: [
        { name:'RMSE', type:'bar', data:rows.map(r=>r.rmse), barMaxWidth:28, itemStyle:{ color:'#FF7E00',borderRadius:[4,4,0,0] } },
        { name:'MAE',  type:'bar', data:rows.map(r=>r.mae),  barMaxWidth:28, itemStyle:{ color:'#22D3EE',borderRadius:[4,4,0,0] } },
      ],
    };
  })();

  // ── Predicted vs actual line chart (animated play) ───────────
  const predOption = (() => {
    if (!preds.length) return {};
    const slice = preds.slice(0, Math.max(1, playIdx + 1));
    const dates = slice.map(p => p.date);
    const aqi_bands: Array<[object,object]> = [
      [{ yAxis:0,     itemStyle:{ color:'rgba(0,228,0,0.07)'   } },{ yAxis:12    }],
      [{ yAxis:12,    itemStyle:{ color:'rgba(255,214,10,0.07)' } },{ yAxis:35.4  }],
      [{ yAxis:35.4,  itemStyle:{ color:'rgba(255,126,0,0.07)' }  },{ yAxis:55.4  }],
      [{ yAxis:55.4,  itemStyle:{ color:'rgba(255,45,45,0.07)' }  },{ yAxis:150.4 }],
      [{ yAxis:150.4, itemStyle:{ color:'rgba(143,63,151,0.07)'}  },{ yAxis:250.4 }],
      [{ yAxis:250.4, itemStyle:{ color:'rgba(126,0,35,0.07)'  }  },{ yAxis:500   }],
    ];
    return {
      backgroundColor:'transparent',
      animation: false,
      grid:{ top:30, right:20, bottom:60, left:55 },
      legend:{ data:['Actual','Persistence','7-day MA','Ridge','Random Forest'], top:4, textStyle:{ color:'#94A3B8',fontSize:9 } },
      tooltip:{
        trigger:'axis', backgroundColor:'#111A2E', borderColor:'rgba(34,211,238,0.2)',
        textStyle:{ color:'#E2E8F0',fontSize:11 },
        formatter:(p:Array<{seriesName:string;value:number;axisValue:string}>)=>
          `${p[0].axisValue}<br/>${p.map(s=>`<span style="color:${s.seriesName==='Actual'?'#E2E8F0':'inherit'}">${s.seriesName}: ${s.value?.toFixed(1)}</span>`).join('<br/>')}`,
      },
      dataZoom:[
        { type:'slider', bottom:4, height:16, fillerColor:'rgba(34,211,238,0.1)', borderColor:'rgba(34,211,238,0.2)', textStyle:{ color:'#64748B',fontSize:8 } },
        { type:'inside' },
      ],
      xAxis:{ type:'category', data:dates, axisLabel:{ color:'#64748B',fontSize:9,rotate:30 }, axisLine:{ lineStyle:{ color:'#1E293B' } } },
      yAxis:{ type:'value', name:'µg/m³', nameTextStyle:{ color:'#64748B',fontSize:9 }, axisLabel:{ color:'#64748B',fontSize:9 }, splitLine:{ lineStyle:{ color:'rgba(255,255,255,0.05)' } } },
      series:[
        { name:'Actual',       type:'line', data:slice.map(p=>p.actual),      smooth:true, symbol:'none', lineStyle:{ color:'#E2E8F0',width:1.5 }, markArea:{ data:aqi_bands,silent:true } },
        { name:'Persistence',  type:'line', data:slice.map(p=>p.persistence), smooth:true, symbol:'none', lineStyle:{ color:'#64748B',width:1,type:'dashed' } },
        { name:'7-day MA',     type:'line', data:slice.map(p=>p.moving_avg7), smooth:true, symbol:'none', lineStyle:{ color:'#FFD60A',width:1 } },
        { name:'Ridge',        type:'line', data:slice.map(p=>p.ridge),       smooth:true, symbol:'none', lineStyle:{ color:'#8B5CF6',width:1.5 } },
        { name:'Random Forest',type:'line', data:slice.map(p=>p.rf),          smooth:true, symbol:'none', lineStyle:{ color:'#22D3EE',width:2 } },
      ],
    };
  })();

  // ── Confusion matrix ────────────────────────────────────────
  const cmOption = (() => {
    if (!models.data) return {};
    const { labels, matrix } = models.data.confusion_matrix;
    // Row-normalise
    const data: Array<[number,number,number]> = [];
    matrix.forEach((row, r) => {
      const rowSum = row.reduce((a,b)=>a+b,0) || 1;
      row.forEach((v, c) => data.push([c, r, Math.round(v/rowSum*100)]));
    });
    return {
      backgroundColor:'transparent',
      tooltip:{
        formatter:(p:{data:[number,number,number]})=>
          `Actual: ${labels[p.data[1]]}<br/>Pred: ${labels[p.data[0]]}<br/><b>${p.data[2]}%</b>`,
        backgroundColor:'#111A2E', borderColor:'rgba(34,211,238,0.2)', textStyle:{ color:'#E2E8F0',fontSize:11 },
      },
      grid:{ top:30, right:80, bottom:80, left:120 },
      xAxis:{ type:'category', data:labels.map(l=>l.slice(0,8)), axisLabel:{ color:'#64748B',rotate:30,fontSize:9 }, name:'Predicted', nameLocation:'middle', nameGap:50, nameTextStyle:{ color:'#64748B' } },
      yAxis:{ type:'category', data:labels, axisLabel:{ color:'#64748B',fontSize:9 }, name:'Actual', nameLocation:'middle', nameGap:100, nameTextStyle:{ color:'#64748B' } },
      visualMap:{
        min:0, max:100, calculable:true, orient:'vertical', right:4, top:'center',
        inRange:{ color:['#0B1220','rgba(34,211,238,0.3)','#22D3EE'] },
        textStyle:{ color:'#64748B',fontSize:9 },
        formatter:(v:number)=>`${v}%`,
      },
      series:[{
        type:'heatmap', data,
        label:{ show:true, fontSize:9, color:'white', formatter:(p:{data:[number,number,number]})=>`${p.data[2]}%` },
        emphasis:{ itemStyle:{ shadowBlur:8, shadowColor:'rgba(34,211,238,0.4)' } },
      }],
    };
  })();

  // ── Feature importance / Ridge coefficients ─────────────────
  const importanceOption = (() => {
    if (!models.data) return {};
    if (tab === 'rf') {
      const imp = models.data.feature_importance.importances;
      const items = Object.entries(imp).sort((a,b)=>b[1]-a[1]);
      return {
        backgroundColor:'transparent',
        grid:{ top:10, right:20, bottom:10, left:120 },
        tooltip:{ backgroundColor:'#111A2E', borderColor:'rgba(34,211,238,0.2)', textStyle:{ color:'#E2E8F0',fontSize:11 } },
        xAxis:{ type:'value', axisLabel:{ color:'#64748B',fontSize:9 }, splitLine:{ lineStyle:{ color:'rgba(255,255,255,0.05)' } } },
        yAxis:{ type:'category', data:items.map(i=>i[0]), axisLabel:{ color:'#64748B',fontSize:9 }, inverse:true },
        series:[{
          type:'bar', data:items.map(i=>i[1]),
          itemStyle:{ color:(p:{value:number})=>p.value>0.1?'#FF7E00':'#22D3EE', borderRadius:[0,4,4,0] },
          label:{ show:true, position:'right', color:'#94A3B8', fontSize:8, formatter:(p:{value:number})=>`${(p.value*100).toFixed(1)}%` },
        }],
      };
    } else {
      // Ridge coefficients (absolute values, sorted)
      // We use the seed_stability placeholder; real coefs come from /api/models feature_importance
      const imp = models.data.feature_importance.importances;
      const items = Object.entries(imp).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]));
      return {
        backgroundColor:'transparent',
        grid:{ top:10, right:20, bottom:10, left:120 },
        tooltip:{ backgroundColor:'#111A2E', borderColor:'rgba(34,211,238,0.2)', textStyle:{ color:'#E2E8F0',fontSize:11 } },
        xAxis:{ type:'value', axisLabel:{ color:'#64748B',fontSize:9 }, splitLine:{ lineStyle:{ color:'rgba(255,255,255,0.05)' } } },
        yAxis:{ type:'category', data:items.map(i=>i[0]), axisLabel:{ color:'#64748B',fontSize:9 }, inverse:true },
        series:[{
          type:'bar', data:items.map(i=>i[1]),
          itemStyle:{ color:(p:{value:number})=>p.value>0.05?'#8B5CF6':'#22D3EE', borderRadius:[0,4,4,0] },
          label:{ show:true, position:'right', color:'#94A3B8', fontSize:8, formatter:(p:{value:number})=>`${(p.value*100).toFixed(1)}%` },
        }],
      };
    }
  })();

  // ── Seed spread (RF 10 seeds) ───────────────────────────────
  const seedOption = (() => {
    if (!models.data) return {};
    const ss = models.data.seed_stability;
    const metrics = ['RMSE','MAE','F1 (×100)'];
    const means = [ss.rmse_mean, ss.mae_mean, ss.f1_mean * 100];
    const stds  = [ss.rmse_std,  ss.mae_std,  ss.f1_std  * 100];
    return {
      backgroundColor:'transparent',
      grid:{ top:30, right:20, bottom:40, left:70 },
      tooltip:{ backgroundColor:'#111A2E', borderColor:'rgba(34,211,238,0.2)', textStyle:{ color:'#E2E8F0',fontSize:11 } },
      xAxis:{ type:'category', data:metrics, axisLabel:{ color:'#64748B',fontSize:10 }, axisLine:{ lineStyle:{ color:'#1E293B' } } },
      yAxis:{ type:'value', axisLabel:{ color:'#64748B',fontSize:9 }, splitLine:{ lineStyle:{ color:'rgba(255,255,255,0.05)' } } },
      series:[
        {
          type:'bar', data:means, barWidth:40,
          itemStyle:{ color:'#22D3EE', borderRadius:[6,6,0,0] },
          label:{ show:true, position:'top', color:'#94A3B8', fontSize:9, formatter:(p:{value:number})=>`${p.value.toFixed(2)}` },
        },
        {
          // Error bar simulation with custom marks
          type:'custom',
          renderItem: (_: unknown, api: { value:(i:number)=>number; coord:(v:[number,number])=>[number,number]; style:()=>object }) => {
            const x     = api.coord([api.value(0), 0])[0];
            const mean  = api.value(1);
            const std   = api.value(2);
            const yMean = api.coord([0, mean])[1];
            const yUp   = api.coord([0, mean + std])[1];
            const yDn   = api.coord([0, mean - std])[1];
            return {
              type:'group', children:[
                { type:'line', shape:{ x1:x,y1:yUp,x2:x,y2:yDn }, style:{ stroke:'#FF7E00',lineWidth:2 } },
                { type:'line', shape:{ x1:x-8,y1:yUp,x2:x+8,y2:yUp }, style:{ stroke:'#FF7E00',lineWidth:2 } },
                { type:'line', shape:{ x1:x-8,y1:yDn,x2:x+8,y2:yDn }, style:{ stroke:'#FF7E00',lineWidth:2 } },
              ],
            };
          },
          data: means.map((m,i)=>[i,m,stds[i]]),
          z: 10,
        },
      ],
    };
  })();

  if (models.isLoading) return (
    <div className="flex flex-col gap-6 p-6">
      {[1,2,3].map(i=><Skeleton key={i} className="h-48" />)}
    </div>
  );

  if (models.error) return (
    <div className="p-6"><ErrorState message={models.error.message} onRetry={()=>models.refetch()} /></div>
  );

  return (
    <div className="flex flex-col gap-6 p-6 overflow-y-auto h-full">
      {/* Header */}
      <motion.div initial={{opacity:0,y:-10}} animate={{opacity:1,y:0}}>
        <h2 className="text-xl font-bold text-white mb-1">Model Comparison</h2>
        <p className="text-sm text-slate-400">Test set: Apr 2024 – Feb 2025 · 255 observations</p>
      </motion.div>

      {/* Comparison table */}
      <div className="glass rounded-2xl overflow-hidden">
        <div className="p-4 border-b border-white/5">
          <h3 className="text-sm font-semibold text-white">Performance Metrics</h3>
          <p className="text-xs text-slate-400 mt-0.5">Skill score = 1 − RMSE(model)/RMSE(persistence) · Negative = worse than naive baseline</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/5">
                {(['model','rmse','mae','f1','skill'] as SortKey[]).map(col=>(
                  <th key={col} className="px-4 py-3 text-left cursor-pointer hover:text-accent transition-colors" onClick={()=>toggleSort(col)}>
                    <div className="flex items-center gap-1 text-slate-400 text-xs font-medium uppercase tracking-wide">
                      {col.toUpperCase()}
                      {sort===col ? (dir==='asc'?<ChevronUp className="w-3 h-3"/>:<ChevronDown className="w-3 h-3"/>):<ArrowUpDown className="w-3 h-3 opacity-30"/>}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((row,i)=>{
                const isBest = row.model === bestModel;
                const maxRmse = Math.max(...sorted.map(r=>r.rmse));
                return (
                  <motion.tr key={row.model} initial={{opacity:0,x:-10}} animate={{opacity:1,x:0}} transition={{delay:i*0.05}}
                    className={`border-b border-white/5 hover:bg-white/5 transition-colors ${isBest?'relative':''}`}
                    style={isBest ? { boxShadow:'inset 0 0 0 1px rgba(34,211,238,0.3)', background:'rgba(34,211,238,0.04)' } : {}}>
                    <td className="px-4 py-3 font-medium text-white">
                      {MODEL_DISPLAY[row.model]??row.model}
                      {isBest && <span className="ml-2 text-xs text-accent">★ best</span>}
                    </td>
                    <td className="px-4 py-3 font-mono text-slate-200">
                      <div className="flex items-center gap-2">
                        <span>{row.rmse.toFixed(3)}</span>
                        <div className="flex-1 h-1.5 rounded bg-white/5 max-w-[60px]">
                          <div className="h-full rounded bg-orange-400/70" style={{width:`${(row.rmse/maxRmse)*100}%`}} />
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-3 font-mono text-slate-200">
                      <div className="flex items-center gap-2">
                        <span>{row.mae.toFixed(3)}</span>
                        <div className="flex-1 h-1.5 rounded bg-white/5 max-w-[60px]">
                          <div className="h-full rounded bg-accent/60" style={{width:`${(row.mae/maxRmse)*100}%`}} />
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-3 font-mono text-slate-200">{row.f1.toFixed(4)}</td>
                    <td className="px-4 py-3 font-mono">
                      {row.skill !== null ? (
                        <span className={row.skill >= 0 ? 'text-green-400' : 'text-red-400'}>
                          {row.skill >= 0 ? '+' : ''}{row.skill.toFixed(4)}
                        </span>
                      ) : <span className="text-slate-500">—</span>}
                    </td>
                  </motion.tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* RMSE/MAE bars */}
      <ChartCard title="RMSE and MAE by Model" subtitle="Lower is better · µg/m³">
        <ReactECharts option={rmseMaeOption} style={{height:200}} theme="dark" opts={{renderer:'canvas'}} />
      </ChartCard>

      {/* Predicted vs actual with play button */}
      <ChartCard
        title="Predicted vs Actual — Test Period"
        subtitle="AQI bands in background · dataZoom · animated draw"
      >
        <div className="flex justify-end mb-2">
          <button
            onClick={()=>{ setPlayIdx(0); setPlaying(true); }}
            disabled={playing}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg glass text-xs text-accent border border-accent/30 hover:bg-accent/10 transition-all disabled:opacity-50">
            <span>{playing ? '▶ Playing…' : '▶ Replay'}</span>
          </button>
        </div>
        <ReactECharts option={predOption} style={{height:260}} theme="dark" opts={{renderer:'canvas'}} notMerge />
      </ChartCard>

      {/* Row: Confusion matrix + Feature importance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <ChartCard title="Confusion Matrix — Random Forest" subtitle="Row-normalised % · cyan intensity = correct">
          <ReactECharts option={cmOption} style={{height:300}} theme="dark" opts={{renderer:'canvas'}} />
        </ChartCard>

        <ChartCard title="Feature Importances" subtitle="">
          <div className="flex gap-1 mb-3">
            {(['rf','ridge'] as const).map(t=>(
              <button key={t} onClick={()=>setTab(t)}
                className={`px-3 py-1 text-xs rounded-lg border transition-all
                  ${tab===t ? 'bg-accent/20 border-accent/40 text-accent' : 'border-white/10 text-slate-400 hover:border-white/20'}`}>
                {t==='rf' ? 'Random Forest' : 'Ridge Coefficients'}
              </button>
            ))}
          </div>
          <ReactECharts option={importanceOption} style={{height:260}} theme="dark" opts={{renderer:'canvas'}} />
        </ChartCard>
      </div>

      {/* Seed spread */}
      <ChartCard
        title="Random Forest — 10-Seed Stability"
        subtitle="Bars = mean · orange error bars = ±1 std · seeds 42–51"
      >
        <ReactECharts option={seedOption} style={{height:200}} theme="dark" opts={{renderer:'canvas'}} />
        <p className="text-xs text-slate-500 mt-2 text-center italic">
          Random Forest repeated over 10 seeds (42–51), mean ± std.
          RMSE: {models.data?.seed_stability.rmse_mean.toFixed(3)} ± {models.data?.seed_stability.rmse_std.toFixed(3)} µg/m³
        </p>
      </ChartCard>
    </div>
  );
}
