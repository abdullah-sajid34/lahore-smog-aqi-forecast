import { useEffect, useState } from 'react';
import { AQIBadge } from './AQIBadge';
import { aqiColor } from '../theme';

interface GaugeProps { value: number | null; category: string; }

const MAX_VAL = 500;
const SA = -220; const EA = 40; const ARC = EA - SA;
// SVG circle circumference for r=110: 2*pi*110 ~ 691
const CIRC = 2 * Math.PI * 110;

function valToAngle(v: number) { return SA + (Math.min(v, MAX_VAL) / MAX_VAL) * ARC; }
function polar(cx: number, cy: number, r: number, deg: number) {
  const rad = (deg * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}
function arcPath(cx: number, cy: number, r: number, s: number, e: number) {
  const sp = polar(cx, cy, r, s); const ep = polar(cx, cy, r, e);
  const large = e - s > 180 ? 1 : 0;
  return `M ${sp.x} ${sp.y} A ${r} ${r} 0 ${large} 1 ${ep.x} ${ep.y}`;
}

function useCountUp(target: number | null, dur = 1500) {
  const [v, setV] = useState(0);
  useEffect(() => {
    if (target === null) return;
    const finalTarget = target;
    let s: number | null = null;
    function step(ts: number) {
      if (s === null) s = ts;
      const p = Math.min((ts - s) / dur, 1);
      setV(finalTarget * (1 - Math.pow(1 - p, 3)));
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }, [target, dur]);
  return v;
}

// Arc progress length in SVG units
// The arc spans 260/360 of the full circle = 260/360 * 2*pi*r
const TRACK_LEN = (ARC / 360) * CIRC;

export function Gauge({ value, category }: GaugeProps) {
  const cx = 150; const cy = 150; const r = 110;
  const color = aqiColor(category);
  const display = useCountUp(value);
  const prog   = value !== null ? Math.min(value, MAX_VAL) / MAX_VAL : 0;
  const fillLen = prog * TRACK_LEN;
  const nAngle = valToAngle(value ?? 0);
  const npt    = polar(cx, cy, 90, nAngle);
  const who    = polar(cx, cy, r + 5, valToAngle(15));
  const naaqs  = polar(cx, cy, r + 5, valToAngle(35));

  return (
    <div className="flex flex-col items-center gap-3">
      <svg width="300" height="220" viewBox="0 0 300 220">
        {/* Track */}
        <path d={arcPath(cx, cy, r, SA, EA)} fill="none"
          stroke="rgba(255,255,255,0.08)" strokeWidth="14" strokeLinecap="round" />
        {/* Filled arc — animate via strokeDasharray transition */}
        <path d={arcPath(cx, cy, r, SA, EA)} fill="none"
          stroke={color} strokeWidth="14" strokeLinecap="round"
          strokeDasharray={`${fillLen} ${TRACK_LEN}`}
          strokeDashoffset={0}
          style={{
            filter: `drop-shadow(0 0 8px ${color}88)`,
            transition: 'stroke-dasharray 1.5s cubic-bezier(0.16,1,0.3,1)',
          }} />
        {/* Needle */}
        <line x1={cx} y1={cy} x2={npt.x} y2={npt.y}
          stroke={color} strokeWidth="2.5" strokeLinecap="round" />
        <circle cx={cx} cy={cy} r="6" fill={color} />
        {/* WHO 15 marker */}
        <circle cx={who.x} cy={who.y} r="4" fill="#00E400" />
        <text x={who.x} y={who.y - 8} textAnchor="middle" fill="#00E400" fontSize="9">15</text>
        {/* NAAQS 35 marker */}
        <circle cx={naaqs.x} cy={naaqs.y} r="4" fill="#FFD60A" />
        <text x={naaqs.x} y={naaqs.y - 8} textAnchor="middle" fill="#FFD60A" fontSize="9">35</text>
        {/* Value */}
        <text x={cx} y={cy + 30} textAnchor="middle" fill="white"
          fontSize="38" fontFamily="'JetBrains Mono',monospace" fontWeight="700">
          {Math.round(display)}
        </text>
        <text x={cx} y={cy + 50} textAnchor="middle" fill="rgba(255,255,255,0.5)" fontSize="11">
          µg/m³
        </text>
        <text x="30"  y={cy + 30} fill="rgba(255,255,255,0.4)" fontSize="9">0</text>
        <text x="260" y={cy + 30} fill="rgba(255,255,255,0.4)" fontSize="9">500</text>
      </svg>
      <AQIBadge category={category} pulse size="lg" />
      <p className="text-xs text-slate-500">WHO: 15 · NAAQS: 35 µg/m³</p>
    </div>
  );
}
