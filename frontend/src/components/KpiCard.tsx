import { useRef, useEffect, useState, ReactNode } from 'react';
import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';

interface KpiCardProps {
  title: string; value: number | null; unit?: string;
  subtitle?: string; icon?: ReactNode; color?: string;
  format?: (v: number) => string; delay?: number;
}

function useCountUp(target: number | null, duration = 1200) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    if (target === null) return;
    const finalTarget = target;
    let start: number | null = null;
    function step(ts: number) {
      if (start === null) start = ts;
      const p = Math.min((ts - start) / duration, 1);
      setDisplay(finalTarget * (1 - Math.pow(1 - p, 3)));
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }, [target, duration]);
  return display;
}

export function KpiCard({ title, value, unit = '', subtitle, icon, color = '#22D3EE', format, delay = 0 }: KpiCardProps) {
  const cardRef = useRef<HTMLDivElement>(null);
  const mx = useMotionValue(0); const my = useMotionValue(0);
  const sx = useSpring(mx, { stiffness: 200, damping: 20 });
  const sy = useSpring(my, { stiffness: 200, damping: 20 });
  const rotateX = useTransform(sy, [-0.5, 0.5], [6, -6]);
  const rotateY = useTransform(sx, [-0.5, 0.5], [-6, 6]);

  function onMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    mx.set((e.clientX - rect.left) / rect.width - 0.5);
    my.set((e.clientY - rect.top)  / rect.height - 0.5);
  }

  const displayed = useCountUp(value);
  const formatted = value !== null ? (format ? format(displayed) : displayed.toFixed(value < 10 ? 2 : 1)) : '—';

  return (
    <motion.div
      ref={cardRef}
      initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
      transition={{ delay: delay * 0.1, duration: 0.4 }}
      style={{ rotateX, rotateY, transformPerspective: 800 }}
      onMouseMove={onMouseMove} onMouseLeave={() => { mx.set(0); my.set(0); }}
      className="glass rounded-2xl p-5 relative overflow-hidden cursor-default"
    >
      <div className="flex items-start justify-between mb-3">
        <p className="text-xs text-slate-400 font-medium uppercase tracking-wide">{title}</p>
        {icon && <span className="text-slate-400">{icon}</span>}
      </div>
      <div className="flex items-baseline gap-1 mb-1">
        <span className="text-3xl font-bold font-mono" style={{ color }}>{formatted}</span>
        {unit && <span className="text-sm text-slate-400">{unit}</span>}
      </div>
      {subtitle && <p className="text-xs text-slate-500">{subtitle}</p>}
      <div className="absolute bottom-0 left-0 right-0 h-0.5 rounded-b-2xl"
        style={{ background: `linear-gradient(90deg,transparent,${color},transparent)` }} />
    </motion.div>
  );
}