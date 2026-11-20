import { ReactNode } from 'react';
interface ChartCardProps { title: string; subtitle?: string; children: ReactNode; className?: string; }
export function ChartCard({ title, subtitle, children, className = '' }: ChartCardProps) {
  return (
    <div className={`glass rounded-2xl p-5 ${className}`}>
      <div className="mb-4">
        <h3 className="text-base font-semibold text-slate-200">{title}</h3>
        {subtitle && <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>}
      </div>
      {children}
    </div>
  );
}