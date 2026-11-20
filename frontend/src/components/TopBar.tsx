import { Download } from 'lucide-react';
import { ThemeToggle } from './ThemeToggle';
interface TopBarProps { isDark: boolean; onToggle: () => void; }
export function TopBar({ isDark, onToggle }: TopBarProps) {
  async function downloadKpis() {
    const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8000';
    const res  = await fetch(`${BASE}/api/kpis`);
    const data: unknown = await res.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'lahore_kpis.json'; a.click();
    URL.revokeObjectURL(url);
  }
  return (
    <header className="glass border-b border-white/5 h-14 flex items-center justify-between px-6 flex-shrink-0 z-10">
      <div>
        <h1 className="text-sm font-bold text-white">Lahore PM2.5 Dashboard</h1>
        <p className="text-xs text-slate-500">Air Quality Analysis · 2019–2025</p>
      </div>
      <div className="flex items-center gap-2">
        <button onClick={() => { void downloadKpis(); }}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl glass text-xs text-slate-300 hover:text-white hover:bg-white/10 transition-all">
          <Download className="w-3.5 h-3.5" />
          Download KPIs
        </button>
        <ThemeToggle isDark={isDark} onToggle={onToggle} />
      </div>
    </header>
  );
}