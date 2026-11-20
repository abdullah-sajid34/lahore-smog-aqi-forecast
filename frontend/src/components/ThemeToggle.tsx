import { Sun, Moon } from 'lucide-react';
interface ThemeToggleProps { isDark: boolean; onToggle: () => void; }
export function ThemeToggle({ isDark, onToggle }: ThemeToggleProps) {
  return (
    <button onClick={onToggle} className="p-2 rounded-xl glass hover:bg-white/10 transition-all duration-200" title={isDark ? 'Light mode' : 'Dark mode'}>
      {isDark ? <Sun className="w-4 h-4 text-accent" /> : <Moon className="w-4 h-4 text-violet-400" />}
    </button>
  );
}