import { aqiColor } from '../theme';
interface AQIBadgeProps { category: string; pulse?: boolean; size?: 'sm' | 'md' | 'lg'; }
export function AQIBadge({ category, pulse = false, size = 'md' }: AQIBadgeProps) {
  const color = aqiColor(category);
  const sizes = { sm: 'text-xs px-2 py-0.5', md: 'text-sm px-3 py-1', lg: 'text-base px-4 py-1.5' };
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-semibold font-sans ${sizes[size]} ${pulse ? 'animate-pulse2' : ''}`}
      style={{ background: `${color}22`, color, border: `1px solid ${color}44` }}
    >
      <span className="w-2 h-2 rounded-full inline-block" style={{ background: color }} />
      {category}
    </span>
  );
}