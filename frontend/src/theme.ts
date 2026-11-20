export const COLORS = {
  bg: '#0B1220',
  card: '#111A2E',
  cardLight: '#F0F4FF',
  bgLight: '#F7F9FC',
  accent: '#22D3EE',
  violet: '#8B5CF6',
  border: 'rgba(34,211,238,0.12)',
  text: '#E2E8F0',
  textMuted: '#94A3B8',
  aqi: {
    Good:                     '#00E400',
    Moderate:                 '#FFD60A',
    'Unhealthy for Sensitive':'#FF7E00',
    Unhealthy:                '#FF2D2D',
    'Very Unhealthy':         '#8F3F97',
    Hazardous:                '#7E0023',
    'Hazardous+':             '#7E0023',
    Unknown:                  '#64748B',
  } as Record<string, string>,
} as const;

export function aqiColor(category: string): string {
  return COLORS.aqi[category] ?? COLORS.aqi['Unknown'];
}