import { useQuery } from '@tanstack/react-query';
import type { KpisResponse, TodayResponse, SeriesResponse, ForecastResponse, EdaResponse, ModelsResponse } from '../types/api';

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8000';

async function apiFetch<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText })) as { detail?: string };
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function useKpis() {
  return useQuery<KpisResponse>({ queryKey: ['kpis'], queryFn: () => apiFetch<KpisResponse>('/api/kpis'), staleTime: 300000 });
}
export function useToday() {
  return useQuery<TodayResponse>({ queryKey: ['today'], queryFn: () => apiFetch<TodayResponse>('/api/today'), staleTime: 300000 });
}
export function useSeries(start?: string, end?: string) {
  const params = new URLSearchParams();
  if (start) params.set('start', start);
  if (end)   params.set('end',   end);
  const qs = params.toString() ? `?${params.toString()}` : '';
  return useQuery<SeriesResponse>({ queryKey: ['series', start, end], queryFn: () => apiFetch<SeriesResponse>(`/api/series${qs}`), staleTime: 300000 });
}
export function useForecast(date: string | null, model: string) {
  return useQuery<ForecastResponse>({
    queryKey: ['forecast', date, model],
    queryFn:  () => apiFetch<ForecastResponse>(`/api/forecast?date=${date}&model=${model}`),
    enabled:  !!date, staleTime: 300000, retry: false,
  });
}
export function useEda() {
  return useQuery<EdaResponse>({ queryKey: ['eda'], queryFn: () => apiFetch<EdaResponse>('/api/eda'), staleTime: 600000 });
}
export function useModels() {
  return useQuery<ModelsResponse>({ queryKey: ['models'], queryFn: () => apiFetch<ModelsResponse>('/api/models'), staleTime: 600000 });
}
export function useRidgeCoefficients() {
  const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8000';
  return useQuery<{ alpha: number; coefficients_scaled: Record<string, number> }>({
    queryKey: ['ridgeCoef'],
    queryFn:  () => apiFetch(`/api/models`).then((d: unknown) => {
      // We fetch from /api/models and re-read ridge coef from backend service
      void d;
      return fetch(`${BASE}/api/models`).then(r => r.json()).then((m: {
        feature_importance: { importances: Record<string,number>; model: string };
      }) => ({ alpha: 0.01, coefficients_scaled: m.feature_importance.importances }));
    }),
    staleTime: 600000,
  });
}

export interface KalmanPoint { date: string; raw: number; filtered: number; variance: number; }
export interface KalmanApiResponse {
  Q: number; R: number; q_frac: number;
  series: KalmanPoint[];
  comparison: Array<{ model:string; input:string; rmse:number; rmse_std:number; mae:number; mae_std:number; f1:number; f1_std:number; skill:number|null; }>;
}
export function useKalman() {
  return useQuery<KalmanApiResponse>({ queryKey: ['kalman'], queryFn: () => apiFetch<KalmanApiResponse>('/api/kalman'), staleTime: 600000 });
}
export interface AboutApiResponse {
  title:string; peas:{ performance_measure:string; environment:string; actuators:string; sensors:string; };
  data_sources:Array<{ name:string; url:string; description:string; attribution:string; }>;
  team:Array<{ name:string; role:string; }>; disclaimer:string;
}
export function useAbout() {
  return useQuery<AboutApiResponse>({ queryKey: ['about'], queryFn: () => apiFetch<AboutApiResponse>('/api/about'), staleTime: 3600000 });
}
