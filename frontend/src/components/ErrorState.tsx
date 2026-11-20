import { AlertCircle } from 'lucide-react';
interface ErrorStateProps { message?: string; onRetry?: () => void; }
export function ErrorState({ message = 'Failed to load data', onRetry }: ErrorStateProps) {
  return (
    <div className="glass rounded-2xl p-6 flex flex-col items-center gap-3 text-center">
      <AlertCircle className="w-10 h-10 text-red-400" />
      <p className="text-sm text-slate-400">{message}</p>
      {onRetry && (
        <button onClick={onRetry} className="px-4 py-1.5 rounded-lg bg-accent/20 text-accent text-sm hover:bg-accent/30 transition-colors">
          Retry
        </button>
      )}
    </div>
  );
}