interface SkeletonProps { className?: string; }
export function Skeleton({ className = '' }: SkeletonProps) {
  return <div className={`shimmer rounded-xl bg-white/5 ${className}`} />;
}