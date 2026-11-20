import { motion } from 'framer-motion';
export function LoadingScreen() {
  return (
    <div className="fixed inset-0 bg-bg flex flex-col items-center justify-center gap-6 z-50"
      role="status" aria-label="Loading dashboard">
      <motion.div
        animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 2, ease: 'linear' }}
        className="w-16 h-16 rounded-full border-2 border-white/10 border-t-accent"
      />
      <div className="text-center">
        <p className="text-white font-semibold text-lg">Lahore Air Quality</p>
        <p className="text-slate-400 text-sm mt-1">Loading dashboard…</p>
      </div>
    </div>
  );
}
