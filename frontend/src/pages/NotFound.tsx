import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
export function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-5 p-8 text-center">
      <motion.div initial={{scale:0.8,opacity:0}} animate={{scale:1,opacity:1}} transition={{type:'spring',stiffness:200}}>
        <div className="text-7xl mb-2">🌫</div>
        <h1 className="text-5xl font-bold font-mono text-accent mb-2">404</h1>
        <p className="text-white text-lg font-semibold mb-1">Page not found</p>
        <p className="text-slate-400 text-sm mb-6">This page has drifted into the smog.</p>
        <Link to="/" className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-sm text-white transition-all focus:outline-none focus:ring-2 focus:ring-accent/50"
          style={{ background:'linear-gradient(135deg,#22D3EE,#8B5CF6)' }}>
          ← Back to Dashboard
        </Link>
      </motion.div>
    </div>
  );
}
