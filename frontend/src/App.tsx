import { lazy, Suspense, useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { motion, AnimatePresence } from 'framer-motion';
import { Sidebar }        from './components/Sidebar';
import { TopBar }         from './components/TopBar';
import { LoadingScreen }  from './components/LoadingScreen';
import './index.css';

// Code-split every route ─────────────────────────────────────
const Dashboard = lazy(() => import('./pages/Dashboard').then(m=>({default: m.Dashboard})));
const Forecast  = lazy(() => import('./pages/Forecast').then(m=>({default: m.Forecast})));
const EDA       = lazy(() => import('./pages/EDA').then(m=>({default: m.EDA})));
const Models    = lazy(() => import('./pages/Models').then(m=>({default: m.Models})));
const Kalman    = lazy(() => import('./pages/Kalman').then(m=>({default: m.Kalman})));
const About     = lazy(() => import('./pages/About').then(m=>({default: m.About})));
const NotFound  = lazy(() => import('./pages/NotFound').then(m=>({default: m.NotFound})));

const qc = new QueryClient({ defaultOptions: { queries: { retry: 1 } } });

const PAGE_TITLES: Record<string, string> = {
  '/':         'Dashboard | Lahore AQI',
  '/forecast': 'Forecast | Lahore AQI',
  '/eda':      'EDA | Lahore AQI',
  '/models':   'Models | Lahore AQI',
  '/kalman':   'Kalman Filter | Lahore AQI',
  '/about':    'About | Lahore AQI',
};

function PageTitleUpdater() {
  const loc = useLocation();
  useEffect(() => {
    document.title = PAGE_TITLES[loc.pathname] ?? 'Lahore AQI';
  }, [loc.pathname]);
  return null;
}

const pv = { initial:{ opacity:0 }, animate:{ opacity:1 }, exit:{ opacity:0 } };
const pt = { duration: 0.22 };

function PageFallback() {
  return (
    <div className="flex items-center justify-center h-full" role="status" aria-live="polite">
      <div className="w-8 h-8 rounded-full border-2 border-white/10 border-t-accent animate-spin" />
    </div>
  );
}

function AppRoutes({ isDark, setIsDark }: { isDark:boolean; setIsDark:(v:boolean)=>void }) {
  const loc = useLocation();
  return (
    <>
      <PageTitleUpdater />
      <Sidebar />
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        <TopBar isDark={isDark} onToggle={()=>setIsDark(!isDark)} />
        <main className="flex-1 overflow-hidden" role="main">
          <AnimatePresence mode="wait">
            <Suspense fallback={<PageFallback />}>
              <Routes location={loc} key={loc.pathname}>
                <Route path="/" element={<motion.div key="dash" {...pv} transition={pt} className="h-full"><Dashboard /></motion.div>}/>
                <Route path="/forecast" element={<motion.div key="fc" {...pv} transition={pt} className="h-full"><Forecast /></motion.div>}/>
                <Route path="/eda" element={<motion.div key="eda" {...pv} transition={pt} className="h-full"><EDA /></motion.div>}/>
                <Route path="/models" element={<motion.div key="mod" {...pv} transition={pt} className="h-full"><Models /></motion.div>}/>
                <Route path="/kalman" element={<motion.div key="kal" {...pv} transition={pt} className="h-full"><Kalman /></motion.div>}/>
                <Route path="/about" element={<motion.div key="abt" {...pv} transition={pt} className="h-full"><About /></motion.div>}/>
                <Route path="/404" element={<motion.div key="404" {...pv} transition={pt} className="h-full"><NotFound /></motion.div>}/>
                <Route path="*" element={<Navigate to="/404" replace />} />
              </Routes>
            </Suspense>
          </AnimatePresence>
        </main>
      </div>
    </>
  );
}

export default function App() {
  const [isDark, setIsDark] = useState(true);
  const [ready,  setReady]  = useState(false);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDark);
  }, [isDark]);

  // Brief branded loading screen on first mount
  useEffect(() => {
    const t = setTimeout(() => setReady(true), 800);
    return () => clearTimeout(t);
  }, []);

  if (!ready) return <LoadingScreen />;

  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <div className={`flex h-screen overflow-hidden font-sans ${isDark ? 'bg-bg text-slate-200' : 'bg-bg-light text-slate-800'}`}>
          <AppRoutes isDark={isDark} setIsDark={setIsDark} />
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
