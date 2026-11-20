import { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LayoutDashboard, TrendingUp, BarChart3, FlaskConical,
  Filter, Info, ChevronLeft, ChevronRight, Wind,
} from 'lucide-react';

const NAV = [
  { to: '/',         icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/forecast', icon: TrendingUp,       label: 'Forecast'  },
  { to: '/eda',      icon: BarChart3,         label: 'EDA'       },
  { to: '/models',   icon: FlaskConical,      label: 'Models'    },
  { to: '/kalman',   icon: Filter,            label: 'Kalman'    },
  { to: '/about',    icon: Info,              label: 'About'     },
];

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();
  return (
    <motion.aside
      animate={{ width: collapsed ? 64 : 220 }}
      transition={{ type:'spring', stiffness:260, damping:26 }}
      className="glass flex flex-col h-full border-r border-white/5 overflow-hidden relative z-20 flex-shrink-0"
      role="navigation"
      aria-label="Main navigation"
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-5 border-b border-white/5">
        <Wind className="w-7 h-7 text-accent flex-shrink-0" aria-hidden="true" />
        <AnimatePresence>
          {!collapsed && (
            <motion.span initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}
              className="text-sm font-bold text-white whitespace-nowrap">
              Lahore AQI
            </motion.span>
          )}
        </AnimatePresence>
      </div>

      {/* Nav items */}
      <nav className="flex flex-col gap-1 px-2 py-3 flex-1 overflow-y-auto" aria-label="Pages">
        {NAV.map(({ to, icon: Icon, label }) => {
          const active = location.pathname === to;
          return (
            <NavLink key={to} to={to}
              aria-label={label}
              aria-current={active ? 'page' : undefined}
              className={`relative flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors
                outline-none focus-visible:ring-2 focus-visible:ring-accent/50
                ${active ? 'text-white' : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'}`}>
              {active && (
                <motion.div layoutId="active-pill"
                  className="absolute inset-0 rounded-xl"
                  style={{ background:'linear-gradient(135deg,rgba(34,211,238,0.2),rgba(139,92,246,0.15))' }}
                  transition={{ type:'spring', stiffness:300, damping:30 }} />
              )}
              <Icon className="w-4 h-4 flex-shrink-0 relative z-10" aria-hidden="true" />
              <AnimatePresence>
                {!collapsed && (
                  <motion.span initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}
                    className="relative z-10 whitespace-nowrap">{label}</motion.span>
                )}
              </AnimatePresence>
            </NavLink>
          );
        })}
      </nav>

      {/* Collapse toggle */}
      <button
        onClick={() => setCollapsed(c => !c)}
        aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        className="flex items-center justify-center p-3 m-2 rounded-xl hover:bg-white/5 transition-colors text-slate-400 border border-white/5 outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
      >
        {collapsed ? <ChevronRight className="w-4 h-4" aria-hidden="true" /> : <ChevronLeft className="w-4 h-4" aria-hidden="true" />}
      </button>
    </motion.aside>
  );
}
