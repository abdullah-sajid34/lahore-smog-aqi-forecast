import { motion } from 'framer-motion';
import { ExternalLink, AlertTriangle } from 'lucide-react';
import { useAbout } from '../hooks/useApi';
import { Skeleton }   from '../components/Skeleton';
import { ErrorState } from '../components/ErrorState';
import { COLORS } from '../theme';

const TEAM = [
  { name:'Muhammad Abdullah Sajid', id:'24K-3045', role:'Lead Developer & ML Engineer' },
  { name:'Muhammad Hammad Rafi',    id:'24K-3117', role:'Data Engineer & Backend Developer' },
  { name:'Kabeer Danish',           id:'24K-3118', role:'Frontend Developer & Visualisation' },
];

const PEAS_ICONS: Record<string, string> = {
  performance_measure: '📊',
  environment:         '🌍',
  actuators:           '⚡',
  sensors:             '📡',
};
const PEAS_LABELS: Record<string, string> = {
  performance_measure: 'Performance',
  environment:         'Environment',
  actuators:           'Actuators',
  sensors:             'Sensors',
};
const PEAS_COLORS: Record<string, string> = {
  performance_measure: '#22D3EE',
  environment:         '#8B5CF6',
  actuators:           '#FF7E00',
  sensors:             '#00E400',
};

const LIMITATIONS = [
  'Data ends 2025-02-18. Predictions beyond this date are extrapolated.',
  'Sensor data from a single US Embassy monitor — may not represent all Lahore neighbourhoods.',
  '21 data gaps > 3 days exist (largest: 47 days, Oct–Dec 2021) concentrated in peak smog season.',
  'Models predict the next calendar day only; multi-day forecasts use the h=3 bonus model.',
  'Kalman filter improvement is modest (<1 µg/m³ RMSE) — seasonal signal dominates noise.',
  'AQI advisory texts are informational only and not an official health advisory.',
];

export function About() {
  const about = useAbout();

  if (about.isLoading) return (
    <div className="flex flex-col gap-6 p-6">
      {[1,2,3,4].map(i=><Skeleton key={i} className="h-32" />)}
    </div>
  );
  if (about.error) return (
    <div className="p-6"><ErrorState message={about.error.message} onRetry={()=>about.refetch()} /></div>
  );

  const data = about.data!;

  return (
    <div className="flex flex-col gap-7 p-6 overflow-y-auto h-full max-w-4xl mx-auto w-full">

      {/* Title */}
      <motion.div initial={{opacity:0,y:-10}} animate={{opacity:1,y:0}}>
        <h2 className="text-xl font-bold text-white mb-1">{data.title}</h2>
        <p className="text-sm text-slate-400">
          AI Project · FAST NUCES Karachi · AL-2002 Fall 2026
        </p>
      </motion.div>

      {/* PEAS */}
      <div>
        <h3 className="text-base font-semibold text-white mb-3">PEAS Description</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {(Object.keys(PEAS_ICONS) as Array<keyof typeof PEAS_ICONS>).map((key,i) => {
            const val = data.peas[key as keyof typeof data.peas];
            const col = PEAS_COLORS[key];
            return (
              <motion.div key={key}
                initial={{opacity:0,y:16}} animate={{opacity:1,y:0}} transition={{delay:i*0.08}}
                whileHover={{ rotateX:3, rotateY:-3, scale:1.015 }}
                style={{ transformPerspective:800 }}
                className="glass rounded-2xl p-5 relative overflow-hidden"
              >
                <div className="text-2xl mb-2">{PEAS_ICONS[key]}</div>
                <p className="text-xs text-slate-400 uppercase tracking-wide mb-1">{PEAS_LABELS[key]}</p>
                <p className="text-sm text-slate-200 leading-relaxed">{val}</p>
                <div className="absolute bottom-0 left-0 right-0 h-0.5"
                  style={{ background:`linear-gradient(90deg,transparent,${col},transparent)` }} />
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Data sources */}
      <div>
        <h3 className="text-base font-semibold text-white mb-3">Data Sources</h3>
        <div className="flex flex-col gap-3">
          {data.data_sources.map((ds,i) => (
            <motion.div key={ds.name}
              initial={{opacity:0,x:-16}} animate={{opacity:1,x:0}} transition={{delay:i*0.07}}
              className="glass rounded-xl p-4 flex flex-col sm:flex-row sm:items-start gap-3"
            >
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <p className="text-sm font-semibold text-white">{ds.name}</p>
                  <a href={ds.url} target="_blank" rel="noopener noreferrer"
                    aria-label={`Open ${ds.name} in new tab`}
                    className="text-accent hover:text-accent/80 transition-colors focus:outline-none focus:ring-2 focus:ring-accent/50 rounded">
                    <ExternalLink className="w-3.5 h-3.5" />
                  </a>
                </div>
                <p className="text-xs text-slate-400 mb-1">{ds.description}</p>
                <p className="text-xs text-slate-500 italic">{ds.attribution}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Limitations */}
      <div>
        <h3 className="text-base font-semibold text-white mb-3">Known Limitations</h3>
        <div className="glass rounded-2xl p-5">
          <ul className="flex flex-col gap-2.5">
            {LIMITATIONS.map((lim,i) => (
              <motion.li key={i}
                initial={{opacity:0,x:-8}} animate={{opacity:1,x:0}} transition={{delay:i*0.05}}
                className="flex items-start gap-2.5 text-xs text-slate-400"
              >
                <span className="text-orange-400 flex-shrink-0 mt-0.5">•</span>
                {lim}
              </motion.li>
            ))}
          </ul>
        </div>
      </div>

      {/* Disclaimer */}
      <motion.div
        initial={{opacity:0}} animate={{opacity:1}} transition={{delay:0.4}}
        className="flex items-start gap-3 p-4 rounded-2xl border border-amber-500/30 bg-amber-500/06"
      >
        <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
        <p className="text-sm text-amber-300">{data.disclaimer}</p>
      </motion.div>

      {/* Team */}
      <div>
        <h3 className="text-base font-semibold text-white mb-3">Team</h3>
        <p className="text-xs text-slate-400 mb-4">FAST NUCES Karachi · AL-2002 Artificial Intelligence · Fall 2026</p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {TEAM.map((member, i) => (
            <motion.div key={member.id}
              initial={{opacity:0,y:16}} animate={{opacity:1,y:0}} transition={{delay:0.1+i*0.08}}
              whileHover={{ rotateX:4, rotateY:-4, scale:1.02 }}
              style={{ transformPerspective:800 }}
              className="glass rounded-2xl p-5 text-center relative overflow-hidden"
            >
              {/* Avatar circle */}
              <div className="w-12 h-12 rounded-full mx-auto mb-3 flex items-center justify-center text-lg font-bold"
                style={{ background:`linear-gradient(135deg,${COLORS.accent}44,${COLORS.violet}44)`, color: COLORS.accent }}>
                {member.name.split(' ').map(n=>n[0]).slice(0,2).join('')}
              </div>
              <p className="text-sm font-semibold text-white mb-0.5">{member.name}</p>
              <p className="text-xs text-accent font-mono mb-1">{member.id}</p>
              <p className="text-xs text-slate-400">{member.role}</p>
              <div className="absolute bottom-0 left-0 right-0 h-0.5"
                style={{ background:`linear-gradient(90deg,transparent,${COLORS.accent},transparent)` }} />
            </motion.div>
          ))}
        </div>
      </div>

    </div>
  );
}
