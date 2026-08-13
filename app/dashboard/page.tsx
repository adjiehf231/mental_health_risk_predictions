'use client';

import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { BarChart3, Database, Layers, Activity, Users } from 'lucide-react';
import { useApp } from '@/lib/AppContext';

export default function DashboardPage() {
  const { t } = useApp();

  const riskDistribution = [
    { name: 'Low Risk (0)', value: 11250, color: '#10b981', percent: '45.0%' },
    { name: 'Moderate Risk (1)', value: 8750, color: '#f59e0b', percent: '35.0%' },
    { name: 'High Risk (2)', value: 5000, color: '#ef4444', percent: '20.0%' },
  ];

  const ageStressTrends = [
    { ageGroup: '18-25', avgStress: 6.8, avgSleep: 6.2, avgAnxiety: 6.4 },
    { ageGroup: '26-35', avgStress: 7.2, avgSleep: 6.5, avgAnxiety: 5.9 },
    { ageGroup: '36-45', avgStress: 6.1, avgSleep: 7.0, avgAnxiety: 5.1 },
    { ageGroup: '46-55', avgStress: 5.4, avgSleep: 7.2, avgAnxiety: 4.5 },
    { ageGroup: '56-65+', avgStress: 4.2, avgSleep: 7.8, avgAnxiety: 3.8 },
  ];

  const educationDistribution = [
    { level: 'High School', count: 6250, highRiskPct: 24.5 },
    { level: 'Bachelor', count: 11250, highRiskPct: 18.2 },
    { level: 'Master', count: 5000, highRiskPct: 16.4 },
    { level: 'PhD', count: 2500, highRiskPct: 12.1 },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
      
      {/* Header */}
      <div className="text-center space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-cyan-500/10 text-cyan-600 dark:text-cyan-300 border border-cyan-500/20">
          <BarChart3 className="w-4 h-4 text-cyan-500 dark:text-cyan-400" />
          <span>{t.dashboard.badge}</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-adaptive-white">
          📊 {t.dashboard.title}
        </h1>
        <p className="text-adaptive-muted max-w-2xl mx-auto text-sm sm:text-base">
          {t.dashboard.subtitle}
        </p>
      </div>

      {/* Stats Counter Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: t.dashboard.stats.records, val: '25,000', sub: 'Clinical Observations', icon: Database },
          { label: t.dashboard.stats.features, val: '15 Top', sub: 'f_classif Feature Scoring', icon: Layers },
          { label: t.dashboard.stats.distribution, val: '45% / 35% / 20%', sub: 'Low / Mod / High', icon: Activity },
          { label: t.dashboard.stats.demographics, val: '4 Levels', sub: 'Edu & Gender Strata', icon: Users },
        ].map((stat, i) => {
          const Icon = stat.icon;
          return (
            <div key={i} className="glass-panel p-5 rounded-2xl border border-slate-200/40 dark:border-white/10 space-y-1">
              <div className="flex items-center justify-between text-cyan-600 dark:text-cyan-400 mb-2">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-adaptive-muted">{stat.label}</span>
                <Icon className="w-4 h-4" />
              </div>
              <p className="text-2xl sm:text-3xl font-extrabold text-adaptive-white">{stat.val}</p>
              <p className="text-xs text-cyan-600 dark:text-cyan-300">{stat.sub}</p>
            </div>
          );
        })}
      </div>

      {/* Chart Grid Section 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Risk Level Distribution */}
        <div className="lg:col-span-5 glass-panel p-6 rounded-3xl border border-slate-200/40 dark:border-white/10 space-y-4">
          <div className="border-b border-slate-200/20 dark:border-white/10 pb-3">
            <h3 className="text-lg font-bold text-adaptive-white">{t.dashboard.charts.riskTitle}</h3>
            <p className="text-xs text-adaptive-muted">{t.dashboard.charts.riskSub}</p>
          </div>

          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={riskDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={85}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {riskDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px', color: '#fff' }}
                  formatter={(val: any) => [`${val.toLocaleString()} records`, 'Count']}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-3 gap-2 text-center text-xs">
            {riskDistribution.map((item, i) => (
              <div key={i} className="p-2.5 rounded-xl bg-black/5 dark:bg-white/5 border border-slate-200/40 dark:border-white/5 space-y-1">
                <span className="w-2 h-2 rounded-full inline-block mr-1" style={{ backgroundColor: item.color }} />
                <span className="text-adaptive-muted font-medium block text-[11px]">{item.name.split(' ')[0]}</span>
                <p className="font-bold text-adaptive-white">{item.percent}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Age Group vs Stress & Sleep Trends */}
        <div className="lg:col-span-7 glass-panel p-6 rounded-3xl border border-slate-200/40 dark:border-white/10 space-y-4">
          <div className="border-b border-slate-200/20 dark:border-white/10 pb-3">
            <h3 className="text-lg font-bold text-adaptive-white">{t.dashboard.charts.ageTitle}</h3>
            <p className="text-xs text-adaptive-muted">{t.dashboard.charts.ageSub}</p>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={ageStressTrends} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorStress" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6366f1" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="colorSleep" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="ageGroup" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} domain={[0, 10]} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px', color: '#fff' }} />
                <Area type="monotone" dataKey="avgStress" name="Avg Work Stress (0-10)" stroke="#6366f1" fillOpacity={1} fill="url(#colorStress)" />
                <Area type="monotone" dataKey="avgSleep" name="Avg Sleep Hours" stroke="#10b981" fillOpacity={1} fill="url(#colorSleep)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>

      {/* Chart Grid Section 2: Education Level Risk */}
      <div className="glass-panel p-6 rounded-3xl border border-slate-200/40 dark:border-white/10 space-y-4">
        <div className="border-b border-slate-200/20 dark:border-white/10 pb-3">
          <h3 className="text-lg font-bold text-adaptive-white">{t.dashboard.charts.eduTitle}</h3>
          <p className="text-xs text-adaptive-muted">{t.dashboard.charts.eduSub}</p>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={educationDistribution} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <XAxis dataKey="level" stroke="#94a3b8" fontSize={12} />
              <YAxis stroke="#94a3b8" fontSize={12} unit="%" />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px', color: '#fff' }} />
              <Bar dataKey="highRiskPct" name="High Risk Prevalence (%)" fill="#818cf8" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
}
