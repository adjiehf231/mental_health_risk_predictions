'use client';

import Link from 'next/link';
import {
  Sparkles,
  BarChart3,
  Cpu,
  History,
  ShieldCheck,
  Zap,
  ArrowRight,
  Activity,
  CheckCircle2,
} from 'lucide-react';
import { useApp } from '@/lib/AppContext';

export default function HomePage() {
  const { t } = useApp();

  const features = [
    {
      title: t.home.modPredictor.title,
      desc: t.home.modPredictor.desc,
      icon: Sparkles,
      href: '/prediction',
      badge: t.home.modPredictor.badge,
      color: 'from-indigo-500 to-purple-600',
      borderGlow: 'hover:border-indigo-500/50',
    },
    {
      title: t.home.modDashboard.title,
      desc: t.home.modDashboard.desc,
      icon: BarChart3,
      href: '/dashboard',
      badge: t.home.modDashboard.badge,
      color: 'from-blue-500 to-cyan-500',
      borderGlow: 'hover:border-cyan-500/50',
    },
    {
      title: t.home.modModels.title,
      desc: t.home.modModels.desc,
      icon: Cpu,
      href: '/models',
      badge: t.home.modModels.badge,
      color: 'from-emerald-500 to-teal-500',
      borderGlow: 'hover:border-emerald-500/50',
    },
    {
      title: t.home.modHistory.title,
      desc: t.home.modHistory.desc,
      icon: History,
      href: '/history',
      badge: t.home.modHistory.badge,
      color: 'from-amber-500 to-orange-500',
      borderGlow: 'hover:border-amber-500/50',
    },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 sm:py-16 space-y-16 sm:space-y-20">
      
      {/* Hero Section */}
      <section className="relative text-center space-y-8 pt-4 sm:pt-8">
        
        {/* Ambient Glow Aura */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[30rem] h-[30rem] bg-gradient-to-tr from-indigo-500/20 via-purple-500/20 to-pink-500/10 rounded-full blur-[100px] -z-10 pointer-events-none" />

        <h1 className="text-4xl sm:text-6xl lg:text-7xl font-black tracking-tight text-adaptive-white max-w-4xl mx-auto leading-[1.15]">
          {t.home.title}
        </h1>

        <p className="text-adaptive-muted text-base sm:text-xl max-w-2xl mx-auto font-normal leading-relaxed">
          {t.home.subtitle}
        </p>

        {/* Hero Action Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-2">
          <Link
            href="/prediction"
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2.5 px-7 py-4 rounded-2xl font-bold text-white bg-gradient-to-r from-indigo-600 via-purple-600 to-indigo-700 hover:brightness-110 shadow-xl shadow-indigo-500/25 transition-all duration-300 transform hover:-translate-y-0.5"
          >
            <Sparkles className="w-5 h-5 text-indigo-200" />
            <span>{t.home.ctaStart}</span>
            <ArrowRight className="w-4 h-4" />
          </Link>

          <Link
            href="/dashboard"
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2.5 px-7 py-4 rounded-2xl font-bold text-adaptive-white glass-panel glass-panel-hover border border-slate-200/80 dark:border-white/10"
          >
            <BarChart3 className="w-5 h-5 text-indigo-500 dark:text-indigo-400" />
            <span>{t.home.ctaEda}</span>
          </Link>
        </div>

      </section>

      {/* High-Impact Key Metrics Banner */}
      <section className="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
        {[
          { label: t.home.stats.accuracy, val: '99.5%', sub: t.home.stats.accSub, icon: CheckCircle2, accent: 'text-emerald-500' },
          { label: t.home.stats.dataset, val: '25,000+', sub: t.home.stats.datasetSub, icon: Activity, accent: 'text-indigo-500' },
          { label: t.home.stats.features, val: '15 Top', sub: t.home.stats.featuresSub, icon: Zap, accent: 'text-amber-500' },
          { label: t.home.stats.latency, val: '< 250ms', sub: t.home.stats.latencySub, icon: ShieldCheck, accent: 'text-purple-500' },
        ].map((stat, i) => {
          const Icon = stat.icon;
          return (
            <div key={i} className="glass-panel p-6 rounded-3xl space-y-2 relative overflow-hidden group">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-bold uppercase tracking-wider text-adaptive-muted">{stat.label}</span>
                <Icon className={`w-5 h-5 ${stat.accent} transition-transform group-hover:scale-110`} />
              </div>
              <p className="text-3xl sm:text-4xl font-black text-adaptive-white tracking-tight">{stat.val}</p>
              <p className="text-xs font-semibold text-indigo-600 dark:text-indigo-300">{stat.sub}</p>
            </div>
          );
        })}
      </section>

      {/* Premium Platform Module Cards */}
      <section className="space-y-8">
        <div className="text-center space-y-2">
          <h2 className="text-3xl sm:text-4xl font-extrabold text-adaptive-white">{t.home.modulesTitle}</h2>
          <p className="text-adaptive-muted text-sm sm:text-base max-w-xl mx-auto">{t.home.modulesSub}</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 sm:gap-8">
          {features.map((feat, i) => {
            const Icon = feat.icon;
            return (
              <Link
                key={i}
                href={feat.href}
                className={`group glass-panel p-8 rounded-3xl glass-panel-hover flex flex-col justify-between space-y-6 ${feat.borderGlow}`}
              >
                <div className="space-y-5">
                  <div className="flex items-center justify-between">
                    <div className={`w-14 h-14 rounded-2xl bg-gradient-to-tr ${feat.color} flex items-center justify-center shadow-lg shadow-indigo-500/20 group-hover:scale-105 transition-transform duration-300`}>
                      <Icon className="w-7 h-7 text-white" />
                    </div>
                    <span className="text-xs font-bold px-3 py-1.5 rounded-full bg-slate-200/80 dark:bg-white/10 text-slate-800 dark:text-indigo-200 border border-slate-300 dark:border-white/10 shadow-sm">
                      {feat.badge}
                    </span>
                  </div>

                  <div>
                    <h3 className="text-2xl font-extrabold text-adaptive-white group-hover:text-indigo-500 dark:group-hover:text-indigo-300 transition-colors">
                      {feat.title}
                    </h3>
                    <p className="text-adaptive-muted text-sm sm:text-base mt-2 leading-relaxed font-normal">
                      {feat.desc}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2 text-sm font-bold text-indigo-600 dark:text-indigo-400 group-hover:translate-x-1.5 transition-transform pt-2">
                  <span>{t.home.launch}</span>
                  <ArrowRight className="w-4 h-4" />
                </div>
              </Link>
            );
          })}
        </div>
      </section>

    </div>
  );
}
