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
  BrainCircuit,
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
      color: 'from-indigo-500 to-purple-500',
    },
    {
      title: t.home.modDashboard.title,
      desc: t.home.modDashboard.desc,
      icon: BarChart3,
      href: '/dashboard',
      badge: t.home.modDashboard.badge,
      color: 'from-blue-500 to-cyan-500',
    },
    {
      title: t.home.modModels.title,
      desc: t.home.modModels.desc,
      icon: Cpu,
      href: '/models',
      badge: t.home.modModels.badge,
      color: 'from-emerald-500 to-teal-500',
    },
    {
      title: t.home.modHistory.title,
      desc: t.home.modHistory.desc,
      icon: History,
      href: '/history',
      badge: t.home.modHistory.badge,
      color: 'from-amber-500 to-orange-500',
    },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-16">
      
      {/* Hero Section */}
      <section className="relative text-center space-y-6 pt-6">
        
        {/* Ambient Glow */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-indigo-500/15 dark:bg-indigo-500/20 rounded-full blur-3xl -z-10 pointer-events-none" />

        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-semibold bg-indigo-500/10 text-indigo-600 dark:text-indigo-300 border border-indigo-500/20 shadow-sm">
          <BrainCircuit className="w-4 h-4 text-indigo-500 dark:text-indigo-400 animate-pulse" />
          <span>{t.home.badge}</span>
        </div>

        <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight text-adaptive-white max-w-4xl mx-auto leading-tight">
          {t.home.title}
        </h1>

        <p className="text-adaptive-muted text-base sm:text-lg max-w-2xl mx-auto font-normal leading-relaxed">
          {t.home.subtitle}
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
          <Link
            href="/prediction"
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl font-semibold text-white bg-gradient-to-r from-indigo-500 via-purple-600 to-indigo-700 hover:brightness-110 shadow-lg shadow-indigo-500/25 transition-all duration-200"
          >
            <Sparkles className="w-5 h-5" />
            <span>{t.home.ctaStart}</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
          <Link
            href="/dashboard"
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl font-semibold text-adaptive-white glass-panel glass-panel-hover border border-slate-200/40 dark:border-white/10"
          >
            <BarChart3 className="w-5 h-5 text-indigo-500 dark:text-indigo-400" />
            <span>{t.home.ctaEda}</span>
          </Link>
        </div>

      </section>

      {/* Metrics Banner */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: t.home.stats.accuracy, val: '99.5%', sub: t.home.stats.accSub, icon: CheckCircle2 },
          { label: t.home.stats.dataset, val: '25,000+', sub: t.home.stats.datasetSub, icon: Activity },
          { label: t.home.stats.features, val: '15 Top', sub: t.home.stats.featuresSub, icon: Zap },
          { label: t.home.stats.latency, val: '< 250ms', sub: t.home.stats.latencySub, icon: ShieldCheck },
        ].map((stat, i) => {
          const Icon = stat.icon;
          return (
            <div key={i} className="glass-panel p-5 rounded-2xl border border-slate-200/40 dark:border-white/10 space-y-1">
              <div className="flex items-center justify-between text-indigo-500 dark:text-indigo-400 mb-2">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-adaptive-muted">{stat.label}</span>
                <Icon className="w-4 h-4" />
              </div>
              <p className="text-2xl sm:text-3xl font-extrabold text-adaptive-white">{stat.val}</p>
              <p className="text-xs text-indigo-600 dark:text-indigo-300">{stat.sub}</p>
            </div>
          );
        })}
      </section>

      {/* Module Navigation Grid */}
      <section className="space-y-6">
        <div className="text-center space-y-2">
          <h2 className="text-2xl sm:text-3xl font-bold text-adaptive-white">{t.home.modulesTitle}</h2>
          <p className="text-adaptive-muted text-sm">{t.home.modulesSub}</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {features.map((feat, i) => {
            const Icon = feat.icon;
            return (
              <Link
                key={i}
                href={feat.href}
                className="group glass-panel p-6 rounded-2xl border border-slate-200/40 dark:border-white/10 glass-panel-hover flex flex-col justify-between"
              >
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className={`w-12 h-12 rounded-xl bg-gradient-to-tr ${feat.color} flex items-center justify-center shadow-lg`}>
                      <Icon className="w-6 h-6 text-white" />
                    </div>
                    <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-slate-200/60 dark:bg-white/10 text-slate-800 dark:text-indigo-200 border border-slate-300 dark:border-white/10">
                      {feat.badge}
                    </span>
                  </div>

                  <div>
                    <h3 className="text-xl font-bold text-adaptive-white group-hover:text-indigo-500 dark:group-hover:text-indigo-300 transition-colors">
                      {feat.title}
                    </h3>
                    <p className="text-adaptive-muted text-sm mt-1 leading-relaxed">
                      {feat.desc}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-1 text-sm font-semibold text-indigo-500 dark:text-indigo-400 mt-6 group-hover:translate-x-1 transition-transform">
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
