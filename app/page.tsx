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

export default function HomePage() {
  const features = [
    {
      title: 'AI Risk Predictor',
      desc: 'Evaluate patient risk scores across 15 lifestyle and psychological parameters in real-time.',
      icon: Sparkles,
      href: '/prediction',
      badge: '99.5% Decision Tree Acc',
      color: 'from-indigo-500 to-purple-500',
    },
    {
      title: 'EDA Analytics Dashboard',
      desc: 'Explore dataset insights across 25,000+ records, age trends, and stress correlation heatmaps.',
      icon: BarChart3,
      href: '/dashboard',
      badge: 'Interactive Visuals',
      color: 'from-blue-500 to-cyan-500',
    },
    {
      title: 'ML Benchmarks',
      desc: 'Inspect accuracy, F1-scores, and recall metrics across 5 algorithms (DT, RF, SVM, KNN, NB).',
      icon: Cpu,
      href: '/models',
      badge: '5 ML Models',
      color: 'from-emerald-500 to-teal-500',
    },
    {
      title: 'Supabase Assessment Log',
      desc: 'Real-time database tracking of risk predictions, patient profiles, and CSV report exports.',
      icon: History,
      href: '/history',
      badge: 'PostgreSQL Sync',
      color: 'from-amber-500 to-orange-500',
    },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-16">
      
      {/* Hero Section */}
      <section className="relative text-center space-y-6 pt-6">
        
        {/* Glow backdrop */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-indigo-500/20 rounded-full blur-3xl -z-10 pointer-events-none" />

        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-semibold bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 shadow-lg shadow-indigo-500/10">
          <BrainCircuit className="w-4 h-4 text-indigo-400 animate-pulse" />
          <span>Next.js 14/15 + Vercel Python Serverless + Supabase</span>
        </div>

        <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight text-white max-w-4xl mx-auto leading-tight">
          Intelligent <span className="bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-purple-300 to-pink-400">Mental Health Risk</span> Analytics Platform
        </h1>

        <p className="text-slate-300 text-lg sm:text-xl max-w-2xl mx-auto font-normal leading-relaxed">
          Empowering individuals and counselors with high-accuracy machine learning predictions, interactive EDA charts, and real-time clinical assessment logging.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
          <Link
            href="/prediction"
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl font-semibold text-white bg-gradient-to-r from-indigo-500 via-purple-600 to-indigo-700 hover:brightness-110 shadow-lg shadow-indigo-500/30 transition-all duration-200"
          >
            <Sparkles className="w-5 h-5" />
            <span>Start Risk Assessment</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
          <Link
            href="/dashboard"
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl font-semibold text-slate-200 glass-panel glass-panel-hover border border-white/10"
          >
            <BarChart3 className="w-5 h-5 text-indigo-400" />
            <span>Explore Dataset EDA</span>
          </Link>
        </div>

      </section>

      {/* Metrics Banner */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Decision Tree Accuracy', val: '99.5%', sub: 'F1 Score: 99.3%', icon: CheckCircle2 },
          { label: 'Analyzed Dataset', val: '25,000+', sub: 'Kaggle Clinical Dataset', icon: Activity },
          { label: 'Selected Features', val: '15 Top', sub: 'SelectKBest F-Classif', icon: Zap },
          { label: 'Response Latency', val: '< 250ms', sub: 'Serverless Python API', icon: ShieldCheck },
        ].map((stat, i) => {
          const Icon = stat.icon;
          return (
            <div key={i} className="glass-panel p-5 rounded-2xl border border-white/10 space-y-1">
              <div className="flex items-center justify-between text-indigo-400 mb-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">{stat.label}</span>
                <Icon className="w-4 h-4" />
              </div>
              <p className="text-2xl sm:text-3xl font-extrabold text-white">{stat.val}</p>
              <p className="text-xs text-indigo-300/80">{stat.sub}</p>
            </div>
          );
        })}
      </section>

      {/* Feature Grid */}
      <section className="space-y-6">
        <div className="text-center space-y-2">
          <h2 className="text-2xl sm:text-3xl font-bold text-white">Platform Modules</h2>
          <p className="text-slate-400 text-sm">Select a module below to begin analyzing patient data or exploring ML benchmarks.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {features.map((feat, i) => {
            const Icon = feat.icon;
            return (
              <Link
                key={i}
                href={feat.href}
                className="group glass-panel p-6 rounded-2xl border border-white/10 glass-panel-hover flex flex-col justify-between"
              >
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className={`w-12 h-12 rounded-xl bg-gradient-to-tr ${feat.color} flex items-center justify-center shadow-lg`}>
                      <Icon className="w-6 h-6 text-white" />
                    </div>
                    <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-white/10 text-indigo-200 border border-white/10">
                      {feat.badge}
                    </span>
                  </div>

                  <div>
                    <h3 className="text-xl font-bold text-white group-hover:text-indigo-300 transition-colors">
                      {feat.title}
                    </h3>
                    <p className="text-slate-300 text-sm mt-1 leading-relaxed">
                      {feat.desc}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-1 text-sm font-semibold text-indigo-400 mt-6 group-hover:translate-x-1 transition-transform">
                  <span>Launch Module</span>
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
