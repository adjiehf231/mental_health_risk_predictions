import Link from 'next/link';
import { HelpCircle, Home, Sparkles, BarChart3 } from 'lucide-react';

export default function NotFound() {
  return (
    <div className="min-h-[70vh] flex items-center justify-center p-4">
      <div className="glass-panel p-8 sm:p-10 rounded-3xl border border-white/10 max-w-md w-full text-center space-y-6">
        
        <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center mx-auto text-indigo-400">
          <HelpCircle className="w-8 h-8" />
        </div>

        <div className="space-y-2">
          <span className="text-xs font-bold uppercase tracking-wider text-indigo-400">404 Error</span>
          <h2 className="text-3xl font-extrabold text-white">Page Not Found</h2>
          <p className="text-slate-300 text-xs leading-relaxed">
            The module or page route you are trying to access does not exist or has been relocated.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
          <Link
            href="/prediction"
            className="inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 transition-colors shadow-lg"
          >
            <Sparkles className="w-4 h-4" />
            <span>AI Predictor</span>
          </Link>

          <Link
            href="/"
            className="inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-xs font-semibold text-slate-300 glass-panel glass-panel-hover border border-white/10"
          >
            <Home className="w-4 h-4" />
            <span>Homepage</span>
          </Link>
        </div>

      </div>
    </div>
  );
}
