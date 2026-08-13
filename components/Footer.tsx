import Link from 'next/link';
import { Brain, Heart, Github, ShieldCheck } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="mt-auto border-t border-white/10 glass-panel py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-600/30 flex items-center justify-center border border-indigo-500/30">
              <Brain className="w-4 h-4 text-indigo-400" />
            </div>
            <div>
              <span className="text-sm font-semibold text-slate-200">MindRisk AI Platform</span>
              <p className="text-xs text-slate-400">Mental Health Assessment & Prediction Powered by ML</p>
            </div>
          </div>

          <div className="flex items-center gap-6 text-xs text-slate-400">
            <Link href="/PRD.md" className="hover:text-indigo-400 transition-colors">PRD Document</Link>
            <Link href="/guide_deploy.md" className="hover:text-indigo-400 transition-colors">Deployment Guide</Link>
            <Link href="/qa_automation.md" className="hover:text-indigo-400 transition-colors">QA Manual</Link>
            <a
              href="https://github.com/adjiehf231/mental_health_risk_predictions"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 hover:text-white transition-colors"
            >
              <Github className="w-4 h-4" />
              <span>GitHub Repo</span>
            </a>
          </div>

          <div className="flex items-center gap-1.5 text-xs text-slate-400">
            <span>Built with</span>
            <Heart className="w-3.5 h-3.5 text-rose-500 fill-rose-500" />
            <span>using Next.js & Supabase</span>
          </div>

        </div>
      </div>
    </footer>
  );
}
