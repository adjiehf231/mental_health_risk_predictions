'use client';

import { Github, Linkedin, Mail, Phone, Globe, Heart, Brain } from 'lucide-react';
import { useApp } from '@/lib/AppContext';

export default function Footer() {
  const { t } = useApp();

  return (
    <footer className="mt-auto border-t border-slate-200/20 dark:border-white/10 glass-panel py-10 transition-colors duration-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
        
        {/* Main Footer Row */}
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
          
          {/* Personal Branding & Name */}
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center text-white shadow-md">
                <Brain className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-lg font-extrabold text-adaptive-white">
                  {t.footer.name}
                </h3>
                <p className="text-xs text-indigo-500 dark:text-indigo-400 font-medium">
                  {t.footer.title}
                </p>
              </div>
            </div>
          </div>

          {/* Contact Details & Links */}
          <div className="flex flex-wrap items-center gap-4 sm:gap-6 text-xs text-adaptive-muted">
            <a
              href="mailto:adjieharifajar2301@gmail.com"
              className="flex items-center gap-1.5 hover:text-indigo-500 dark:hover:text-indigo-300 transition-colors"
            >
              <Mail className="w-4 h-4 text-indigo-500 dark:text-indigo-400" />
              <span>adjieharifajar2301@gmail.com</span>
            </a>

            <a
              href="https://wa.me/6282315193603"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 hover:text-indigo-500 dark:hover:text-indigo-300 transition-colors"
            >
              <Phone className="w-4 h-4 text-emerald-500 dark:text-emerald-400" />
              <span>+62 823-1519-3603</span>
            </a>

            <a
              href="https://portofolio-ahf.vercel.app/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 hover:text-indigo-500 dark:hover:text-indigo-300 transition-colors font-medium text-slate-800 dark:text-slate-200"
            >
              <Globe className="w-4 h-4 text-cyan-500 dark:text-cyan-400" />
              <span>Portfolio</span>
            </a>

            <a
              href="https://github.com/adjiehf231"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 hover:text-indigo-500 dark:hover:text-indigo-300 transition-colors"
            >
              <Github className="w-4 h-4 text-purple-500 dark:text-purple-400" />
              <span>GitHub</span>
            </a>

            <a
              href="https://www.linkedin.com/in/adjieharifajar"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 hover:text-indigo-500 dark:hover:text-indigo-300 transition-colors"
            >
              <Linkedin className="w-4 h-4 text-blue-500 dark:text-blue-400" />
              <span>LinkedIn</span>
            </a>
          </div>

        </div>

        {/* Bottom Copyright */}
        <div className="pt-6 border-t border-slate-200/20 dark:border-white/10 flex flex-col sm:flex-row items-center justify-between text-xs text-adaptive-muted gap-2">
          <p>© {new Date().getFullYear()} {t.footer.name}. {t.footer.copyright}</p>
          <div className="flex items-center gap-1">
            <span>MindRisk AI Analytics</span>
          </div>
        </div>

      </div>
    </footer>
  );
}
