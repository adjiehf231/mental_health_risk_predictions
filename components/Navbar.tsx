'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Brain, BarChart3, Cpu, Sparkles, History, Menu, X, Sun, Moon, Globe } from 'lucide-react';
import { useState } from 'react';
import { useApp } from '@/lib/AppContext';

export default function Navbar() {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const { theme, language, toggleTheme, toggleLanguage, t } = useApp();

  const navLinks = [
    { href: '/', label: t.nav.overview, icon: Brain },
    { href: '/prediction', label: t.nav.predictor, icon: Sparkles },
    { href: '/dashboard', label: t.nav.dashboard, icon: BarChart3 },
    { href: '/models', label: t.nav.benchmarks, icon: Cpu },
    { href: '/history', label: t.nav.history, icon: History },
  ];

  return (
    <header className="sticky top-0 z-50 w-full glass-panel border-b border-slate-200/20 dark:border-white/10 transition-colors duration-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Brand Logo */}
          <Link href="/" className="flex items-center gap-3 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/30 group-hover:scale-105 transition-transform duration-200">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <span className="text-lg font-bold text-adaptive-white">
                {t.nav.brand}
              </span>
              <span className="block text-[10px] uppercase tracking-wider text-indigo-500 dark:text-indigo-300 font-medium">
                {t.nav.sub}
              </span>
            </div>
          </Link>

          {/* Desktop Navigation Links */}
          <nav className="hidden md:flex items-center space-x-1">
            {navLinks.map((link) => {
              const Icon = link.icon;
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    isActive
                      ? 'bg-indigo-600/10 dark:bg-indigo-600/30 text-indigo-600 dark:text-white border border-indigo-500/30 font-semibold shadow-sm'
                      : 'text-adaptive-muted hover:text-adaptive-white hover:bg-black/5 dark:hover:bg-white/5'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-indigo-600 dark:text-indigo-400' : 'text-slate-400'}`} />
                  <span>{link.label}</span>
                </Link>
              );
            })}
          </nav>

          {/* Controls: Language Switcher & Theme Switcher */}
          <div className="flex items-center gap-2">
            
            {/* Language Switcher */}
            <button
              onClick={toggleLanguage}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold bg-slate-200/60 dark:bg-white/10 text-slate-800 dark:text-slate-200 hover:bg-slate-300 dark:hover:bg-white/20 transition-all border border-slate-300 dark:border-white/10 shadow-sm"
              title="Switch Language (ID / EN)"
            >
              <Globe className="w-3.5 h-3.5 text-indigo-500 dark:text-indigo-400" />
              <span>{language.toUpperCase()}</span>
            </button>

            {/* Dark / Light Theme Switcher */}
            <button
              onClick={toggleTheme}
              className="p-2 rounded-xl bg-slate-200/60 dark:bg-white/10 text-slate-800 dark:text-slate-200 hover:bg-slate-300 dark:hover:bg-white/20 transition-all border border-slate-300 dark:border-white/10 shadow-sm"
              title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            >
              {theme === 'dark' ? (
                <Sun className="w-4 h-4 text-amber-400" />
              ) : (
                <Moon className="w-4 h-4 text-indigo-600" />
              )}
            </button>

            {/* Mobile Menu Button */}
            <div className="md:hidden">
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="p-2 rounded-xl text-adaptive-muted hover:text-adaptive-white hover:bg-black/5 dark:hover:bg-white/10"
                aria-label="Toggle Menu"
              >
                {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
              </button>
            </div>

          </div>
        </div>
      </div>

      {/* Mobile Dropdown */}
      {mobileMenuOpen && (
        <div className="md:hidden glass-panel border-b border-slate-200/20 dark:border-white/10 px-4 pt-2 pb-4 space-y-1">
          {navLinks.map((link) => {
            const Icon = link.icon;
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                onClick={() => setMobileMenuOpen(false)}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-base font-medium ${
                  isActive
                    ? 'bg-indigo-600/10 dark:bg-indigo-600/30 text-indigo-600 dark:text-white border border-indigo-500/40'
                    : 'text-adaptive-muted hover:text-adaptive-white hover:bg-black/5 dark:hover:bg-white/5'
                }`}
              >
                <Icon className="w-5 h-5 text-indigo-500 dark:text-indigo-400" />
                <span>{link.label}</span>
              </Link>
            );
          })}
        </div>
      )}
    </header>
  );
}
