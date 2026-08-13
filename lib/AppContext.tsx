'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';
import { Language, Theme, translations } from './i18n';

interface AppContextType {
  theme: Theme;
  language: Language;
  toggleTheme: () => void;
  toggleLanguage: () => void;
  setLanguage: (lang: Language) => void;
  t: typeof translations['en'];
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>('dark');
  const [language, setLanguageState] = useState<Language>('id');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const savedTheme = localStorage.getItem('app_theme') as Theme;
    const savedLang = localStorage.getItem('app_lang') as Language;

    if (savedTheme) {
      setTheme(savedTheme);
      applyThemeClass(savedTheme);
    } else {
      applyThemeClass('dark');
    }

    if (savedLang) {
      setLanguageState(savedLang);
    }
  }, []);

  const applyThemeClass = (newTheme: Theme) => {
    const root = document.documentElement;
    if (newTheme === 'dark') {
      root.classList.add('dark');
      root.classList.remove('light');
    } else {
      root.classList.add('light');
      root.classList.remove('dark');
    }
  };

  const toggleTheme = () => {
    const nextTheme = theme === 'dark' ? 'light' : 'dark';
    setTheme(nextTheme);
    localStorage.setItem('app_theme', nextTheme);
    applyThemeClass(nextTheme);
  };

  const toggleLanguage = () => {
    const nextLang = language === 'en' ? 'id' : 'en';
    setLanguageState(nextLang);
    localStorage.setItem('app_lang', nextLang);
  };

  const setLanguage = (lang: Language) => {
    setLanguageState(lang);
    localStorage.setItem('app_lang', lang);
  };

  const t = translations[language];

  return (
    <AppContext.Provider
      value={{
        theme,
        language,
        toggleTheme,
        toggleLanguage,
        setLanguage,
        t,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}
