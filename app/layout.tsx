import type { Metadata } from 'next';
import './globals.css';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { AppProvider } from '@/lib/AppContext';

export const metadata: Metadata = {
  title: 'MindRisk AI - Mental Health Risk Prediction Platform',
  description:
    'Interactive web platform for real-time mental health risk assessment powered by Machine Learning, Next.js, Vercel, and Supabase.',
  keywords: [
    'Mental Health',
    'Risk Prediction',
    'Machine Learning',
    'Decision Tree',
    'Next.js',
    'Healthcare Analytics',
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="id" className="dark">
      <body className="flex flex-col min-h-screen antialiased">
        <AppProvider>
          <Navbar />
          <main className="flex-grow">{children}</main>
          <Footer />
        </AppProvider>
      </body>
    </html>
  );
}
