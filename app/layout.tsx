import type { Metadata } from 'next';
import './globals.css';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';

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
    'Supabase',
    'Healthcare Analytics',
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="flex flex-col min-h-screen antialiased bg-slate-950 text-slate-100">
        <Navbar />
        <main className="flex-grow">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
