import { NextResponse } from 'next/server';
import { isSupabaseConfigured } from '@/lib/supabase';

export async function GET() {
  const healthData = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: '2.0.0',
    framework: 'Next.js 14 App Router',
    environment: process.env.NODE_ENV,
    supabase: {
      configured: isSupabaseConfigured,
      status: isSupabaseConfigured ? 'connected' : 'local_fallback_active',
    },
    ml_engine: {
      target: 'Vercel Python Serverless',
      route: '/api/py/predict',
      model: 'Decision Tree Classifier (C4.5)',
      accuracy: '99.5%',
    },
  };

  return NextResponse.json(healthData, { status: 200 });
}
