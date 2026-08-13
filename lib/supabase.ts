import { createClient } from '@supabase/supabase-js';
import { AssessmentRecord } from './types';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || '';
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';

export const isSupabaseConfigured = Boolean(supabaseUrl && supabaseAnonKey);

export const supabase = isSupabaseConfigured
  ? createClient(supabaseUrl, supabaseAnonKey)
  : null;

// Local storage fallback key
const LOCAL_STORAGE_KEY = 'mental_health_assessments_local';

/**
 * Save assessment record to Supabase or local storage fallback
 */
export async function saveAssessmentRecord(
  record: AssessmentRecord
): Promise<{ success: boolean; data?: AssessmentRecord; error?: string }> {
  try {
    if (supabase && isSupabaseConfigured) {
      const { data, error } = await supabase
        .from('assessments')
        .insert([record])
        .select()
        .single();

      if (error) {
        console.warn('Supabase insert error, falling back to local:', error.message);
        return saveToLocalStorage(record);
      }
      return { success: true, data: data as AssessmentRecord };
    } else {
      return saveToLocalStorage(record);
    }
  } catch (err: any) {
    console.error('Save assessment error:', err);
    return saveToLocalStorage(record);
  }
}

/**
 * Fetch assessment history records
 */
export async function fetchAssessmentHistory(
  limit = 50
): Promise<AssessmentRecord[]> {
  try {
    if (supabase && isSupabaseConfigured) {
      const { data, error } = await supabase
        .from('assessments')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(limit);

      if (error) {
        console.warn('Supabase fetch error, fallback to local:', error.message);
        return getFromLocalStorage();
      }

      return (data as AssessmentRecord[]) || [];
    } else {
      return getFromLocalStorage();
    }
  } catch (err) {
    console.error('Fetch history error:', err);
    return getFromLocalStorage();
  }
}

// Fallback Helper Functions
function saveToLocalStorage(record: AssessmentRecord): { success: boolean; data?: AssessmentRecord } {
  if (typeof window === 'undefined') {
    return { success: true, data: record };
  }

  try {
    const existing = getFromLocalStorage();
    const newRecord: AssessmentRecord = {
      ...record,
      id: record.id || `local_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      created_at: record.created_at || new Date().toISOString(),
    };
    const updated = [newRecord, ...existing];
    localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(updated));
    return { success: true, data: newRecord };
  } catch (e) {
    console.error('LocalStorage write error:', e);
    return { success: true, data: record };
  }
}

function getFromLocalStorage(): AssessmentRecord[] {
  if (typeof window === 'undefined') return [];
  try {
    const data = localStorage.getItem(LOCAL_STORAGE_KEY);
    return data ? JSON.parse(data) : [];
  } catch {
    return [];
  }
}
