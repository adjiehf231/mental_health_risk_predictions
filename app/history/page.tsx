'use client';

import { useState, useEffect } from 'react';
import { History, Search, Download, Filter, Database, Calendar, ShieldAlert } from 'lucide-react';
import { AssessmentRecord } from '@/lib/types';
import { fetchAssessmentHistory, isSupabaseConfigured } from '@/lib/supabase';

export default function HistoryPage() {
  const [records, setRecords] = useState<AssessmentRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterRisk, setFilterRisk] = useState<string>('ALL');
  const [searchTerm, setSearchTerm] = useState<string>('');

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    setLoading(true);
    const data = await fetchAssessmentHistory(100);
    setRecords(data);
    setLoading(false);
  };

  const filteredRecords = records.filter((rec) => {
    const matchesRisk = filterRisk === 'ALL' || rec.risk_level.includes(filterRisk);
    const matchesSearch =
      rec.gender.toLowerCase().includes(searchTerm.toLowerCase()) ||
      rec.education_level.toLowerCase().includes(searchTerm.toLowerCase()) ||
      rec.risk_level.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesRisk && matchesSearch;
  });

  const exportToCSV = () => {
    if (records.length === 0) return;
    const headers = ['ID', 'Date', 'Age', 'Gender', 'Education', 'Sleep Hours', 'Work Stress', 'Anxiety', 'Depression', 'Risk Level', 'Confidence'];
    const csvRows = [headers.join(',')];

    filteredRecords.forEach((r) => {
      csvRows.push([
        r.id || '',
        r.created_at ? new Date(r.created_at).toLocaleString() : '',
        r.age,
        r.gender,
        r.education_level,
        r.sleep_hours,
        r.work_stress_level,
        r.anxiety_score,
        r.depression_score,
        `"${r.risk_level}"`,
        `${r.confidence}%`,
      ].join(','));
    });

    const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mental_health_assessments_${Date.now()}.csv`;
    a.click();
  };

  const getBadgeStyle = (risk: string) => {
    if (risk.includes('Low')) return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
    if (risk.includes('Moderate')) return 'bg-amber-500/10 text-amber-400 border-amber-500/30';
    return 'bg-rose-500/10 text-rose-400 border-rose-500/30';
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/10 text-amber-300 border border-amber-500/20 mb-2">
            <Database className="w-4 h-4 text-amber-400" />
            <span>{isSupabaseConfigured ? 'Supabase PostgreSQL Sync' : 'Local Persistence Sync'}</span>
          </div>
          <h1 className="text-3xl font-extrabold text-white">📜 Assessment History Log</h1>
          <p className="text-slate-400 text-sm">Real-time log of mental health risk assessments and patient records.</p>
        </div>

        <button
          onClick={exportToCSV}
          disabled={filteredRecords.length === 0}
          className="inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl font-semibold text-xs text-white bg-indigo-600 hover:bg-indigo-500 transition-colors shadow-lg shadow-indigo-600/20 disabled:opacity-50"
        >
          <Download className="w-4 h-4" />
          <span>Export CSV Report</span>
        </button>
      </div>

      {/* Filter & Search Bar */}
      <div className="glass-panel p-4 rounded-2xl border border-white/10 flex flex-col sm:flex-row items-center justify-between gap-4">
        
        {/* Search */}
        <div className="relative w-full sm:w-72">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search logs..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full glass-input pl-10 pr-4 py-2 rounded-xl text-xs"
          />
        </div>

        {/* Filter Dropdown */}
        <div className="flex items-center gap-2 w-full sm:w-auto">
          <Filter className="w-4 h-4 text-indigo-400" />
          <span className="text-xs text-slate-300 font-medium">Filter Risk:</span>
          <select
            value={filterRisk}
            onChange={(e) => setFilterRisk(e.target.value)}
            className="glass-input px-3 py-1.5 rounded-xl text-xs"
          >
            <option value="ALL" className="bg-slate-900">All Risk Levels</option>
            <option value="Low" className="bg-slate-900">Low Risk Only</option>
            <option value="Moderate" className="bg-slate-900">Moderate Risk Only</option>
            <option value="High" className="bg-slate-900">High Risk Only</option>
          </select>
        </div>

      </div>

      {/* Records Table */}
      <div className="glass-panel rounded-3xl border border-white/10 overflow-hidden">
        {loading ? (
          <div className="p-12 text-center text-slate-400 space-y-2">
            <History className="w-8 h-8 text-indigo-400 animate-spin mx-auto" />
            <p className="text-xs font-medium">Fetching history records...</p>
          </div>
        ) : filteredRecords.length === 0 ? (
          <div className="p-12 text-center text-slate-400 space-y-3">
            <ShieldAlert className="w-10 h-10 text-slate-500 mx-auto" />
            <p className="text-sm font-semibold text-white">No Assessment Records Found</p>
            <p className="text-xs text-slate-400 max-w-sm mx-auto">
              Perform an AI Risk Assessment on the <a href="/prediction" className="text-indigo-400 underline">Prediction Page</a> to populate your database history log.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-slate-200">
              <thead className="bg-white/5 text-xs text-slate-400 uppercase tracking-wider border-b border-white/10">
                <tr>
                  <th className="px-6 py-4">Timestamp</th>
                  <th className="px-6 py-4">Demographics</th>
                  <th className="px-6 py-4">Sleep / Stress</th>
                  <th className="px-6 py-4">Anxiety / Dep</th>
                  <th className="px-6 py-4">Risk Level</th>
                  <th className="px-6 py-4 text-right">Confidence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 text-xs">
                {filteredRecords.map((r, idx) => (
                  <tr key={r.id || idx} className="hover:bg-white/5">
                    <td className="px-6 py-4 text-slate-400">
                      <div className="flex items-center gap-1.5">
                        <Calendar className="w-3.5 h-3.5 text-indigo-400" />
                        <span>{r.created_at ? new Date(r.created_at).toLocaleString() : 'Just now'}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-white font-medium">
                      {r.age} yrs • {r.gender} • {r.education_level}
                    </td>
                    <td className="px-6 py-4">
                      {r.sleep_hours}h sleep • Stress {r.work_stress_level}/10
                    </td>
                    <td className="px-6 py-4">
                      Anx {r.anxiety_score}/10 • Dep {r.depression_score}/10
                    </td>
                    <td className="px-6 py-4">
                      <span className={`px-2.5 py-1 rounded-full text-[11px] font-bold border ${getBadgeStyle(r.risk_level)}`}>
                        {r.risk_level}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-right font-extrabold text-white">
                      {typeof r.confidence === 'number' ? r.confidence.toFixed(1) : r.confidence}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

    </div>
  );
}
