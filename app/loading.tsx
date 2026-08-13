import { Brain } from 'lucide-react';

export default function Loading() {
  return (
    <div className="min-h-[70vh] flex flex-col items-center justify-center p-4 space-y-4">
      <div className="w-14 h-14 rounded-2xl bg-indigo-600/30 border border-indigo-500/30 flex items-center justify-center text-indigo-400 animate-bounce">
        <Brain className="w-8 h-8" />
      </div>
      <div className="text-center space-y-1">
        <p className="text-sm font-bold text-white">Loading MindRisk AI...</p>
        <p className="text-xs text-slate-400">Initializing glassmorphism design tokens & ML modules</p>
      </div>
    </div>
  );
}
