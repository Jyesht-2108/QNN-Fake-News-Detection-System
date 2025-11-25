import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, Cpu, Layers, Zap } from 'lucide-react';
import { Stats } from '../types';

interface StatsPanelProps {
  stats: Stats;
}

const StatsPanel: React.FC<StatsPanelProps> = ({ stats }) => {
  const metrics = [
    { label: 'Accuracy', value: stats.performance.accuracy, icon: TrendingUp, color: 'text-real-400' },
    { label: 'Precision', value: stats.performance.precision, icon: Zap, color: 'text-quantum-400' },
    { label: 'Recall', value: stats.performance.recall, icon: Cpu, color: 'text-purple-400' },
    { label: 'F1 Score', value: stats.performance.f1_score, icon: Layers, color: 'text-blue-400' },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.3 }}
      className="sticky top-8 space-y-6"
    >
      {/* Model Info Card */}
      <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
        <h3 className="text-xl font-bold text-white mb-4">Model Information</h3>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
            <span className="text-slate-300">Type</span>
            <span className="text-white font-semibold">{stats.model_info.model_type}</span>
          </div>

          <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
            <span className="text-slate-300">Qubits</span>
            <span className="text-quantum-400 font-semibold">{stats.quantum_specs.qubits}</span>
          </div>

          <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
            <span className="text-slate-300">Layers</span>
            <span className="text-purple-400 font-semibold">{stats.quantum_specs.layers}</span>
          </div>

          <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
            <span className="text-slate-300">Parameters</span>
            <span className="text-white font-semibold">{stats.quantum_specs.parameters}</span>
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
        <h3 className="text-xl font-bold text-white mb-4">Performance Metrics</h3>
        
        <div className="space-y-4">
          {metrics.map((metric, index) => {
            const Icon = metric.icon;
            return (
              <motion.div
                key={metric.label}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4 + index * 0.1 }}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Icon className={`w-4 h-4 ${metric.color}`} />
                    <span className="text-slate-300 text-sm">{metric.label}</span>
                  </div>
                  <span className="text-white font-semibold">{metric.value.toFixed(1)}%</span>
                </div>
                <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${metric.value}%` }}
                    transition={{ duration: 1, delay: 0.5 + index * 0.1 }}
                    className={`h-full bg-gradient-to-r ${
                      metric.label === 'Accuracy' ? 'from-real-500 to-real-600' :
                      metric.label === 'Precision' ? 'from-quantum-500 to-quantum-600' :
                      metric.label === 'Recall' ? 'from-purple-500 to-purple-600' :
                      'from-blue-500 to-blue-600'
                    }`}
                  />
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Quantum Badge */}
      <motion.div
        className="bg-gradient-to-br from-quantum-500/20 to-purple-500/20 rounded-2xl p-6 border border-quantum-500/50"
        animate={{ boxShadow: ['0 0 20px rgba(14, 165, 233, 0.3)', '0 0 40px rgba(14, 165, 233, 0.5)', '0 0 20px rgba(14, 165, 233, 0.3)'] }}
        transition={{ duration: 2, repeat: Infinity }}
      >
        <div className="text-center">
          <div className="text-4xl mb-2">⚛️</div>
          <div className="text-white font-bold mb-1">Quantum Powered</div>
          <div className="text-slate-300 text-sm">
            Using {stats.quantum_specs.device}
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default StatsPanel;
