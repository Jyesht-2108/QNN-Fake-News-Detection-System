import React from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, Info, Clock, FileText } from 'lucide-react';
import { PredictionResult } from '../types';

interface ResultCardProps {
  result: PredictionResult;
}

const ResultCard: React.FC<ResultCardProps> = ({ result }) => {
  const isFake = result.prediction === 'fake';
  
  const getRiskColor = (level: string) => {
    switch (level) {
      case 'very_high': return 'from-fake-600 to-fake-700';
      case 'high': return 'from-fake-500 to-fake-600';
      case 'medium': return 'from-yellow-500 to-orange-500';
      case 'low': return 'from-real-500 to-real-600';
      case 'very_low': return 'from-real-600 to-real-700';
      default: return 'from-slate-500 to-slate-600';
    }
  };

  const getRiskLabel = (level: string) => {
    switch (level) {
      case 'very_high': return 'Very High Risk';
      case 'high': return 'High Risk';
      case 'medium': return 'Medium Risk';
      case 'low': return 'Low Risk';
      case 'very_low': return 'Very Low Risk';
      default: return 'Uncertain';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className="space-y-6"
    >
      {/* Main Result Card */}
      <div className={`relative overflow-hidden rounded-2xl p-8 ${
        isFake ? 'bg-gradient-to-br from-fake-500/20 to-fake-600/20 border-2 border-fake-500/50' : 'bg-gradient-to-br from-real-500/20 to-real-600/20 border-2 border-real-500/50'
      }`}>
        {/* Animated background effect */}
        <div className="absolute inset-0 opacity-20">
          <div className={`absolute top-0 right-0 w-64 h-64 ${isFake ? 'bg-fake-500' : 'bg-real-500'} rounded-full blur-3xl animate-pulse-slow`} />
        </div>

        <div className="relative z-10">
          <div className="flex items-start justify-between mb-6">
            <div className="flex items-center gap-3">
              {isFake ? (
                <motion.div
                  animate={{ rotate: [0, 10, -10, 0] }}
                  transition={{ duration: 0.5, repeat: 2 }}
                >
                  <AlertTriangle className="w-12 h-12 text-fake-400" />
                </motion.div>
              ) : (
                <motion.div
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 0.5, repeat: 2 }}
                >
                  <CheckCircle className="w-12 h-12 text-real-400" />
                </motion.div>
              )}
              <div>
                <h3 className={`text-3xl font-bold ${isFake ? 'text-fake-300' : 'text-real-300'}`}>
                  {isFake ? 'FAKE NEWS' : 'REAL NEWS'}
                </h3>
                <p className="text-slate-300 text-sm mt-1">
                  {getRiskLabel(result.risk_level)}
                </p>
              </div>
            </div>

            <div className="text-right">
              <div className="text-4xl font-bold text-white">
                {result.confidence.toFixed(1)}%
              </div>
              <div className="text-slate-300 text-sm">Confidence</div>
            </div>
          </div>

          {/* Probability Bars */}
          <div className="space-y-3 mb-6">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-fake-300">Fake Probability</span>
                <span className="text-white font-medium">{result.probability_fake.toFixed(1)}%</span>
              </div>
              <div className="h-3 bg-white/10 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${result.probability_fake}%` }}
                  transition={{ duration: 1, ease: 'easeOut' }}
                  className="h-full bg-gradient-to-r from-fake-500 to-fake-600 rounded-full"
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-real-300">Real Probability</span>
                <span className="text-white font-medium">{result.probability_real.toFixed(1)}%</span>
              </div>
              <div className="h-3 bg-white/10 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${result.probability_real}%` }}
                  transition={{ duration: 1, ease: 'easeOut', delay: 0.2 }}
                  className="h-full bg-gradient-to-r from-real-500 to-real-600 rounded-full"
                />
              </div>
            </div>
          </div>

          {/* Explanation */}
          {result.explanation.length > 0 && (
            <div className="bg-white/5 rounded-xl p-4 mb-4">
              <div className="flex items-center gap-2 mb-2">
                <Info className="w-5 h-5 text-quantum-400" />
                <h4 className="text-white font-semibold">Analysis Insights</h4>
              </div>
              <ul className="space-y-1">
                {result.explanation.map((item, index) => (
                  <motion.li
                    key={index}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="text-slate-300 text-sm flex items-center gap-2"
                  >
                    <span className="w-1.5 h-1.5 bg-quantum-400 rounded-full" />
                    {item}
                  </motion.li>
                ))}
              </ul>
            </div>
          )}

          {/* Analysis Details */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white/5 rounded-lg p-3">
              <div className="flex items-center gap-2 mb-1">
                <FileText className="w-4 h-4 text-quantum-400" />
                <span className="text-slate-400 text-xs">Words</span>
              </div>
              <div className="text-white font-semibold">{result.analysis.word_count}</div>
            </div>

            <div className="bg-white/5 rounded-lg p-3">
              <div className="flex items-center gap-2 mb-1">
                <Clock className="w-4 h-4 text-quantum-400" />
                <span className="text-slate-400 text-xs">Processing</span>
              </div>
              <div className="text-white font-semibold">{result.analysis.processing_time}ms</div>
            </div>

            <div className="bg-white/5 rounded-lg p-3">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-quantum-400 text-xs">⚛️</span>
                <span className="text-slate-400 text-xs">Qubits</span>
              </div>
              <div className="text-white font-semibold">{result.model_info.qubits}</div>
            </div>

            <div className="bg-white/5 rounded-lg p-3">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-quantum-400 text-xs">🔬</span>
                <span className="text-slate-400 text-xs">Layers</span>
              </div>
              <div className="text-white font-semibold">{result.model_info.layers}</div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default ResultCard;
