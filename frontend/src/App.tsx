import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Shield, Zap, Activity } from 'lucide-react';
import InputSection from './components/InputSection';
import ResultCard from './components/ResultCard';
import StatsPanel from './components/StatsPanel';
import ExamplesSection from './components/ExamplesSection';
import Header from './components/Header';
import { PredictionResult, Stats } from './types';
import { predictNews, getStats, checkHealth } from './utils/api';

function App() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState<Stats | null>(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Check if model is loaded
    checkHealth().then(setModelLoaded);
    
    // Load stats
    getStats()
      .then(setStats)
      .catch(() => console.log('Stats not available'));
  }, []);

  const handlePredict = async (text: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const prediction = await predictNews(text);
      setResult(prediction);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to analyze text');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Animated background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-quantum-500/20 rounded-full blur-3xl animate-pulse-slow" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse-slow" style={{ animationDelay: '1s' }} />
      </div>

      <div className="relative z-10">
        <Header modelLoaded={modelLoaded} />

        <main className="container mx-auto px-4 py-8 max-w-7xl">
          {/* Hero Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-12"
          >
            <motion.div
              className="inline-flex items-center gap-2 px-4 py-2 bg-quantum-500/20 rounded-full mb-6"
              animate={{ scale: [1, 1.05, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <Sparkles className="w-4 h-4 text-quantum-400" />
              <span className="text-quantum-300 text-sm font-medium">
                Powered by Quantum Neural Networks
              </span>
            </motion.div>

            <h1 className="text-5xl md:text-6xl font-bold text-white mb-4">
              Quantum Fake News
              <span className="bg-gradient-to-r from-quantum-400 to-purple-400 bg-clip-text text-transparent">
                {' '}Detector
              </span>
            </h1>
            
            <p className="text-xl text-slate-300 max-w-2xl mx-auto">
              Advanced quantum computing meets AI to detect misinformation with unprecedented accuracy
            </p>

            {/* Feature badges */}
            <div className="flex flex-wrap justify-center gap-4 mt-8">
              <motion.div
                whileHover={{ scale: 1.05 }}
                className="flex items-center gap-2 px-4 py-2 bg-white/10 backdrop-blur-sm rounded-lg"
              >
                <Shield className="w-5 h-5 text-real-400" />
                <span className="text-white text-sm">Adversarial Robust</span>
              </motion.div>
              
              <motion.div
                whileHover={{ scale: 1.05 }}
                className="flex items-center gap-2 px-4 py-2 bg-white/10 backdrop-blur-sm rounded-lg"
              >
                <Zap className="w-5 h-5 text-quantum-400" />
                <span className="text-white text-sm">Real-time Analysis</span>
              </motion.div>
              
              <motion.div
                whileHover={{ scale: 1.05 }}
                className="flex items-center gap-2 px-4 py-2 bg-white/10 backdrop-blur-sm rounded-lg"
              >
                <Activity className="w-5 h-5 text-purple-400" />
                <span className="text-white text-sm">
                  {stats ? `${stats.performance.accuracy.toFixed(1)}% Accurate` : 'High Accuracy'}
                </span>
              </motion.div>
            </div>
          </motion.div>

          {/* Main Content Grid */}
          <div className="grid lg:grid-cols-3 gap-8">
            {/* Left Column - Input and Examples */}
            <div className="lg:col-span-2 space-y-8">
              <InputSection
                onPredict={handlePredict}
                loading={loading}
                modelLoaded={modelLoaded}
              />

              <AnimatePresence mode="wait">
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="bg-fake-500/20 border border-fake-500/50 rounded-xl p-4 text-fake-300"
                  >
                    {error}
                  </motion.div>
                )}

                {result && (
                  <ResultCard result={result} />
                )}
              </AnimatePresence>

              <ExamplesSection onSelectExample={handlePredict} />
            </div>

            {/* Right Column - Stats */}
            <div className="lg:col-span-1">
              {stats && <StatsPanel stats={stats} />}
            </div>
          </div>
        </main>

        {/* Footer */}
        <footer className="text-center py-8 text-slate-400 text-sm">
          <p>Built with React, TypeScript, Tailwind CSS & PennyLane</p>
          <p className="mt-2">Quantum Neural Network for Fake News Detection</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
