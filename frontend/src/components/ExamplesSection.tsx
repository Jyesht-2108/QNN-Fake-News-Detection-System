import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, ArrowRight } from 'lucide-react';
import { Example } from '../types';
import { getExamples } from '../utils/api';

interface ExamplesSectionProps {
  onSelectExample: (text: string) => void;
}

const ExamplesSection: React.FC<ExamplesSectionProps> = ({ onSelectExample }) => {
  const [examples, setExamples] = useState<Example[]>([]);

  useEffect(() => {
    getExamples().then(setExamples).catch(() => {});
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4 }}
      className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20"
    >
      <div className="flex items-center gap-2 mb-4">
        <Sparkles className="w-5 h-5 text-quantum-400" />
        <h3 className="text-xl font-bold text-white">Try These Examples</h3>
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        {examples.map((example, index) => (
          <motion.button
            key={index}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 + index * 0.1 }}
            whileHover={{ scale: 1.02, y: -2 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => onSelectExample(example.text)}
            className={`text-left p-4 rounded-xl border-2 transition-all group ${
              example.expected === 'fake'
                ? 'bg-fake-500/10 border-fake-500/30 hover:border-fake-500/60'
                : 'bg-real-500/10 border-real-500/30 hover:border-real-500/60'
            }`}
          >
            <div className="flex items-start justify-between mb-2">
              <span className={`text-xs font-semibold px-2 py-1 rounded ${
                example.expected === 'fake'
                  ? 'bg-fake-500/20 text-fake-300'
                  : 'bg-real-500/20 text-real-300'
              }`}>
                {example.category}
              </span>
              <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-white transition-colors" />
            </div>
            <p className="text-slate-300 text-sm line-clamp-3 group-hover:text-white transition-colors">
              {example.text}
            </p>
          </motion.button>
        ))}
      </div>
    </motion.div>
  );
};

export default ExamplesSection;
