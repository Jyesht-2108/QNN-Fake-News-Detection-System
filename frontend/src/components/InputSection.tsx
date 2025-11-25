import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Send, Loader2, AlertCircle } from 'lucide-react';

interface InputSectionProps {
  onPredict: (text: string) => void;
  loading: boolean;
  modelLoaded: boolean;
}

const InputSection: React.FC<InputSectionProps> = ({ onPredict, loading, modelLoaded }) => {
  const [text, setText] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (text.trim() && !loading && modelLoaded) {
      onPredict(text);
    }
  };

  const charCount = text.length;
  const wordCount = text.trim().split(/\s+/).filter(Boolean).length;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
      className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20 shadow-2xl"
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-white font-medium mb-2">
            Enter News Article or Headline
          </label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste or type the news text you want to analyze..."
            className="w-full h-40 px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-quantum-500 focus:border-transparent resize-none transition-all"
            disabled={loading || !modelLoaded}
          />
          <div className="flex justify-between mt-2 text-sm text-slate-400">
            <span>{wordCount} words</span>
            <span>{charCount} characters</span>
          </div>
        </div>

        {!modelLoaded && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center gap-2 p-3 bg-fake-500/20 border border-fake-500/50 rounded-lg"
          >
            <AlertCircle className="w-5 h-5 text-fake-400" />
            <span className="text-fake-300 text-sm">
              Model not loaded. Please train a model first.
            </span>
          </motion.div>
        )}

        <motion.button
          type="submit"
          disabled={!text.trim() || loading || !modelLoaded}
          whileHover={{ scale: modelLoaded && !loading ? 1.02 : 1 }}
          whileTap={{ scale: modelLoaded && !loading ? 0.98 : 1 }}
          className={`w-full py-4 rounded-xl font-semibold text-white flex items-center justify-center gap-2 transition-all ${
            !text.trim() || loading || !modelLoaded
              ? 'bg-slate-600 cursor-not-allowed'
              : 'bg-gradient-to-r from-quantum-500 to-purple-500 hover:from-quantum-600 hover:to-purple-600 shadow-lg shadow-quantum-500/50'
          }`}
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Analyzing with Quantum AI...
            </>
          ) : (
            <>
              <Send className="w-5 h-5" />
              Analyze News
            </>
          )}
        </motion.button>
      </form>
    </motion.div>
  );
};

export default InputSection;
