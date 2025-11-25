import React from 'react';
import { motion } from 'framer-motion';
import { Atom, CheckCircle, XCircle } from 'lucide-react';

interface HeaderProps {
  modelLoaded: boolean;
}

const Header: React.FC<HeaderProps> = ({ modelLoaded }) => {
  return (
    <header className="border-b border-white/10 backdrop-blur-sm bg-white/5">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <motion.div
            className="flex items-center gap-3"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <div className="relative">
              <Atom className="w-8 h-8 text-quantum-400 animate-spin" style={{ animationDuration: '8s' }} />
              <div className="absolute inset-0 bg-quantum-400/20 blur-xl rounded-full" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">Quantum Detector</h2>
              <p className="text-xs text-slate-400">Neural Network v1.0</p>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center gap-2"
          >
            {modelLoaded ? (
              <>
                <CheckCircle className="w-5 h-5 text-real-400" />
                <span className="text-sm text-real-400 font-medium">Model Ready</span>
              </>
            ) : (
              <>
                <XCircle className="w-5 h-5 text-fake-400" />
                <span className="text-sm text-fake-400 font-medium">Model Not Loaded</span>
              </>
            )}
          </motion.div>
        </div>
      </div>
    </header>
  );
};

export default Header;
