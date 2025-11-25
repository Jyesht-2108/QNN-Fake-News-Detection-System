export interface PredictionResult {
  prediction: 'fake' | 'real';
  confidence: number;
  probability_fake: number;
  probability_real: number;
  risk_level: 'very_high' | 'high' | 'medium' | 'low' | 'very_low' | 'uncertain';
  explanation: string[];
  analysis: {
    word_count: number;
    has_exclamation: boolean;
    has_caps: boolean;
    processing_time: number;
  };
  model_info: ModelInfo;
}

export interface ModelInfo {
  accuracy: number;
  precision?: number;
  recall?: number;
  f1?: number;
  qubits: number;
  layers: number;
  model_type: string;
}

export interface Example {
  text: string;
  expected: 'real' | 'fake';
  category: string;
}

export interface Stats {
  model_info: ModelInfo;
  quantum_specs: {
    qubits: number;
    layers: number;
    parameters: number;
    device: string;
  };
  performance: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  };
}
