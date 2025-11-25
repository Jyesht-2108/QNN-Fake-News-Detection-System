import axios from 'axios';
import { PredictionResult, Example, Stats } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const predictNews = async (text: string): Promise<PredictionResult> => {
  const response = await api.post('/api/predict', { text });
  return response.data;
};

export const getExamples = async (): Promise<Example[]> => {
  const response = await api.get('/api/examples');
  return response.data.examples;
};

export const getStats = async (): Promise<Stats> => {
  const response = await api.get('/api/stats');
  return response.data;
};

export const checkHealth = async (): Promise<boolean> => {
  try {
    const response = await api.get('/api/health');
    return response.data.model_loaded;
  } catch {
    return false;
  }
};
