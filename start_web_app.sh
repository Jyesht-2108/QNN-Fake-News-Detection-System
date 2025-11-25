#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║     QUANTUM FAKE NEWS DETECTOR - WEB APP LAUNCHER                    ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if model exists
if [ ! -f "results/quantum_model_optimized.pkl" ] && [ ! -f "results/quantum_model.pkl" ]; then
    echo "⚠️  No trained model found!"
    echo ""
    echo "Please train a model first:"
    echo "  python train_optimized_fast.py"
    echo ""
    exit 1
fi

echo "✓ Model found"
echo ""

# Install Flask dependencies if needed
echo "📦 Checking backend dependencies..."
pip install -q flask flask-cors 2>/dev/null
echo "✓ Backend dependencies ready"
echo ""

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
    echo "✓ Frontend dependencies installed"
else
    echo "✓ Frontend dependencies ready"
fi

echo ""
echo "🚀 Starting servers..."
echo ""
echo "Backend API: http://localhost:5000"
echo "Frontend UI: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Start backend in background
python api_server.py &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 3

# Start frontend
cd frontend
npm start &
FRONTEND_PID=$!

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT

wait
