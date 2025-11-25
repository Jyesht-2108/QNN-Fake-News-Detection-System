# Quantum Fake News Detector - Web App Guide 🌐

Beautiful, interactive web interface for the Quantum Fake News Detector!

## 🎨 Features

- **Beautiful UI**: Modern gradient design with glassmorphism effects
- **Smooth Animations**: Framer Motion animations throughout
- **Real-time Analysis**: Instant fake news detection
- **Interactive Examples**: Pre-loaded test cases
- **Performance Stats**: Live model metrics display
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Dark Theme**: Eye-friendly dark mode interface

## 🚀 Quick Start

### Step 1: Install Backend Dependencies

```bash
# Install Flask and CORS support
pip install flask flask-cors
```

### Step 2: Start the API Server

```bash
# Make sure you have a trained model first!
# If not, run: python train_optimized_fast.py

# Start the Flask API server
python api_server.py
```

The API will start at `http://localhost:5000`

### Step 3: Install Frontend Dependencies

```bash
cd frontend
npm install
```

### Step 4: Start the React App

```bash
npm start
```

The web app will open at `http://localhost:3000`

## 📁 Project Structure

```
QNN-Fake_News_Detection/
├── api_server.py              # Flask backend API
├── frontend/
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── Header.tsx
│   │   │   ├── InputSection.tsx
│   │   │   ├── ResultCard.tsx
│   │   │   ├── StatsPanel.tsx
│   │   │   └── ExamplesSection.tsx
│   │   ├── types/             # TypeScript types
│   │   ├── utils/             # API utilities
│   │   ├── App.tsx            # Main app component
│   │   ├── index.tsx          # Entry point
│   │   └── index.css          # Global styles
│   ├── public/
│   ├── package.json
│   ├── tailwind.config.js     # Tailwind configuration
│   └── tsconfig.json          # TypeScript configuration
└── requirements_web.txt       # Web dependencies
```

## 🎯 API Endpoints

### Health Check
```
GET /api/health
```
Returns server status and model info.

### Predict
```
POST /api/predict
Body: { "text": "Your news article here" }
```
Returns prediction with confidence, risk level, and analysis.

### Get Examples
```
GET /api/examples
```
Returns pre-loaded example news articles.

### Get Stats
```
GET /api/stats
```
Returns model performance metrics and quantum specs.

## 🎨 UI Components

### 1. Header
- Shows model status (loaded/not loaded)
- Animated quantum atom logo
- Real-time connection indicator

### 2. Input Section
- Large textarea for news input
- Word and character counter
- Animated submit button
- Loading state with spinner

### 3. Result Card
- Large, animated result display
- Fake/Real indicator with icons
- Confidence percentage
- Probability bars (animated)
- Analysis insights
- Processing metrics
- Quantum specs display

### 4. Stats Panel
- Model information
- Quantum specifications
- Performance metrics with animated bars
- Quantum-powered badge

### 5. Examples Section
- Pre-loaded test cases
- Color-coded by category
- One-click testing
- Hover animations

## 🎭 Animations

- **Entrance animations**: Smooth fade-in and slide-up
- **Hover effects**: Scale and lift on hover
- **Loading states**: Spinning loaders and pulsing effects
- **Result reveal**: Animated bars and counters
- **Background**: Floating gradient orbs
- **Quantum badge**: Pulsing glow effect

## 🎨 Color Scheme

### Quantum Blue
- Primary: `#0ea5e9`
- Used for: Buttons, accents, quantum elements

### Fake Red
- Primary: `#ef4444`
- Used for: Fake news indicators, warnings

### Real Green
- Primary: `#22c55e`
- Used for: Real news indicators, success

### Purple
- Primary: `#a855f7`
- Used for: Gradients, secondary accents

## 🔧 Customization

### Change Colors

Edit `frontend/tailwind.config.js`:

```javascript
colors: {
  quantum: {
    500: '#YOUR_COLOR',
    // ...
  }
}
```

### Modify Animations

Edit component files or `tailwind.config.js`:

```javascript
animation: {
  'your-animation': 'your-keyframes 3s ease-in-out infinite',
}
```

### Add New Features

1. Create new component in `frontend/src/components/`
2. Add API endpoint in `api_server.py`
3. Update types in `frontend/src/types/index.ts`
4. Import and use in `App.tsx`

## 🐛 Troubleshooting

### Backend Issues

**"Model not loaded"**
```bash
# Train a model first
python train_optimized_fast.py

# Then restart API server
python api_server.py
```

**"Port 5000 already in use"**
```bash
# Kill the process using port 5000
lsof -ti:5000 | xargs kill -9

# Or change port in api_server.py
app.run(port=5001)
```

### Frontend Issues

**"Cannot connect to API"**
- Make sure API server is running on port 5000
- Check CORS is enabled
- Verify API_BASE_URL in `frontend/src/utils/api.ts`

**"npm install fails"**
```bash
# Clear cache and retry
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

**"Tailwind styles not working"**
```bash
# Rebuild Tailwind
npm run build
```

## 📱 Responsive Design

The app is fully responsive:
- **Desktop**: Full layout with sidebar
- **Tablet**: Stacked layout
- **Mobile**: Single column, optimized touch targets

## 🚀 Deployment

### Build for Production

```bash
cd frontend
npm run build
```

This creates an optimized build in `frontend/build/`

### Deploy Backend

```bash
# Use gunicorn for production
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
```

### Deploy Frontend

Upload the `frontend/build/` folder to:
- Vercel
- Netlify
- GitHub Pages
- Your own server

## 🎉 Features Showcase

### Real-time Analysis
- Type or paste news text
- Click "Analyze News"
- Get instant results with confidence scores

### Visual Feedback
- Color-coded results (red for fake, green for real)
- Animated probability bars
- Risk level indicators
- Processing time display

### Quantum Insights
- Shows number of qubits used
- Displays circuit layers
- Model accuracy metrics
- Real-time performance stats

### Interactive Examples
- Click any example to test
- See different categories
- Compare results
- Learn from patterns

## 💡 Tips

1. **Best Results**: Use complete sentences or paragraphs
2. **Test Examples**: Try the pre-loaded examples first
3. **Watch Animations**: Enjoy the smooth transitions
4. **Check Stats**: Monitor model performance in real-time
5. **Mobile Friendly**: Works great on phones too!

## 🎨 Design Philosophy

- **Glassmorphism**: Frosted glass effects with backdrop blur
- **Gradients**: Smooth color transitions
- **Animations**: Purposeful, not distracting
- **Dark Theme**: Reduces eye strain
- **Quantum Aesthetic**: Futuristic, tech-forward design

## 📊 Performance

- **Fast Loading**: Optimized React build
- **Smooth Animations**: 60 FPS with Framer Motion
- **Quick API**: Sub-second predictions
- **Responsive**: Instant UI feedback

## 🔮 Future Enhancements

- [ ] Batch analysis (multiple articles)
- [ ] History of predictions
- [ ] Export results as PDF
- [ ] Share results via link
- [ ] Dark/Light theme toggle
- [ ] Multi-language support
- [ ] Voice input
- [ ] Browser extension

---

## 🎊 You're Ready!

Your beautiful quantum fake news detector web app is ready to use!

**Start the servers:**
```bash
# Terminal 1: Backend
python api_server.py

# Terminal 2: Frontend
cd frontend && npm start
```

**Open your browser:**
```
http://localhost:3000
```

Enjoy your beautiful, quantum-powered fake news detector! 🚀✨
