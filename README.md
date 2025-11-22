# 🏥 CareCompass - Medical AI Consultant

AI-Powered Medical Cost & Reliability Comparison Platform

Team: Shi Hao Ng, Alex Goldman, Davin Wong and Ryan Deng

## Features

- 💬 **AI Consultant Chat** - Ask questions about medical procedure costs and get AI-powered recommendations
- 🗺️ **Global Hospital Map** - Visualize hospitals worldwide with reliability scores and pricing
- 🔍 **Cost Comparison** - Compare US vs international medical tourism options
- 📊 **Reliability Scores** - Hospital quality ratings based on multiple factors

## Quick Start

### Prerequisites

- Node.js 18+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Mapbox token ([Free signup](https://account.mapbox.com/auth/signup/))

### Backend Setup

```bash
cd backend

# Install dependencies
npm install

# Create .env file
cp .env.example .env

# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-...

# Start backend server
npm run dev
```

Backend will run on `http://localhost:4000`

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Create .env file
cp .env.example .env

# Edit .env and add:
# VITE_MAPBOX_TOKEN=pk.eyJ1...
# VITE_API_URL=http://localhost:4000

# Start frontend dev server
npm run dev
```

Frontend will run on `http://localhost:5173`

## Usage

1. Open `http://localhost:5173` in your browser
2. Try the **AI Consultant** tab:
   - Ask: "I need a hip replacement. What are my options?"
   - Ask: "Compare knee replacement costs between US and Thailand"
3. Check the **Browse Hospitals** tab to see the map view

## Data

The app uses mock data for 18 hospitals across:
- 🇺🇸 United States (9 hospitals)
- 🇹🇭 Thailand (2 hospitals)
- 🇲🇽 Mexico (1 hospital)
- 🇮🇳 India (2 hospitals)
- 🇹🇷 Turkey (1 hospital)
- 🇸🇬 Singapore (1 hospital)
- 🇩🇪 Germany (1 hospital)
- 🇰🇷 South Korea (1 hospital)

### Procedures Available
- Hip Replacement
- Knee Replacement
- Heart Surgery

## Tech Stack

- **Frontend**: React + Vite + Mapbox GL JS
- **Backend**: Node.js + Express + OpenAI API
- **Data**: Mock JSON (extendable to real databases)

## Project Structure

```
.
├── backend/
│   ├── server.js          # Express server with OpenAI integration
│   ├── package.json
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatTab.jsx    # AI chat interface
│   │   │   └── MapTab.jsx     # Mapbox hospital visualization
│   │   ├── App.jsx
│   │   └── App.css
│   ├── package.json
│   └── .env.example
└── data/
    └── hospitals.json     # Mock hospital data
```

## Deployment (Optional)

### Deploy to Vercel (Frontend)
```bash
cd frontend
npm run build
vercel deploy
```

### Deploy to Render (Backend)
1. Push to GitHub
2. Connect to Render
3. Add environment variables
4. Deploy

## Future Enhancements

- [ ] Real-time price data from hospital APIs
- [ ] User authentication & saved comparisons
- [ ] Prescription drug price comparison
- [ ] Insurance coverage checker
- [ ] Appointment booking integration
- [ ] Flight + accommodation cost calculator
- [ ] More procedures and hospitals

## License

MIT
