# 🚀 QUICK SETUP GUIDE

## Step 1: Get API Keys

### OpenAI API Key (REQUIRED)
1. Go to https://platform.openai.com/api-keys
2. Sign in or create account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-...`)

### Mapbox Token (REQUIRED for map)
1. Go to https://account.mapbox.com/auth/signup/
2. Create free account
3. Get your default public token (starts with `pk.eyJ...`)

## Step 2: Configure Backend

```bash
cd backend

# Copy example env
cp .env.example .env

# Edit .env file and add your OpenAI key
nano .env  # or use any text editor
```

Your `backend/.env` should look like:
```
OPENAI_API_KEY=sk-your-actual-key-here
PORT=4000
```

## Step 3: Configure Frontend

```bash
cd ../frontend

# Copy example env
cp .env.example .env

# Edit .env file
nano .env  # or use any text editor
```

Your `frontend/.env` should look like:
```
VITE_MAPBOX_TOKEN=pk.your-actual-token-here
VITE_API_URL=http://localhost:4000
```

## Step 4: Start Both Servers

### Terminal 1 - Backend
```bash
cd backend
npm run dev
```

You should see:
```
🚀 Backend server running on http://localhost:4000
📊 Health check: http://localhost:4000/api/health
✅ Loaded 18 hospitals
```

### Terminal 2 - Frontend
```bash
cd frontend
npm run dev
```

You should see:
```
  VITE v7.x.x  ready in XXX ms

  ➜  Local:   http://localhost:5173/
  ➜  press h + enter to show help
```

## Step 5: Test the App

1. Open http://localhost:5173 in your browser
2. You should see the CareCompass app with two tabs
3. Try the chat:
   - "I need a hip replacement. Compare US vs Thailand options"
   - "Show me the cheapest hospitals for knee replacement"
4. Click the map tab to see hospital locations

## Troubleshooting

### Backend won't start
- Check that your OPENAI_API_KEY is correct in `backend/.env`
- Make sure port 4000 is not in use
- Run `npm install` again in the backend folder

### Frontend won't start
- Check that your VITE_MAPBOX_TOKEN is correct in `frontend/.env`
- Make sure port 5173 is not in use
- Run `npm install` again in the frontend folder

### Map doesn't show
- Verify your Mapbox token is correct
- Check browser console for errors
- Make sure you copied the PUBLIC token (starts with `pk.`)

### Chat doesn't work
- Make sure backend is running on port 4000
- Check backend terminal for errors
- Verify OpenAI API key has credits

## Ready to Demo!

Once both servers are running, you're ready to demonstrate:
1. AI-powered cost consultation
2. Global hospital comparison
3. Interactive map with reliability scores
4. Medical tourism cost analysis
