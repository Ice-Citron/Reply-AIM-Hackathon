#!/bin/bash

echo "🏥 Starting CareCompass..."
echo ""

# Check if backend is already running
if lsof -Pi :4000 -sTCP:LISTEN -t >/dev/null ; then
    echo "✅ Backend already running on port 4000"
else
    echo "🚀 Starting backend server..."
    cd backend && npm run dev &
    BACKEND_PID=$!
    echo "   Backend PID: $BACKEND_PID"
    sleep 2
    cd ..
fi

echo ""
echo "🎨 Starting frontend..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 IMPORTANT: Before you start the frontend..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Get a FREE Mapbox token:"
echo "   👉 https://account.mapbox.com/auth/signup/"
echo ""
echo "2. Edit frontend/.env and add your token:"
echo "   VITE_MAPBOX_TOKEN=pk.eyJ1..."
echo ""
echo "3. Then run: cd frontend && npm run dev"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Backend is ready at: http://localhost:4000"
echo "Health check: http://localhost:4000/api/health"
echo ""
