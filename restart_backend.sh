#!/bin/bash
# Force Backend Reload Script

echo "🔄 Forcing backend reload..."
echo ""

# Kill the uvicorn process
echo "1️⃣  Stopping backend (killing uvicorn processes)..."
pkill -f "uvicorn main:app" 2>/dev/null
sleep 2

# Verify stopped
if pgrep -f "uvicorn main:app" > /dev/null; then
    echo "❌ Backend still running, trying force kill..."
    pkill -9 -f "uvicorn main:app"
    sleep 1
else
    echo "✅ Backend stopped"
fi

echo ""
echo "2️⃣  Starting backend with validation..."
cd /Users/shahriar/Desktop/Desktop/Work/Document_Archive/MyDev/ArchiveFlow_AI/backend

# Activate conda and start
echo "   Activating conda llm10..."
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate llm10

echo "   Starting uvicorn..."
echo ""
echo "   Watch for: 'Loading CLIP classifier model' on first image upload"
echo "   Watch for: '🔍 Running zero-shot image validation' for each image"
echo ""

uvicorn main:app --reload --host 0.0.0.0 --port 8000

