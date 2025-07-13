#!/bin/bash

# IntelliDoc AI - FastAPI Server Startup Script

echo "🚀 Starting IntelliDoc AI FastAPI Server..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import fastapi, uvicorn" 2>/dev/null || {
    echo "⚠️  FastAPI dependencies not found. Installing..."
    pip install fastapi uvicorn python-multipart
}

# Set environment variables if not already set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  GEMINI_API_KEY not set. Please export your API key:"
    echo "export GEMINI_API_KEY='your_api_key_here'"
    exit 1
fi

# Start FastAPI server
echo "🌟 Starting server on http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔧 Alternative docs: http://localhost:8000/redoc"
echo ""
echo "Press Ctrl+C to stop the server"

uvicorn api:app --host 0.0.0.0 --port 8000 --reload
