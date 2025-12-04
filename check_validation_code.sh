#!/bin/bash
# Quick test to verify backend has new code

echo "🔍 Checking if updated code exists in document_processor.py..."
echo ""

cd /Users/shahriar/Desktop/Desktop/Work/Document_Archive/MyDev/ArchiveFlow_AI/backend/services

# Check for key validation code
if grep -q "🔍 Running zero-shot image validation" document_processor.py; then
    echo "✅ Validation log message found"
else
    echo "❌ Validation log message NOT found"
fi

if grep -q "_get_clip_classifier" document_processor.py; then
    echo "✅ CLIP classifier singleton found"
else
    echo "❌ CLIP classifier singleton NOT found"
fi

if grep -q "validation_status == 'failed'" document_processor.py; then
    echo "✅ Fail-safe validation found"
else
    echo "❌ Fail-safe validation NOT found"
fi

echo ""
echo "🔄 Checking if backend process is running..."
ps aux | grep -i "python.*main.py" | grep -v grep | head -2

echo ""
echo "⚠️  REMEMBER: You MUST restart backend for changes to take effect!"
echo "   In Terminal 2:"
echo "   1. Press Ctrl+C"
echo "   2. Run: conda activate llm10"
echo "   3. Run: python main.py"

