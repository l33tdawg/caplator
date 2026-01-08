#!/bin/bash
#
# Build script for creating Translator.app macOS bundle
#
# Usage: ./scripts/build_app.sh
#

set -e

echo "🔨 Building Translator.app..."

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_DIR"

# Install build dependencies
echo "📦 Installing build dependencies..."
pip3 install pyinstaller pillow -q 2>/dev/null || true

# Create icon
echo "🎨 Creating app icon..."
python3 scripts/create_icon.py || {
    echo "⚠️  Could not create custom icon, continuing..."
}

# Clean previous build
echo "🧹 Cleaning previous build..."
rm -rf build dist

# Build the app
echo "📦 Building app bundle with PyInstaller..."
pyinstaller Translator.spec --noconfirm

# Check if build succeeded
if [ -d "dist/Translator.app" ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    SIZE=$(du -sh dist/Translator.app | cut -f1)
    echo "📍 App location: dist/Translator.app ($SIZE)"
    echo ""
    echo "To install to Applications:"
    echo "  cp -r dist/Translator.app /Applications/"
    echo ""
    echo "To run:"
    echo "  open dist/Translator.app"
    echo ""
    
    # Ask to open
    read -p "Open the app now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open dist/Translator.app
    fi
else
    echo ""
    echo "❌ Build failed. Check the output above for errors."
    exit 1
fi
