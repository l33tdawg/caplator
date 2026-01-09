#!/bin/bash
#
# Create a lightweight .app launcher using AppleScript
# This is the most reliable way to launch Python apps on macOS
#

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_NAME="Translator.app"
APP_PATH="/Applications/$APP_NAME"

echo "🔨 Creating Translator.app launcher..."

# Find the right Python (the one with PyQt6 installed)
PYTHON_PATH=$(which python3)

# Remove old app if exists
rm -rf "$APP_PATH"

# Create AppleScript-based app (most reliable on macOS)
osacompile -o "$APP_PATH" -e "do shell script \"cd '$PROJECT_DIR' && '$PYTHON_PATH' main.py &> /dev/null &\""

# Set icon if exists
if [ -f "$PROJECT_DIR/assets/icon.icns" ]; then
    cp "$PROJECT_DIR/assets/icon.icns" "$APP_PATH/Contents/Resources/applet.icns"
fi

# Update Info.plist with better metadata
/usr/libexec/PlistBuddy -c "Set :CFBundleName Translator" "$APP_PATH/Contents/Info.plist" 2>/dev/null || true
/usr/libexec/PlistBuddy -c "Set :CFBundleDisplayName 'Real-Time Translator'" "$APP_PATH/Contents/Info.plist" 2>/dev/null || true
/usr/libexec/PlistBuddy -c "Add :CFBundleIdentifier string com.translator.app" "$APP_PATH/Contents/Info.plist" 2>/dev/null || true
# Make the launcher a background app (no dock icon) - only Python app shows in dock
/usr/libexec/PlistBuddy -c "Add :LSUIElement bool true" "$APP_PATH/Contents/Info.plist" 2>/dev/null || true

echo ""
echo "✅ Created $APP_PATH"
echo ""
echo "You can now:"
echo "  1. Find 'Translator' in your Applications folder"
echo "  2. Spotlight search: ⌘+Space and type 'Translator'"  
echo "  3. Run from terminal: open '$APP_PATH'"
echo ""
