#!/bin/bash
#
# Create a lightweight .app launcher that runs the Python script
# This avoids all the bundling issues with PyInstaller/py2app
#

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_NAME="Translator.app"
APP_PATH="/Applications/$APP_NAME"

echo "🔨 Creating Translator.app launcher..."

# Remove old app if exists
rm -rf "$APP_PATH"

# Create app structure
mkdir -p "$APP_PATH/Contents/MacOS"
mkdir -p "$APP_PATH/Contents/Resources"

# Create Info.plist
cat > "$APP_PATH/Contents/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>Translator</string>
    <key>CFBundleIdentifier</key>
    <string>com.translator.app</string>
    <key>CFBundleName</key>
    <string>Translator</string>
    <key>CFBundleDisplayName</key>
    <string>Real-Time Translator</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Translator needs audio access for transcription.</string>
</dict>
</plist>
EOF

# Create launcher script
cat > "$APP_PATH/Contents/MacOS/Translator" << EOF
#!/bin/bash
# Launcher for Real-Time Translator
cd "$PROJECT_DIR"
exec /usr/bin/python3 main.py
EOF

chmod +x "$APP_PATH/Contents/MacOS/Translator"

# Copy icon if exists
if [ -f "$PROJECT_DIR/assets/icon.icns" ]; then
    cp "$PROJECT_DIR/assets/icon.icns" "$APP_PATH/Contents/Resources/AppIcon.icns"
    # Update Info.plist to reference icon
    /usr/libexec/PlistBuddy -c "Add :CFBundleIconFile string AppIcon" "$APP_PATH/Contents/Info.plist" 2>/dev/null || true
fi

echo ""
echo "✅ Created $APP_PATH"
echo ""
echo "You can now:"
echo "  1. Find 'Translator' in your Applications folder"
echo "  2. Spotlight search: ⌘+Space and type 'Translator'"
echo "  3. Run from terminal: open '$APP_PATH'"
echo ""

# Open Applications folder
open /Applications
