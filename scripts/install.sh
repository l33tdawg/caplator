#!/bin/bash
#
# Install Translator as a system command
#

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL_DIR="/usr/local/bin"
SCRIPT_NAME="translator"

echo "📦 Installing Translator..."
echo ""

# Create launcher script
cat > /tmp/translator_launcher << EOF
#!/bin/bash
# Real-Time Audio Translator Launcher
cd "$PROJECT_DIR"
exec python3 main.py "\$@"
EOF

# Install to /usr/local/bin
sudo mv /tmp/translator_launcher "$INSTALL_DIR/$SCRIPT_NAME"
sudo chmod +x "$INSTALL_DIR/$SCRIPT_NAME"

echo "✅ Installed successfully!"
echo ""
echo "To run the translator, simply type:"
echo "  translator"
echo ""
echo "Or to run from the project directory:"
echo "  cd $PROJECT_DIR && python3 main.py"
echo ""
