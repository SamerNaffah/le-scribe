#!/usr/bin/env bash
# setup.sh — one-shot install for meet-transcriber dependencies

set -e

echo "=========================================="
echo "  Meet Transcriber — Dependency Installer"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]; }; then
    echo "ERROR: Python 3.11+ required (found $PYTHON_VERSION)"
    echo "Install via: brew install python@3.11"
    exit 1
fi
echo "Python $PYTHON_VERSION — OK"

# Install Python dependencies
echo ""
echo "Installing Python packages..."
pip install \
    mlx-whisper \
    sounddevice \
    numpy \
    selenium \
    "undetected-chromedriver>=3.5.0" \
    flask \
    reportlab

# Install BlackHole virtual audio device
echo ""
echo "Installing BlackHole 2ch virtual audio device..."
brew install blackhole-2ch
echo "BlackHole 2ch installed."

echo ""
echo "=========================================="
echo "  Post-install steps (manual):"
echo "=========================================="
echo ""
echo "1. Configure macOS Audio MIDI Setup:"
echo "   - Open Spotlight → type 'Audio MIDI Setup' → open it"
echo "   - Click '+' at the bottom left → 'Create Multi-Output Device'"
echo "   - Check 'BlackHole 2ch' AND 'Built-in Output' (or your speakers)"
echo "   - Optional: rename it to 'Meet Audio'"
echo ""
echo "3. Set Multi-Output Device as system output:"
echo "   - System Settings → Sound → Output → select 'Multi-Output Device'"
echo "   (Chrome audio will now flow through BlackHole AND your speakers)"
echo ""
echo "3. Verify ChromeDriver version matches your Chrome:"
echo "   - Check Chrome version: chrome://settings/help"
echo "   - undetected-chromedriver auto-downloads matching ChromeDriver on first run"
echo ""
echo "4. Run the bot:"
echo "   python bot.py --meet https://meet.google.com/xxx-xxxx-xxx --lang auto"
echo ""
echo "Done!"
