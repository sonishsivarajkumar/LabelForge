#!/usr/bin/env python3
"""
Standalone launcher for LabelForge web interface.
This can be run directly or used as a reference.
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Launch the LabelForge web interface."""
    try:
        # Get the path to the web app
        app_path = Path(__file__).parent / "src" / "labelforge" / "web" / "app.py"
        
        if not app_path.exists():
            print("❌ Web app not found. Make sure you're in the LabelForge directory.")
            sys.exit(1)
        
        print("🚀 Starting LabelForge web interface...")
        print("💡 Press Ctrl+C to stop the server")
        print("🌐 The interface will open in your browser at http://localhost:8501")
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.headless", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down LabelForge web interface...")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you have installed the web dependencies:")
        print("   pip install labelforge[web]")
        print("   or")
        print("   pip install streamlit plotly")


if __name__ == "__main__":
    main()
