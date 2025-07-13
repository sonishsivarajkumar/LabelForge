#!/usr/bin/env python3
"""
Demo script for LabelForge Web Interface

This script demonstrates the key features of the new web interface,
showing how researchers can use it for interactive weak supervision.
"""

import subprocess
import sys
import time
from pathlib import Path


def print_banner():
    """Print a welcome banner."""
    print("=" * 70)
    print("🏷️  LabelForge Web Interface Demo")
    print("=" * 70)
    print()
    print("Welcome to the interactive demonstration of LabelForge's new web interface!")
    print("This interface provides a modern, user-friendly way to experiment with")
    print("weak supervision and programmatic data labeling.")
    print()


def print_features():
    """Print key features."""
    print("🌟 Key Features:")
    print("  📁 Data Upload: Drag-and-drop CSV, JSON, and text files")
    print("  ⚙️  Labeling Functions: Visual creation with guided forms")
    print("  🤖 Model Training: Real-time training with configurable parameters")
    print("  📈 Analysis: Live performance metrics and conflict visualization")
    print("  📋 Export: Download predictions in multiple formats")
    print("  🔬 Research-Focused: Designed for academic experimentation")
    print()


def print_usage_instructions():
    """Print usage instructions."""
    print("📖 How to Use:")
    print("  1. Launch the interface with 'labelforge web' or 'python launch_web.py'")
    print("  2. Navigate to http://localhost:8501 in your browser")
    print("  3. Upload your data or use the sample medical dataset")
    print("  4. Create labeling functions using the visual interface")
    print("  5. Train the label model and analyze results")
    print("  6. Export your predictions for downstream ML pipelines")
    print()


def print_research_workflow():
    """Print example research workflow."""
    print("🔬 Example Research Workflow:")
    print("  1. 📊 Overview: Check project status and load sample data")
    print("  2. 📁 Data Upload: Load your research dataset")
    print("  3. ⚙️ Labeling Functions: Create multiple labeling strategies")
    print("     - Keyword-based functions for domain terms")
    print("     - Regex patterns for structured text")
    print("     - Custom Python functions for complex logic")
    print("  4. 🤖 Label Model: Train with different configurations")
    print("  5. 📈 Analysis: Study function conflicts and coverage")
    print("  6. 📋 Results: Browse predictions and export for evaluation")
    print()


def demonstrate_cli_integration():
    """Show CLI integration."""
    print("💻 CLI Integration:")
    print("  Launch web interface:")
    print("    $ labelforge web")
    print("    $ labelforge web --port 8502  # Custom port")
    print("    $ python launch_web.py        # Direct launcher")
    print()
    print("  Use with existing CLI tools:")
    print("    $ labelforge lf-list          # List functions")
    print("    $ labelforge lf-stats         # Function statistics")
    print("    $ labelforge run              # CLI pipeline")
    print()


def check_dependencies():
    """Check if web dependencies are installed."""
    try:
        import streamlit
        import plotly
        import altair
        print("✅ All web dependencies are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install labelforge[web]")
        return False


def main():
    """Run the demo."""
    print_banner()
    print_features()
    print_usage_instructions()
    print_research_workflow()
    demonstrate_cli_integration()
    
    # Check dependencies
    if not check_dependencies():
        print("Please install the required dependencies first.")
        return
    
    print("🚀 Demo Options:")
    print("  1. Launch web interface now")
    print("  2. Show installation instructions")
    print("  3. Exit")
    print()
    
    try:
        choice = input("Choose an option (1-3): ").strip()
        
        if choice == "1":
            print("🌐 Launching LabelForge web interface...")
            print("📍 Access it at: http://localhost:8501")
            print("💡 Press Ctrl+C to stop")
            print()
            
            # Launch the web interface
            try:
                subprocess.run([sys.executable, "-m", "streamlit", "run", 
                              "src/labelforge/web/app.py"])
            except KeyboardInterrupt:
                print("\n👋 Web interface stopped.")
            except Exception as e:
                print(f"Error launching interface: {e}")
                print("Try: labelforge web")
        
        elif choice == "2":
            print("📦 Installation Instructions:")
            print("  1. Install LabelForge with web support:")
            print("     pip install labelforge[web]")
            print()
            print("  2. Or install dependencies manually:")
            print("     pip install streamlit plotly altair streamlit-aggrid")
            print()
            print("  3. Launch the interface:")
            print("     labelforge web")
            print()
        
        elif choice == "3":
            print("👋 Thanks for checking out LabelForge!")
            print("   Visit: https://github.com/sonishsivarajkumar/LabelForge")
        
        else:
            print("Invalid choice. Exiting.")
    
    except KeyboardInterrupt:
        print("\n👋 Demo cancelled.")


if __name__ == "__main__":
    main()
