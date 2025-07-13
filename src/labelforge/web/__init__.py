"""
LabelForge Web Interface

A Streamlit-based web interface for interactive weak supervision and data labeling.
Designed for researchers and practitioners to easily experiment with labeling functions
and visualize results.
"""

__all__ = ["run_web_interface"]

def run_web_interface():
    """Launch the Streamlit web interface."""
    try:
        from .app import main
        main()
    except ImportError as e:
        print(f"Error importing web interface: {e}")
        print("Make sure to install web dependencies: pip install labelforge[web]")
        raise
