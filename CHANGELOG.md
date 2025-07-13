# Changelog

All notable changes to LabelForge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Major Web Interface Implementation
- **Complete Streamlit-based web interface** with 6 interactive pages
- **Visual labeling function creation** with guided forms for keyword, regex, and custom functions
- **Interactive model training** with real-time progress and configurable parameters
- **Advanced analysis tools** including performance visualization and conflict detection
- **Data upload support** for CSV, JSON, and text files with drag-and-drop functionality
- **Results browsing** with filtering, sorting, and export capabilities
- **Research-focused design** specifically for academic experimentation
- **CLI integration** with `labelforge web` command
- **Web dependencies** management with `pip install labelforge[web]`
- **Launcher scripts** for easy startup (`start_web.sh`)
- **Demo capabilities** with sample medical dataset

### Technical Improvements
- **Fixed import issues** in web interface module
- **Added missing functions** (`get_registered_lfs`) to `lf.py` module
- **Corrected attribute references** (`.matrix` → `.votes`) for `LFOutput` compatibility
- **Enhanced CLI** with web interface command and help documentation
- **Updated dependencies** with modern data science stack (Streamlit, Plotly, Altair)

### Documentation
- **Comprehensive web interface documentation** with usage guide and examples
- **Updated README** with web interface instructions and quick start guide
- **Implementation summary** documenting technical details and architecture
- **Demo script** for interactive exploration of features
- **Installation guides** for different use cases (basic, web, development)

### Project Structure
- **New web module** under `src/labelforge/web/`
- **Web interface app** (`app.py`) with full functionality
- **Support scripts** for launching and demonstration
- **Updated project configuration** with proper dependency management

## [0.1.0] - 2025-07-12

### Added
- Initial release of LabelForge framework
- Core labeling function API with decorator pattern
- Probabilistic label model with EM algorithm
- Command-line interface with multiple commands
- Example datasets and documentation
- Test suite with coverage reporting
- Research-oriented design and documentation

### Features
- **Labeling Functions**: Decorator-based function registration
- **Label Model**: Generative model for combining weak supervision signals
- **CLI Tools**: `lf-list`, `lf-stats`, `lf-test`, `run` commands
- **Examples**: Medical text and research use cases
- **Documentation**: Sphinx-ready docs with academic focus

### Technical
- **Python 3.8+** compatibility
- **Type hints** throughout codebase
- **Testing** with pytest and coverage
- **Code quality** with black, flake8, mypy
- **Packaging** with modern pyproject.toml

## Web Interface Release Notes

### 🎯 Major Milestone: Interactive Web Interface

The v0.1.0+ release introduces a complete web-based interface that transforms LabelForge from a code-only tool into a comprehensive research platform accessible to both technical and non-technical users.

#### Key Features:

1. **📊 Overview Dashboard**
   - Real-time project status monitoring
   - Sample dataset integration
   - Quick start workflows

2. **📁 Data Management**
   - Multi-format file upload (CSV, JSON, TXT)
   - Data preview and statistics
   - Export/import capabilities

3. **⚙️ Function Builder**
   - Visual labeling function creation
   - Three function types: keyword, regex, custom code
   - Real-time testing and validation

4. **🤖 Model Training**
   - Interactive parameter configuration
   - Live training progress
   - Model performance metrics

5. **📈 Advanced Analytics**
   - Function performance visualization
   - Conflict detection and analysis
   - Coverage and agreement metrics

6. **📋 Results Explorer**
   - Prediction browsing with filtering
   - Confidence-based analysis
   - Multi-format export options

#### Research Impact:

- **Democratizes weak supervision** research for broader academic community
- **Enables rapid prototyping** of labeling strategies
- **Facilitates collaboration** through visual demonstrations
- **Supports systematic studies** with comprehensive analysis tools

#### Technical Excellence:

- **Modern architecture** with Streamlit and Plotly
- **Responsive design** optimized for research workflows
- **Scalable performance** for datasets up to 100K examples
- **Seamless integration** with existing LabelForge core

This release positions LabelForge as a leading platform for weak supervision research and education.
