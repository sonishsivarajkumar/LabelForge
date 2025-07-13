# LabelForge Web Interface

A modern, interactive web interface for the LabelForge weak supervision framework. Built with Streamlit, this interface provides researchers and practitioners with an intuitive way to experiment with programmatic data labeling.

## Features

### 🎯 Core Functionality
- **Interactive Data Upload**: Support for CSV, JSON, and text file formats
- **Visual Labeling Function Creation**: Create keyword-based, regex, or custom code labeling functions
- **Real-time Model Training**: Train label models with configurable parameters
- **Comprehensive Analysis**: Visualize performance, conflicts, and model quality
- **Export Results**: Download predictions in multiple formats

### 📊 Visualizations
- Coverage analysis for labeling functions
- Conflict detection and agreement matrices
- Prediction confidence distributions
- Interactive result browsing with filtering
- Real-time performance metrics

### 🔬 Research-Focused Design
- Built for academic experimentation and research workflows
- Clear documentation of methods and algorithms
- Export capabilities for reproducible research
- Integration with the broader LabelForge ecosystem

## Quick Start

### Installation

1. Install LabelForge with web dependencies:
```bash
pip install labelforge[web]
```

Or install dependencies manually:
```bash
pip install streamlit plotly altair streamlit-aggrid
```

### Launch the Interface

#### Option 1: Using the CLI
```bash
labelforge web
```

#### Option 2: Using the launcher script
```bash
python launch_web.py
```

#### Option 3: Direct Streamlit command
```bash
streamlit run src/labelforge/web/app.py
```

### Basic Workflow

1. **📁 Upload Data**: Load your text examples or use sample datasets
2. **⚙️ Create Functions**: Write labeling functions using the visual interface
3. **🤖 Train Model**: Configure and train the label model
4. **📈 Analyze**: Explore performance metrics and visualizations
5. **📋 Export**: Download results for your ML pipeline

## Interface Overview

### Navigation
The interface is organized into six main sections:

- **📊 Overview**: Project status and quick start guide
- **📁 Data Upload**: Data management and preview
- **⚙️ Labeling Functions**: Create and manage labeling functions
- **🤖 Label Model**: Train and configure the model
- **📈 Analysis**: Performance visualization and conflict analysis
- **📋 Results**: Browse predictions and export data

### Data Upload
Supports multiple input formats:
- **CSV files**: Must contain a 'text' column
- **JSON files**: List of texts or object with 'text' key
- **Text files**: One example per line
- **Manual input**: Direct text entry in the interface

### Labeling Function Creation
Three types of labeling functions:

1. **Keyword Match**: Simple keyword-based labeling
2. **Regex Pattern**: Pattern matching with regular expressions
3. **Custom Code**: Full Python function implementation

### Analysis & Visualization
Comprehensive analysis tools:
- Function coverage and performance metrics
- Pairwise agreement and conflict analysis
- Model confidence distributions
- Interactive result exploration

## Technical Details

### Architecture
- **Frontend**: Streamlit with custom CSS styling
- **Visualization**: Plotly for interactive charts
- **Data Processing**: pandas and numpy for efficiency
- **Integration**: Seamless connection to LabelForge core

### Performance
- Optimized for datasets up to 100K examples
- Lazy loading for large datasets
- Efficient visualization with data sampling
- Background processing for model training

### Extensibility
- Modular design for easy feature additions
- Plugin-ready architecture
- Custom visualization support
- Export format extensibility

## Development

### Structure
```
src/labelforge/web/
├── __init__.py      # Web module initialization
├── app.py           # Main Streamlit application
└── README.md        # This documentation
```

### Adding Features
The web interface is designed to be easily extensible:

1. **New Pages**: Add tabs or sidebar options in `app.py`
2. **Custom Visualizations**: Create new chart types with Plotly
3. **Export Formats**: Extend the export functionality
4. **Analysis Tools**: Add new analysis methods

### Contributing
We welcome contributions to the web interface:
- Bug reports and feature requests
- New visualization types
- Performance improvements
- Documentation enhancements

## Research Applications

### Academic Use Cases
- **Comparative Studies**: Evaluate different weak supervision approaches
- **Algorithm Research**: Test new labeling function combination methods
- **Educational Tool**: Teach weak supervision concepts interactively
- **Rapid Prototyping**: Quickly test labeling strategies

### Example Research Workflows

#### Systematic LF Evaluation
1. Upload research dataset
2. Create multiple labeling functions with different strategies
3. Analyze individual and combined performance
4. Export results for statistical analysis

#### Conflict Analysis Studies
1. Design conflicting labeling functions
2. Visualize agreement patterns
3. Study resolution strategies
4. Document findings with export data

#### Model Comparison Research
1. Train models with different configurations
2. Compare confidence distributions
3. Analyze prediction quality
4. Export for downstream evaluation

## Troubleshooting

### Common Issues

#### Import Errors
```
ImportError: No module named 'streamlit'
```
**Solution**: Install web dependencies with `pip install labelforge[web]`

#### Port Already in Use
```
Error: Port 8501 is already in use
```
**Solution**: Use a different port with `labelforge web --port 8502`

#### Memory Issues with Large Datasets
**Solution**: The interface automatically samples large datasets for visualization

### Performance Tips
- Use data sampling for datasets > 10K examples
- Close unused browser tabs to save memory
- Restart the interface periodically for long sessions

## Future Enhancements

### Planned Features
- Real-time collaboration support
- Advanced model diagnostics
- Integration with external ML platforms
- Custom visualization builder
- Automated report generation

### Community Requests
- Multi-modal data support
- Advanced conflict resolution algorithms
- Integration with popular datasets
- Mobile-responsive design

## Support

For help with the web interface:
- Check the main LabelForge documentation
- Report issues on GitHub
- Join community discussions
- Contact the development team

The web interface represents a significant step forward in making weak supervision accessible to researchers and practitioners. We're excited to see how the community uses it for their projects!
