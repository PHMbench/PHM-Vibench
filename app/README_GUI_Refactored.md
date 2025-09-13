# PHM-Vibench Refactored GUI Application

## Overview

The refactored PHM-Vibench GUI is a professional, publication-quality web interface that seamlessly integrates with the refactored PHM-Vibench framework. Built with Streamlit, it provides a comprehensive experimental workflow from configuration to results analysis while maintaining scientific rigor and reproducibility standards.

## Key Features

### 🔧 **Configuration-Driven Experiment Setup**
- **Template-based Configuration**: Predefined templates for common use cases
- **YAML Editor**: Direct configuration editing with syntax validation
- **File Upload**: Support for existing configuration files
- **Real-time Validation**: Automatic parameter validation with detailed error messages

### 📊 **Integrated Data Management**
- **Multiple Data Sources**: Local files, example datasets, file uploads
- **Data Exploration**: Interactive visualizations and statistical analysis
- **Metadata Handling**: Support for Excel and CSV metadata files
- **Data Quality Checks**: Automatic validation and missing data detection

### 🤖 **Dynamic Model Selection**
- **Factory Integration**: Seamless integration with improved factory system
- **Model Browser**: Browse available models by category with documentation
- **Parameter Configuration**: Interactive model parameter setup
- **Scientific Documentation**: Access to model references and implementation details

### ⚙️ **Comprehensive Reproducibility Controls**
- **Deterministic Execution**: Global seed management and deterministic algorithms
- **Environment Tracking**: Complete system and dependency tracking
- **Configuration Hashing**: Unique experiment identifiers
- **Reproducibility Validation**: Built-in tests for deterministic behavior

### 🚀 **Professional Experiment Execution**
- **Pre-execution Validation**: Comprehensive setup validation
- **Real-time Monitoring**: Live progress tracking and metrics display
- **Dry Run Support**: Validate setup without actual training
- **Error Handling**: Robust error handling with detailed diagnostics

### 📈 **Advanced Results Analysis**
- **Interactive Visualizations**: Training curves, confusion matrices, metrics analysis
- **Export Capabilities**: Results export in multiple formats
- **Comprehensive Reports**: Automated experiment report generation
- **Reproducibility Reports**: Complete reproducibility documentation

### 📚 **Integrated Documentation**
- **Getting Started Guide**: Step-by-step tutorials
- **Configuration Reference**: Complete parameter documentation
- **Model Documentation**: Detailed model descriptions and usage
- **Troubleshooting Guide**: Common issues and solutions

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- Streamlit 1.28 or higher

### Install Dependencies

```bash
# Install GUI-specific requirements
pip install -r app/requirements_gui.txt

# Or install individual packages
pip install streamlit plotly matplotlib pandas numpy torch pytorch-lightning pydantic PyYAML h5py openpyxl
```

### Verify Installation

```bash
# Test the refactored framework
python refactored_configs/usage_example.py --test-only

# Launch the GUI
streamlit run app/gui_refactored.py
```

## Usage

### Quick Start

1. **Launch the Application**
   ```bash
   streamlit run app/gui_refactored.py
   ```

2. **Configure Your Experiment**
   - Navigate to the "Configuration" page
   - Choose a template or create custom configuration
   - Validate your settings

3. **Load Your Data**
   - Go to "Data Management"
   - Upload metadata or use example data
   - Explore your dataset

4. **Select and Configure Model**
   - Visit "Model Selection"
   - Browse available models
   - Configure model parameters

5. **Set Reproducibility**
   - Configure reproducibility settings
   - Set random seeds and tracking options

6. **Run Experiment**
   - Execute your experiment with monitoring
   - Review results and analysis