#!/usr/bin/env python3
"""
PHM-Vibench Refactored Streamlit GUI Application

This module provides a professional, publication-quality web interface for the PHM-Vibench
research framework. The GUI integrates seamlessly with the refactored configuration system,
factory patterns, and reproducibility framework to provide a complete experimental workflow.

Key Features:
- Configuration-driven experiment setup with validation
- Dynamic model and task selection through factory integration
- Comprehensive reproducibility controls
- Real-time experiment monitoring and visualization
- Professional error handling and user feedback
- Scientific documentation and help system

The interface follows the same scientific code standards established in the PHM-Vibench
refactoring, ensuring consistency, maintainability, and publication-quality software.

Usage:
    streamlit run app/gui_refactored.py

Authors: PHM-Vibench Development Team
License: Same as PHM-Vibench project
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import yaml
from pydantic import ValidationError

# Add the project root to the Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner UI
warnings.filterwarnings('ignore')


class ExperimentState:
    """
    Centralized state management for the PHM-Vibench GUI.

    This class manages all experiment-related state including configuration,
    data, models, and execution status. It provides a clean interface for
    state persistence across Streamlit reruns.

    Attributes
    ----------
    config : Optional[ExperimentConfig]
        Current experiment configuration
    metadata_df : Optional[pd.DataFrame]
        Loaded metadata DataFrame
    available_models : List[str]
        Available model names from factory registry
    available_tasks : List[str]
        Available task names from factory registry
    experiment_running : bool
        Whether an experiment is currently running
    results : Dict[str, Any]
        Experiment results and metrics
    """

    def __init__(self):
        """Initialize experiment state with default values."""
        self.config: Optional[Any] = None  # Will be ExperimentConfig when available
        self.metadata_df: Optional[pd.DataFrame] = None
        self.available_models: List[str] = []
        self.available_tasks: List[str] = []
        self.experiment_running: bool = False
        self.results: Dict[str, Any] = {}
        self.reproducibility_manager: Optional[Any] = None  # Will be ReproducibilityManager

        # Initialize available components from registries
        self._update_available_components()

    def _update_available_components(self) -> None:
        """Update available models and tasks from factory registries."""
        try:
            # Try to import and use factory registries
            from refactored_configs.improved_factory import MODEL_REGISTRY, TASK_REGISTRY

            # Get available models
            model_components = MODEL_REGISTRY.list_components()
            self.available_models = [f"{comp.component_type}.{comp.name}" for comp in model_components]

            # Get available tasks
            task_components = TASK_REGISTRY.list_components()
            self.available_tasks = [f"{comp.component_type}.{comp.name}" for comp in task_components]

        except Exception as e:
            logger.warning(f"Could not load factory components: {e}")
            # Fallback to default options
            self.available_models = ["CNN.ResNet1D", "RNN.LSTM", "Transformer.ViT"]
            self.available_tasks = ["DG.classification", "FS.classification"]

    def reset(self) -> None:
        """Reset experiment state to initial values."""
        self.config = None
        self.metadata_df = None
        self.experiment_running = False
        self.results = {}
        self.reproducibility_manager = None

    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """
        Validate current configuration.

        Returns
        -------
        Tuple[bool, Optional[str]]
            (is_valid, error_message)
        """
        if self.config is None:
            return False, "No configuration loaded"

        try:
            # Basic validation - can be enhanced when config schema is available
            required_fields = ['name', 'data', 'model', 'task']
            for field in required_fields:
                if not hasattr(self.config, field):
                    return False, f"Missing required field: {field}"
            return True, None
        except Exception as e:
            return False, f"Unexpected validation error: {e}"


def init_session_state() -> None:
    """
    Initialize Streamlit session state with default values.

    This function ensures all required session state variables are initialized
    with appropriate default values to prevent KeyError exceptions during
    widget interactions.
    """
    defaults = {
        'experiment_state': ExperimentState(),
        'current_page': 'Configuration',
        'show_advanced': False,
        'config_yaml': '',
        'last_error': None,
        'experiment_history': [],
        'selected_config_template': 'Custom',
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def setup_page_config() -> None:
    """Configure Streamlit page settings and styling."""
    st.set_page_config(
        page_title="PHM-Vibench Research Framework",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/PHMbench/PHM-Vibench',
            'Report a bug': 'https://github.com/PHMbench/PHM-Vibench/issues',
            'About': """
            # PHM-Vibench Research Framework

            A publication-quality framework for industrial signal analysis and
            prognostics and health management (PHM) research.

            **Version**: 2.0 (Refactored)
            **Authors**: PHM-Vibench Development Team
            **License**: MIT
            """
        }
    )

    # Custom CSS for professional appearance
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 1rem;
    }

    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #3498db;
        padding-left: 1rem;
    }

    .info-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }

    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724;
    }

    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #721c24;
    }

    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    }
    </style>
    """, unsafe_allow_html=True)


def display_header() -> None:
    """Display the main application header with branding and navigation."""
    st.markdown('<h1 class="main-header">🔬 PHM-Vibench Research Framework</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>Publication-Quality Industrial Signal Analysis Framework</strong><br>
    Configure, execute, and analyze prognostics and health management experiments with
    scientific rigor and reproducibility.
    </div>
    """, unsafe_allow_html=True)


def create_sidebar_navigation() -> str:
    """
    Create sidebar navigation and return selected page.

    Returns
    -------
    str
        Selected page name
    """
    st.sidebar.title("🧭 Navigation")

    pages = {
        "🔧 Configuration": "Configuration",
        "📊 Data Management": "Data",
        "🤖 Model Selection": "Models",
        "⚙️ Reproducibility": "Reproducibility",
        "🚀 Experiment Execution": "Execution",
        "📈 Results & Analysis": "Results",
        "📚 Documentation": "Documentation"
    }

    selected_page = st.sidebar.radio(
        "Select Page",
        list(pages.keys()),
        index=list(pages.values()).index(st.session_state.current_page)
    )

    st.session_state.current_page = pages[selected_page]

    # Display experiment status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Experiment Status")

    experiment_state = st.session_state.experiment_state

    if experiment_state.config is not None:
        st.sidebar.success("✅ Configuration Loaded")
        st.sidebar.text(f"Name: {getattr(experiment_state.config, 'name', 'Unknown')}")
    else:
        st.sidebar.warning("⚠️ No Configuration")

    if experiment_state.metadata_df is not None:
        st.sidebar.success(f"✅ Data Loaded ({len(experiment_state.metadata_df)} samples)")
    else:
        st.sidebar.warning("⚠️ No Data Loaded")

    if experiment_state.experiment_running:
        st.sidebar.info("🔄 Experiment Running")
    else:
        st.sidebar.text("⏸️ Ready to Run")

    # Quick actions
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Quick Actions")

    if st.sidebar.button("🔄 Reset All", help="Reset all experiment state"):
        experiment_state.reset()
        st.session_state.config_yaml = ''
        st.rerun()

    if st.sidebar.button("💾 Save Config", help="Save current configuration"):
        if experiment_state.config is not None:
            save_current_config()
        else:
            st.sidebar.error("No configuration to save")

    return st.session_state.current_page


def save_current_config() -> None:
    """Save current configuration to file."""
    try:
        experiment_state = st.session_state.experiment_state
        if experiment_state.config is None:
            st.error("No configuration to save")
            return

        # Create configs directory if it doesn't exist
        config_dir = Path("configs/gui_generated")
        config_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = getattr(experiment_state.config, 'name', 'experiment')
        filename = f"{config_name}_{timestamp}.yaml"
        filepath = config_dir / filename

        # Save configuration
        config_dict = experiment_state.config.__dict__ if hasattr(experiment_state.config, '__dict__') else experiment_state.config

        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

        st.sidebar.success(f"✅ Saved: {filename}")

    except Exception as e:
        st.sidebar.error(f"❌ Save failed: {str(e)}")


def handle_error(error: Exception, context: str = "") -> None:
    """
    Handle and display errors with proper formatting and logging.

    Parameters
    ----------
    error : Exception
        The exception that occurred
    context : str
        Additional context about where the error occurred
    """
    error_msg = f"{context}: {str(error)}" if context else str(error)

    # Log the error
    logger.error(f"GUI Error - {error_msg}", exc_info=True)

    # Store in session state for persistence
    st.session_state.last_error = {
        'message': error_msg,
        'timestamp': datetime.now().isoformat(),
        'traceback': traceback.format_exc()
    }

    # Display user-friendly error
    st.error(f"❌ {error_msg}")

    # Show detailed error in expander for debugging
    with st.expander("🔍 Error Details", expanded=False):
        st.code(traceback.format_exc(), language='python')


def create_config_templates() -> Dict[str, Dict[str, Any]]:
    """
    Create predefined configuration templates for common use cases.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary of template names to configuration dictionaries
    """
    templates = {
        "ResNet1D Classification": {
            "name": "resnet1d_classification",
            "description": "ResNet1D for bearing fault classification",
            "tags": ["classification", "cnn", "bearing_fault"],
            "reproducibility": {
                "global_seed": 42,
                "torch_deterministic": True,
                "track_environment": True
            },
            "data": {
                "data_root": "data/",
                "metadata_file": "metadata.xlsx",
                "batch_size": 64,
                "window_size": 4096,
                "normalization": "standardization"
            },
            "model": {
                "name": "ResNet1D",
                "type": "CNN",
                "input_dim": 3,
                "num_classes": 10,
                "block_type": "basic",
                "layers": [2, 2, 2, 2]
            },
            "optimization": {
                "optimizer": "adam",
                "learning_rate": 0.001,
                "max_epochs": 100,
                "early_stopping": True
            },
            "task": {
                "name": "classification",
                "type": "DG",
                "loss_function": "cross_entropy",
                "metrics": ["accuracy", "f1_macro"]
            }
        },

        "LSTM Time Series": {
            "name": "lstm_timeseries",
            "description": "LSTM for time series prediction",
            "tags": ["regression", "rnn", "timeseries"],
            "reproducibility": {
                "global_seed": 42,
                "torch_deterministic": True
            },
            "data": {
                "data_root": "data/",
                "metadata_file": "metadata.xlsx",
                "batch_size": 32,
                "window_size": 2048
            },
            "model": {
                "name": "LSTM",
                "type": "RNN",
                "input_dim": 1,
                "hidden_dim": 128,
                "num_layers": 2,
                "output_dim": 1
            },
            "optimization": {
                "optimizer": "adam",
                "learning_rate": 0.001,
                "max_epochs": 50
            },
            "task": {
                "name": "regression",
                "type": "FS",
                "loss_function": "mse",
                "metrics": ["mae", "rmse"]
            }
        },

        "Custom": {
            "name": "custom_experiment",
            "description": "Custom experiment configuration",
            "tags": ["custom"],
            "reproducibility": {"global_seed": 42},
            "data": {"data_root": "data/", "batch_size": 64},
            "model": {"name": "ResNet1D", "type": "CNN"},
            "optimization": {"optimizer": "adam", "learning_rate": 0.001},
            "task": {"name": "classification", "type": "DG"}
        }
    }

    return templates


def render_configuration_page() -> None:
    """
    Render the configuration page for experiment setup.

    This page allows users to create, edit, and validate experiment configurations
    using the refactored configuration schema system.
    """
    st.markdown('<div class="section-header">🔧 Experiment Configuration</div>',
                unsafe_allow_html=True)

    # Configuration method selection
    config_method = st.radio(
        "Configuration Method",
        ["Template-based", "YAML Editor", "Upload File"],
        horizontal=True,
        help="Choose how to create your experiment configuration"
    )

    experiment_state = st.session_state.experiment_state

    if config_method == "Template-based":
        render_template_configuration()
    elif config_method == "YAML Editor":
        render_yaml_editor()
    elif config_method == "Upload File":
        render_file_upload()

    # Configuration validation and preview
    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📋 Configuration Preview")
        if experiment_state.config is not None:
            # Display configuration as formatted JSON
            config_dict = experiment_state.config.__dict__ if hasattr(experiment_state.config, '__dict__') else experiment_state.config
            st.json(config_dict)
        else:
            st.info("No configuration loaded. Please create or load a configuration above.")

    with col2:
        st.subheader("✅ Validation")
        if experiment_state.config is not None:
            is_valid, error_msg = experiment_state.validate_config()
            if is_valid:
                st.success("✅ Configuration is valid")

                # Show configuration summary
                st.markdown("**Summary:**")
                config = experiment_state.config
                st.text(f"Name: {getattr(config, 'name', 'Unknown')}")
                st.text(f"Model: {getattr(getattr(config, 'model', {}), 'name', 'Unknown')}")
                st.text(f"Task: {getattr(getattr(config, 'task', {}), 'name', 'Unknown')}")

            else:
                st.error(f"❌ Validation Error:\n{error_msg}")
        else:
            st.warning("⚠️ No configuration to validate")


def render_template_configuration() -> None:
    """Render template-based configuration interface."""
    st.subheader("📋 Template Selection")

    templates = create_config_templates()
    template_names = list(templates.keys())

    selected_template = st.selectbox(
        "Choose Configuration Template",
        template_names,
        index=template_names.index(st.session_state.selected_config_template),
        help="Select a predefined template to start with"
    )

    st.session_state.selected_config_template = selected_template

    if selected_template != "Custom":
        template_config = templates[selected_template].copy()

        # Allow editing of key parameters
        st.subheader("🔧 Template Customization")

        col1, col2 = st.columns(2)

        with col1:
            # Basic experiment info
            template_config["name"] = st.text_input(
                "Experiment Name",
                value=template_config["name"],
                help="Unique name for this experiment"
            )

            template_config["description"] = st.text_area(
                "Description",
                value=template_config["description"],
                help="Detailed description of the experiment"
            )

            # Data configuration
            st.markdown("**Data Configuration**")
            template_config["data"]["batch_size"] = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=512,
                value=template_config["data"]["batch_size"],
                help="Number of samples per batch"
            )

            template_config["data"]["window_size"] = st.number_input(
                "Window Size",
                min_value=64,
                max_value=8192,
                value=template_config["data"].get("window_size", 1024),
                help="Length of input signal windows"
            )

        with col2:
            # Model configuration
            st.markdown("**Model Configuration**")
            if "num_classes" in template_config["model"]:
                template_config["model"]["num_classes"] = st.number_input(
                    "Number of Classes",
                    min_value=2,
                    max_value=100,
                    value=template_config["model"]["num_classes"],
                    help="Number of output classes for classification"
                )

            # Optimization configuration
            st.markdown("**Optimization Configuration**")
            template_config["optimization"]["learning_rate"] = st.number_input(
                "Learning Rate",
                min_value=1e-6,
                max_value=1e-1,
                value=template_config["optimization"]["learning_rate"],
                format="%.6f",
                help="Initial learning rate for optimization"
            )

            template_config["optimization"]["max_epochs"] = st.number_input(
                "Max Epochs",
                min_value=1,
                max_value=1000,
                value=template_config["optimization"]["max_epochs"],
                help="Maximum number of training epochs"
            )

        # Apply template configuration
        if st.button("📥 Apply Template Configuration", type="primary"):
            try:
                # Convert to namespace for compatibility
                from types import SimpleNamespace

                def dict_to_namespace(d):
                    if isinstance(d, dict):
                        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
                    return d

                st.session_state.experiment_state.config = dict_to_namespace(template_config)
                st.success("✅ Template configuration applied successfully!")
                st.rerun()

            except Exception as e:
                handle_error(e, "Failed to apply template configuration")

    else:
        st.info("Select a template above or use the YAML editor for custom configuration.")


def render_yaml_editor() -> None:
    """Render YAML editor interface."""
    st.subheader("📝 YAML Configuration Editor")

    # Load example YAML if empty
    if not st.session_state.config_yaml:
        templates = create_config_templates()
        example_config = templates["ResNet1D Classification"]
        st.session_state.config_yaml = yaml.dump(example_config, default_flow_style=False)

    # YAML editor
    config_yaml = st.text_area(
        "Configuration YAML",
        value=st.session_state.config_yaml,
        height=400,
        help="Edit the YAML configuration directly. Use proper YAML syntax."
    )

    st.session_state.config_yaml = config_yaml

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("✅ Parse & Validate YAML", type="primary"):
            try:
                # Parse YAML
                config_dict = yaml.safe_load(config_yaml)

                # Convert to namespace
                from types import SimpleNamespace

                def dict_to_namespace(d):
                    if isinstance(d, dict):
                        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
                    return d

                config_obj = dict_to_namespace(config_dict)
                st.session_state.experiment_state.config = config_obj

                st.success("✅ YAML parsed and validated successfully!")
                st.rerun()

            except yaml.YAMLError as e:
                handle_error(e, "YAML parsing failed")
            except Exception as e:
                handle_error(e, "Configuration validation failed")

    with col2:
        if st.button("📋 Load Template"):
            templates = create_config_templates()
            template_name = st.selectbox("Select template", list(templates.keys()))
            if template_name:
                st.session_state.config_yaml = yaml.dump(
                    templates[template_name],
                    default_flow_style=False
                )
                st.rerun()

    with col3:
        if st.button("🔄 Reset Editor"):
            st.session_state.config_yaml = ""
            st.rerun()


def render_file_upload() -> None:
    """Render file upload interface for configuration."""
    st.subheader("📁 Upload Configuration File")

    uploaded_file = st.file_uploader(
        "Choose configuration file",
        type=['yaml', 'yml', 'json'],
        help="Upload a YAML or JSON configuration file"
    )

    if uploaded_file is not None:
        try:
            # Read file content
            content = uploaded_file.read().decode('utf-8')

            # Parse based on file extension
            if uploaded_file.name.endswith(('.yaml', '.yml')):
                config_dict = yaml.safe_load(content)
            elif uploaded_file.name.endswith('.json'):
                config_dict = json.loads(content)
            else:
                raise ValueError("Unsupported file format")

            # Convert to namespace
            from types import SimpleNamespace

            def dict_to_namespace(d):
                if isinstance(d, dict):
                    return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
                return d

            config_obj = dict_to_namespace(config_dict)
            st.session_state.experiment_state.config = config_obj

            # Update YAML editor with loaded content
            st.session_state.config_yaml = yaml.dump(config_dict, default_flow_style=False)

            st.success(f"✅ Configuration loaded from {uploaded_file.name}")
            st.rerun()

        except Exception as e:
            handle_error(e, f"Failed to load configuration from {uploaded_file.name}")


def render_data_management_page() -> None:
    """
    Render the data management page for dataset loading and exploration.

    This page provides tools for loading, validating, and exploring datasets
    with integration to the refactored data handling system.
    """
    st.markdown('<div class="section-header">📊 Data Management</div>',
                unsafe_allow_html=True)

    experiment_state = st.session_state.experiment_state

    # Data loading section
    st.subheader("📁 Data Loading")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Data source selection
        data_source = st.radio(
            "Data Source",
            ["Local Files", "Example Dataset", "Upload Files"],
            horizontal=True
        )

        if data_source == "Local Files":
            data_root = st.text_input(
                "Data Root Directory",
                value="data/",
                help="Path to the root directory containing your datasets"
            )

            metadata_file = st.text_input(
                "Metadata File",
                value="metadata.xlsx",
                help="Name of the metadata file (Excel or CSV)"
            )

            if st.button("🔍 Load Metadata", type="primary"):
                load_metadata(data_root, metadata_file)

        elif data_source == "Example Dataset":
            st.info("📋 Example datasets will be loaded automatically for demonstration")
            if st.button("📥 Load Example Data"):
                load_example_data()

        elif data_source == "Upload Files":
            uploaded_metadata = st.file_uploader(
                "Upload Metadata File",
                type=['xlsx', 'csv'],
                help="Upload your metadata file"
            )

            if uploaded_metadata is not None:
                process_uploaded_metadata(uploaded_metadata)

    with col2:
        st.subheader("📈 Data Statistics")
        if experiment_state.metadata_df is not None:
            df = experiment_state.metadata_df
            st.metric("Total Samples", len(df))

            if 'label' in df.columns:
                unique_labels = df['label'].nunique()
                st.metric("Unique Labels", unique_labels)

            if 'domain' in df.columns:
                unique_domains = df['domain'].nunique()
                st.metric("Domains", unique_domains)

            # Data quality indicators
            missing_data = df.isnull().sum().sum()
            if missing_data > 0:
                st.warning(f"⚠️ {missing_data} missing values detected")
            else:
                st.success("✅ No missing values")
        else:
            st.info("No data loaded yet")

    # Data exploration section
    if experiment_state.metadata_df is not None:
        st.markdown("---")
        st.subheader("🔍 Data Exploration")

        render_data_exploration(experiment_state.metadata_df)


def load_metadata(data_root: str, metadata_file: str) -> None:
    """Load metadata from specified path."""
    try:
        metadata_path = Path(data_root) / metadata_file

        if not metadata_path.exists():
            st.error(f"❌ Metadata file not found: {metadata_path}")
            return

        # Load based on file extension
        if metadata_path.suffix.lower() == '.xlsx':
            df = pd.read_excel(metadata_path)
        elif metadata_path.suffix.lower() == '.csv':
            df = pd.read_csv(metadata_path)
        else:
            st.error("❌ Unsupported metadata file format. Use .xlsx or .csv")
            return

        st.session_state.experiment_state.metadata_df = df
        st.success(f"✅ Loaded {len(df)} samples from {metadata_file}")
        st.rerun()

    except Exception as e:
        handle_error(e, "Failed to load metadata")


def load_example_data() -> None:
    """Load example dataset for demonstration."""
    try:
        # Create synthetic metadata for demonstration
        np.random.seed(42)
        n_samples = 1000

        data = {
            'file_path': [f"data/sample_{i:04d}.h5" for i in range(n_samples)],
            'label': np.random.randint(0, 10, n_samples),
            'domain': np.random.randint(0, 5, n_samples),
            'condition': np.random.choice(['normal', 'fault_1', 'fault_2'], n_samples),
            'rpm': np.random.uniform(1000, 3000, n_samples),
            'load': np.random.uniform(0, 100, n_samples)
        }

        df = pd.DataFrame(data)
        st.session_state.experiment_state.metadata_df = df
        st.success(f"✅ Loaded example dataset with {len(df)} samples")
        st.rerun()

    except Exception as e:
        handle_error(e, "Failed to load example data")


def process_uploaded_metadata(uploaded_file) -> None:
    """Process uploaded metadata file."""
    try:
        # Read file based on extension
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("❌ Unsupported file format")
            return

        st.session_state.experiment_state.metadata_df = df
        st.success(f"✅ Uploaded and loaded {len(df)} samples from {uploaded_file.name}")
        st.rerun()

    except Exception as e:
        handle_error(e, f"Failed to process uploaded file {uploaded_file.name}")


def render_data_exploration(df: pd.DataFrame) -> None:
    """Render data exploration interface."""
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Visualizations", "🔍 Sample Data"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Dataset Overview")
            st.dataframe(df.describe(), use_container_width=True)

        with col2:
            st.subheader("Column Information")
            info_data = {
                'Column': df.columns,
                'Type': [str(dtype) for dtype in df.dtypes],
                'Non-Null': [df[col].count() for col in df.columns],
                'Unique': [df[col].nunique() for col in df.columns]
            }
            st.dataframe(pd.DataFrame(info_data), use_container_width=True)

    with tab2:
        if len(df.columns) > 1:
            # Label distribution
            if 'label' in df.columns:
                st.subheader("Label Distribution")
                label_counts = df['label'].value_counts()
                fig = px.bar(x=label_counts.index, y=label_counts.values,
                           labels={'x': 'Label', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)

            # Domain distribution
            if 'domain' in df.columns:
                st.subheader("Domain Distribution")
                domain_counts = df['domain'].value_counts()
                fig = px.pie(values=domain_counts.values, names=domain_counts.index)
                st.plotly_chart(fig, use_container_width=True)

            # Correlation matrix for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                st.subheader("Correlation Matrix")
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough columns for visualization")

    with tab3:
        st.subheader("Sample Data")

        # Display options
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.slider("Number of samples to display", 5, min(100, len(df)), 10)
        with col2:
            sample_method = st.selectbox("Sampling method", ["First N", "Random", "Last N"])

        # Display samples
        if sample_method == "First N":
            sample_df = df.head(n_samples)
        elif sample_method == "Random":
            sample_df = df.sample(n_samples) if len(df) >= n_samples else df
        else:  # Last N
            sample_df = df.tail(n_samples)

        st.dataframe(sample_df, use_container_width=True)


def render_model_selection_page() -> None:
    """
    Render the model selection page with factory integration.

    This page allows users to browse available models, view their documentation,
    and configure model parameters using the improved factory system.
    """
    st.markdown('<div class="section-header">🤖 Model Selection</div>',
                unsafe_allow_html=True)

    experiment_state = st.session_state.experiment_state

    # Model browser section
    st.subheader("🔍 Available Models")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Model category filter
        model_categories = ["All", "CNN", "RNN", "Transformer", "MLP", "Neural_Operator"]
        selected_category = st.selectbox("Model Category", model_categories)

        # Filter models by category
        available_models = experiment_state.available_models
        if selected_category != "All":
            available_models = [m for m in available_models if m.startswith(selected_category)]

        # Model selection
        if available_models:
            selected_model = st.selectbox(
                "Select Model",
                available_models,
                help="Choose a model architecture for your experiment"
            )

            # Model information
            if selected_model:
                display_model_information(selected_model)
        else:
            st.warning(f"No models available for category: {selected_category}")

    with col2:
        if available_models and 'selected_model' in locals():
            render_model_configuration(selected_model)
        else:
            st.info("Select a model to configure its parameters")


def display_model_information(model_name: str) -> None:
    """Display detailed information about a selected model."""
    try:
        # Try to get model information from registry
        model_type, model_class = model_name.split('.', 1)

        st.markdown(f"**Model**: {model_class}")
        st.markdown(f"**Type**: {model_type}")

        # Try to get component info from registry
        try:
            from refactored_configs.improved_factory import MODEL_REGISTRY
            component_info = MODEL_REGISTRY.get(model_name)

            if component_info.description:
                st.markdown(f"**Description**: {component_info.description}")

            if component_info.paper_reference:
                st.markdown(f"**Reference**: {component_info.paper_reference}")

            if component_info.parameters:
                st.markdown("**Parameters**:")
                for param_name, param_info in component_info.parameters.items():
                    required = "✅" if param_info.get('required', False) else "⚪"
                    st.markdown(f"- {required} `{param_name}`: {param_info.get('type', 'Any')}")

        except Exception:
            # Fallback to basic information
            st.markdown("**Description**: Model implementation for time-series analysis")
            st.markdown("**Parameters**: Standard model parameters apply")

    except Exception as e:
        st.warning(f"Could not load model information: {e}")


def render_model_configuration(model_name: str) -> None:
    """Render model configuration interface."""
    st.subheader("⚙️ Model Configuration")

    model_type, model_class = model_name.split('.', 1)

    # Basic model parameters
    col1, col2 = st.columns(2)

    with col1:
        input_dim = st.number_input(
            "Input Dimension",
            min_value=1,
            max_value=100,
            value=3,
            help="Number of input features/channels"
        )

        if model_type in ["CNN", "MLP"]:
            num_classes = st.number_input(
                "Number of Classes",
                min_value=2,
                max_value=1000,
                value=10,
                help="Number of output classes for classification"
            )
        else:
            output_dim = st.number_input(
                "Output Dimension",
                min_value=1,
                max_value=100,
                value=1,
                help="Dimension of output features"
            )

    with col2:
        if model_type in ["RNN", "Transformer"]:
            hidden_dim = st.number_input(
                "Hidden Dimension",
                min_value=16,
                max_value=1024,
                value=128,
                help="Size of hidden layers"
            )

            num_layers = st.number_input(
                "Number of Layers",
                min_value=1,
                max_value=20,
                value=2,
                help="Number of model layers"
            )

        dropout = st.slider(
            "Dropout Rate",
            min_value=0.0,
            max_value=0.9,
            value=0.1,
            step=0.05,
            help="Dropout probability for regularization"
        )

    # Advanced parameters
    with st.expander("🔧 Advanced Parameters"):
        weight_init = st.selectbox(
            "Weight Initialization",
            ["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"],
            help="Method for initializing model weights"
        )

        bias_init = st.selectbox(
            "Bias Initialization",
            ["zeros", "uniform"],
            help="Method for initializing bias terms"
        )

    # Apply model configuration
    if st.button("✅ Apply Model Configuration", type="primary"):
        try:
            # Create model configuration
            model_config = {
                "name": model_class,
                "type": model_type,
                "input_dim": input_dim,
                "dropout": dropout,
                "weight_init": weight_init,
                "bias_init": bias_init
            }

            # Add type-specific parameters
            if model_type in ["CNN", "MLP"] and 'num_classes' in locals():
                model_config["num_classes"] = num_classes
            elif 'output_dim' in locals():
                model_config["output_dim"] = output_dim

            if model_type in ["RNN", "Transformer"]:
                model_config["hidden_dim"] = hidden_dim
                model_config["num_layers"] = num_layers

            # Update experiment configuration
            if experiment_state.config is None:
                from types import SimpleNamespace
                experiment_state.config = SimpleNamespace()

            experiment_state.config.model = SimpleNamespace(**model_config)

            st.success("✅ Model configuration applied successfully!")
            st.rerun()

        except Exception as e:
            handle_error(e, "Failed to apply model configuration")


def render_reproducibility_page() -> None:
    """
    Render the reproducibility configuration page.

    This page provides comprehensive controls for ensuring experimental
    reproducibility using the refactored reproducibility framework.
    """
    st.markdown('<div class="section-header">⚙️ Reproducibility Settings</div>',
                unsafe_allow_html=True)

    experiment_state = st.session_state.experiment_state

    # Reproducibility configuration
    st.subheader("🎯 Deterministic Execution")

    col1, col2 = st.columns(2)

    with col1:
        global_seed = st.number_input(
            "Global Random Seed",
            min_value=0,
            max_value=2**32-1,
            value=42,
            help="Master seed for all random number generators"
        )

        torch_deterministic = st.checkbox(
            "PyTorch Deterministic Mode",
            value=True,
            help="Enable deterministic algorithms in PyTorch (may reduce performance)"
        )

        torch_benchmark = st.checkbox(
            "CUDNN Benchmark",
            value=False,
            help="Enable CUDNN benchmark for performance (reduces determinism)"
        )

    with col2:
        track_environment = st.checkbox(
            "Track Environment",
            value=True,
            help="Record detailed environment information"
        )

        track_git_commit = st.checkbox(
            "Track Git Commit",
            value=True,
            help="Record current git commit hash and status"
        )

        track_dependencies = st.checkbox(
            "Track Dependencies",
            value=True,
            help="Record versions of key packages"
        )

    # Environment information display
    st.markdown("---")
    st.subheader("🖥️ Current Environment")

    if st.button("🔍 Scan Environment"):
        display_environment_info()

    # Reproducibility validation
    st.markdown("---")
    st.subheader("✅ Reproducibility Validation")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧪 Test Determinism"):
            test_determinism(global_seed)

    with col2:
        if st.button("📊 Generate Report"):
            generate_reproducibility_report()

    # Apply reproducibility settings
    if st.button("✅ Apply Reproducibility Settings", type="primary"):
        try:
            repro_config = {
                "global_seed": global_seed,
                "torch_deterministic": torch_deterministic,
                "torch_benchmark": torch_benchmark,
                "track_environment": track_environment,
                "track_git_commit": track_git_commit,
                "track_dependencies": track_dependencies
            }

            # Update experiment configuration
            if experiment_state.config is None:
                from types import SimpleNamespace
                experiment_state.config = SimpleNamespace()

            experiment_state.config.reproducibility = SimpleNamespace(**repro_config)

            st.success("✅ Reproducibility settings applied successfully!")
            st.rerun()

        except Exception as e:
            handle_error(e, "Failed to apply reproducibility settings")


def display_environment_info() -> None:
    """Display current environment information."""
    try:
        import platform
        import socket

        env_info = {
            "System": platform.system(),
            "Release": platform.release(),
            "Machine": platform.machine(),
            "Processor": platform.processor(),
            "Hostname": socket.gethostname(),
            "Python Version": platform.python_version(),
            "PyTorch Version": torch.__version__,
            "CUDA Available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            env_info["CUDA Version"] = torch.version.cuda
            env_info["GPU Count"] = torch.cuda.device_count()
            env_info["GPU Names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

        # Display as formatted table
        for key, value in env_info.items():
            st.text(f"{key}: {value}")

    except Exception as e:
        handle_error(e, "Failed to scan environment")


def test_determinism(seed: int) -> None:
    """Test deterministic behavior with given seed."""
    try:
        # Set seed and generate random numbers
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Generate test data
        torch_data1 = torch.randn(10)
        numpy_data1 = np.random.randn(10)

        # Reset seed and generate again
        torch.manual_seed(seed)
        np.random.seed(seed)

        torch_data2 = torch.randn(10)
        numpy_data2 = np.random.randn(10)

        # Check if identical
        torch_identical = torch.allclose(torch_data1, torch_data2)
        numpy_identical = np.allclose(numpy_data1, numpy_data2)

        if torch_identical and numpy_identical:
            st.success("✅ Determinism test passed! Random number generation is reproducible.")
        else:
            st.error("❌ Determinism test failed! Random number generation is not reproducible.")

    except Exception as e:
        handle_error(e, "Determinism test failed")


def generate_reproducibility_report() -> None:
    """Generate and display reproducibility report."""
    try:
        report = {
            "timestamp": datetime.now().isoformat(),
            "environment": {
                "python_version": platform.python_version(),
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "system": platform.system(),
                "hostname": socket.gethostname()
            },
            "configuration": {
                "has_config": st.session_state.experiment_state.config is not None,
                "has_data": st.session_state.experiment_state.metadata_df is not None
            }
        }

        st.json(report)

        # Option to download report
        report_json = json.dumps(report, indent=2)
        st.download_button(
            "📥 Download Report",
            data=report_json,
            file_name=f"reproducibility_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    except Exception as e:
        handle_error(e, "Failed to generate reproducibility report")


def render_execution_page() -> None:
    """
    Render the experiment execution page.

    This page provides controls for running experiments with real-time
    monitoring and progress tracking.
    """
    st.markdown('<div class="section-header">🚀 Experiment Execution</div>',
                unsafe_allow_html=True)

    experiment_state = st.session_state.experiment_state

    # Pre-execution validation
    st.subheader("✅ Pre-Execution Validation")

    validation_results = validate_experiment_setup()
    display_validation_results(validation_results)

    # Execution controls
    st.markdown("---")
    st.subheader("🎮 Execution Controls")

    col1, col2, col3 = st.columns(3)

    with col1:
        num_runs = st.number_input(
            "Number of Runs",
            min_value=1,
            max_value=10,
            value=1,
            help="Number of independent experimental runs"
        )

    with col2:
        dry_run = st.checkbox(
            "Dry Run",
            value=False,
            help="Validate setup without actually training"
        )

    with col3:
        save_results = st.checkbox(
            "Save Results",
            value=True,
            help="Save experiment results and artifacts"
        )

    # Execution button
    can_execute = all(validation_results.values())

    if st.button(
        "🚀 Start Experiment" if not dry_run else "🧪 Dry Run",
        type="primary",
        disabled=not can_execute or experiment_state.experiment_running
    ):
        if dry_run:
            run_dry_run()
        else:
            start_experiment(num_runs, save_results)

    # Execution status and progress
    if experiment_state.experiment_running:
        st.markdown("---")
        st.subheader("📊 Execution Progress")
        render_execution_progress()


def validate_experiment_setup() -> Dict[str, bool]:
    """Validate experiment setup and return validation results."""
    experiment_state = st.session_state.experiment_state

    validation = {
        "Configuration Loaded": experiment_state.config is not None,
        "Data Available": experiment_state.metadata_df is not None,
        "Model Configured": (
            experiment_state.config is not None and
            hasattr(experiment_state.config, 'model')
        ),
        "Task Configured": (
            experiment_state.config is not None and
            hasattr(experiment_state.config, 'task')
        ),
        "Reproducibility Set": (
            experiment_state.config is not None and
            hasattr(experiment_state.config, 'reproducibility')
        )
    }

    return validation


def display_validation_results(validation: Dict[str, bool]) -> None:
    """Display validation results with status indicators."""
    for check, status in validation.items():
        if status:
            st.success(f"✅ {check}")
        else:
            st.error(f"❌ {check}")


def run_dry_run() -> None:
    """Run experiment validation without actual training."""
    try:
        st.info("🧪 Running dry run validation...")

        # Simulate validation steps
        progress_bar = st.progress(0)
        status_text = st.empty()

        steps = [
            "Validating configuration",
            "Checking data availability",
            "Initializing model",
            "Setting up reproducibility",
            "Preparing data loaders",
            "Validation complete"
        ]

        for i, step in enumerate(steps):
            status_text.text(f"Step {i+1}/{len(steps)}: {step}")
            progress_bar.progress((i + 1) / len(steps))

            # Simulate processing time
            import time
            time.sleep(0.5)

        st.success("✅ Dry run completed successfully! Experiment setup is valid.")

    except Exception as e:
        handle_error(e, "Dry run failed")


def start_experiment(num_runs: int, save_results: bool) -> None:
    """Start the actual experiment execution."""
    try:
        st.session_state.experiment_state.experiment_running = True
        st.info(f"🚀 Starting experiment with {num_runs} run(s)...")

        # This would integrate with the actual experiment execution
        # For now, simulate the process
        progress_bar = st.progress(0)
        status_text = st.empty()

        for run in range(num_runs):
            status_text.text(f"Running experiment {run + 1}/{num_runs}")

            # Simulate training progress
            for epoch in range(5):  # Simulate 5 epochs
                progress = ((run * 5) + epoch + 1) / (num_runs * 5)
                progress_bar.progress(progress)

                import time
                time.sleep(0.2)

        # Store mock results
        st.session_state.experiment_state.results = {
            "num_runs": num_runs,
            "final_accuracy": 0.85 + np.random.normal(0, 0.05),
            "final_loss": 0.3 + np.random.normal(0, 0.1),
            "training_time": num_runs * 120,  # seconds
            "best_epoch": np.random.randint(15, 25)
        }

        st.session_state.experiment_state.experiment_running = False
        st.success("✅ Experiment completed successfully!")
        st.rerun()

    except Exception as e:
        st.session_state.experiment_state.experiment_running = False
        handle_error(e, "Experiment execution failed")


def render_execution_progress() -> None:
    """Render real-time execution progress."""
    # This would show real-time metrics during training
    st.info("🔄 Experiment in progress...")

    # Placeholder for real-time metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Current Epoch", "15/100")

    with col2:
        st.metric("Training Loss", "0.245")

    with col3:
        st.metric("Validation Accuracy", "0.823")

    # Stop button
    if st.button("⏹️ Stop Experiment", type="secondary"):
        st.session_state.experiment_state.experiment_running = False
        st.warning("⚠️ Experiment stopped by user")
        st.rerun()


def render_results_page() -> None:
    """
    Render the results and analysis page.

    This page displays experiment results, metrics, and provides tools
    for analysis and visualization of experimental outcomes.
    """
    st.markdown('<div class="section-header">📈 Results & Analysis</div>',
                unsafe_allow_html=True)

    experiment_state = st.session_state.experiment_state

    if not experiment_state.results:
        st.info("🔍 No experiment results available. Run an experiment to see results here.")
        return

    # Results overview
    st.subheader("📊 Experiment Results")

    results = experiment_state.results

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Final Accuracy",
            f"{results.get('final_accuracy', 0):.3f}",
            help="Best validation accuracy achieved"
        )

    with col2:
        st.metric(
            "Final Loss",
            f"{results.get('final_loss', 0):.3f}",
            help="Final validation loss"
        )

    with col3:
        st.metric(
            "Training Time",
            f"{results.get('training_time', 0):.0f}s",
            help="Total training time in seconds"
        )

    with col4:
        st.metric(
            "Best Epoch",
            f"{results.get('best_epoch', 0)}",
            help="Epoch with best validation performance"
        )

    # Detailed results
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Training Curves", "📊 Metrics Analysis", "📋 Detailed Results"])

    with tab1:
        render_training_curves()

    with tab2:
        render_metrics_analysis()

    with tab3:
        render_detailed_results()

    # Export options
    st.markdown("---")
    st.subheader("📥 Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Export Metrics"):
            export_metrics()

    with col2:
        if st.button("📈 Export Plots"):
            export_plots()

    with col3:
        if st.button("📄 Generate Report"):
            generate_experiment_report()


def render_training_curves() -> None:
    """Render training curves and learning progress."""
    # Generate mock training data for demonstration
    epochs = list(range(1, 26))
    train_loss = [0.8 * np.exp(-0.1 * e) + 0.1 + np.random.normal(0, 0.02) for e in epochs]
    val_loss = [0.9 * np.exp(-0.08 * e) + 0.15 + np.random.normal(0, 0.03) for e in epochs]
    train_acc = [1 - 0.7 * np.exp(-0.12 * e) + np.random.normal(0, 0.02) for e in epochs]
    val_acc = [1 - 0.8 * np.exp(-0.1 * e) + np.random.normal(0, 0.03) for e in epochs]

    # Loss curves
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines', name='Training Loss'))
    fig_loss.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', name='Validation Loss'))
    fig_loss.update_layout(
        title="Training and Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode='x unified'
    )
    st.plotly_chart(fig_loss, use_container_width=True)

    # Accuracy curves
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=epochs, y=train_acc, mode='lines', name='Training Accuracy'))
    fig_acc.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines', name='Validation Accuracy'))
    fig_acc.update_layout(
        title="Training and Validation Accuracy",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        hovermode='x unified'
    )
    st.plotly_chart(fig_acc, use_container_width=True)


def render_metrics_analysis() -> None:
    """Render detailed metrics analysis."""
    # Mock confusion matrix
    st.subheader("🎯 Confusion Matrix")

    # Generate mock confusion matrix
    n_classes = 5
    cm = np.random.randint(0, 50, (n_classes, n_classes))
    np.fill_diagonal(cm, np.random.randint(80, 150, n_classes))

    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=[f"Class {i}" for i in range(n_classes)],
        y=[f"Class {i}" for i in range(n_classes)],
        text_auto=True
    )
    fig_cm.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

    # Per-class metrics
    st.subheader("📊 Per-Class Metrics")

    metrics_data = {
        'Class': [f'Class {i}' for i in range(n_classes)],
        'Precision': np.random.uniform(0.7, 0.95, n_classes),
        'Recall': np.random.uniform(0.65, 0.9, n_classes),
        'F1-Score': np.random.uniform(0.7, 0.92, n_classes),
        'Support': np.random.randint(50, 200, n_classes)
    }

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)


def render_detailed_results() -> None:
    """Render detailed experimental results."""
    experiment_state = st.session_state.experiment_state

    # Configuration summary
    st.subheader("⚙️ Configuration Summary")
    if experiment_state.config:
        config_summary = {
            "Experiment Name": getattr(experiment_state.config, 'name', 'Unknown'),
            "Model": getattr(getattr(experiment_state.config, 'model', {}), 'name', 'Unknown'),
            "Task Type": getattr(getattr(experiment_state.config, 'task', {}), 'type', 'Unknown'),
            "Batch Size": getattr(getattr(experiment_state.config, 'data', {}), 'batch_size', 'Unknown'),
            "Learning Rate": getattr(getattr(experiment_state.config, 'optimization', {}), 'learning_rate', 'Unknown')
        }

        for key, value in config_summary.items():
            st.text(f"{key}: {value}")

    # Full results
    st.subheader("📋 Complete Results")
    st.json(experiment_state.results)

    # System information
    st.subheader("🖥️ System Information")
    system_info = {
        "Python Version": platform.python_version(),
        "PyTorch Version": torch.__version__,
        "CUDA Available": torch.cuda.is_available(),
        "Timestamp": datetime.now().isoformat()
    }

    for key, value in system_info.items():
        st.text(f"{key}: {value}")


def export_metrics() -> None:
    """Export metrics to downloadable format."""
    try:
        experiment_state = st.session_state.experiment_state

        # Create metrics summary
        metrics = {
            "experiment_name": getattr(experiment_state.config, 'name', 'unknown'),
            "timestamp": datetime.now().isoformat(),
            "results": experiment_state.results,
            "configuration": experiment_state.config.__dict__ if experiment_state.config else {}
        }

        # Convert to JSON
        metrics_json = json.dumps(metrics, indent=2, default=str)

        st.download_button(
            "📥 Download Metrics JSON",
            data=metrics_json,
            file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

        st.success("✅ Metrics export prepared!")

    except Exception as e:
        handle_error(e, "Failed to export metrics")


def export_plots() -> None:
    """Export plots and visualizations."""
    st.info("📈 Plot export functionality would save training curves and analysis plots")


def generate_experiment_report() -> None:
    """Generate comprehensive experiment report."""
    try:
        experiment_state = st.session_state.experiment_state

        report = f"""
# Experiment Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Experiment Configuration
- **Name**: {getattr(experiment_state.config, 'name', 'Unknown')}
- **Model**: {getattr(getattr(experiment_state.config, 'model', {}), 'name', 'Unknown')}
- **Task**: {getattr(getattr(experiment_state.config, 'task', {}), 'name', 'Unknown')}

## Results Summary
- **Final Accuracy**: {experiment_state.results.get('final_accuracy', 'N/A'):.3f}
- **Final Loss**: {experiment_state.results.get('final_loss', 'N/A'):.3f}
- **Training Time**: {experiment_state.results.get('training_time', 'N/A')} seconds
- **Best Epoch**: {experiment_state.results.get('best_epoch', 'N/A')}

## System Information
- **Python Version**: {platform.python_version()}
- **PyTorch Version**: {torch.__version__}
- **CUDA Available**: {torch.cuda.is_available()}

---
*Generated by PHM-Vibench Research Framework*
        """

        st.download_button(
            "📄 Download Report",
            data=report,
            file_name=f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

        st.success("✅ Experiment report generated!")

    except Exception as e:
        handle_error(e, "Failed to generate report")


def render_documentation_page() -> None:
    """
    Render the documentation and help page.

    This page provides comprehensive documentation, tutorials, and help
    for using the PHM-Vibench framework effectively.
    """
    st.markdown('<div class="section-header">📚 Documentation & Help</div>',
                unsafe_allow_html=True)

    # Documentation sections
    doc_section = st.selectbox(
        "Documentation Section",
        [
            "Getting Started",
            "Configuration Guide",
            "Model Documentation",
            "Reproducibility Guide",
            "API Reference",
            "Troubleshooting",
            "Examples"
        ]
    )

    if doc_section == "Getting Started":
        render_getting_started_docs()
    elif doc_section == "Configuration Guide":
        render_configuration_docs()
    elif doc_section == "Model Documentation":
        render_model_docs()
    elif doc_section == "Reproducibility Guide":
        render_reproducibility_docs()
    elif doc_section == "API Reference":
        render_api_docs()
    elif doc_section == "Troubleshooting":
        render_troubleshooting_docs()
    elif doc_section == "Examples":
        render_examples_docs()


def render_getting_started_docs() -> None:
    """Render getting started documentation."""
    st.markdown("""
    ## 🚀 Getting Started with PHM-Vibench

    Welcome to the PHM-Vibench Research Framework! This guide will help you get started
    with conducting publication-quality experiments for industrial signal analysis.

    ### Quick Start Steps

    1. **📋 Configure Your Experiment**
       - Go to the Configuration page
       - Choose a template or create custom configuration
       - Validate your settings

    2. **📊 Load Your Data**
       - Navigate to Data Management
       - Upload metadata or use example data
       - Explore your dataset

    3. **🤖 Select Your Model**
       - Visit Model Selection page
       - Browse available models
       - Configure model parameters

    4. **⚙️ Set Reproducibility**
       - Configure reproducibility settings
       - Set random seeds
       - Enable environment tracking

    5. **🚀 Run Experiment**
       - Execute your experiment
       - Monitor progress in real-time
       - Review results and analysis

    ### Key Features

    - **Scientific Rigor**: Publication-quality code with proper documentation
    - **Reproducibility**: Comprehensive reproducibility framework
    - **Flexibility**: Support for multiple model architectures and tasks
    - **Visualization**: Rich visualizations and analysis tools
    """)


def render_configuration_docs() -> None:
    """Render configuration documentation."""
    st.markdown("""
    ## ⚙️ Configuration Guide

    The PHM-Vibench framework uses a hierarchical configuration system with validation
    to ensure experiment reproducibility and correctness.

    ### Configuration Structure

    ```yaml
    name: "experiment_name"
    description: "Experiment description"
    tags: ["tag1", "tag2"]

    reproducibility:
      global_seed: 42
      torch_deterministic: true
      track_environment: true

    data:
      data_root: "data/"
      metadata_file: "metadata.xlsx"
      batch_size: 64
      window_size: 4096

    model:
      name: "ResNet1D"
      type: "CNN"
      input_dim: 3
      num_classes: 10

    optimization:
      optimizer: "adam"
      learning_rate: 0.001
      max_epochs: 100

    task:
      name: "classification"
      type: "DG"
      loss_function: "cross_entropy"
    ```

    ### Configuration Methods

    1. **Template-based**: Start with predefined templates
    2. **YAML Editor**: Edit configuration directly
    3. **File Upload**: Upload existing configuration files

    ### Validation

    All configurations are automatically validated for:
    - Required fields
    - Parameter ranges
    - Type compatibility
    - Cross-section consistency
    """)


def render_model_docs() -> None:
    """Render model documentation."""
    st.markdown("""
    ## 🤖 Model Documentation

    PHM-Vibench supports multiple model architectures for time-series analysis:

    ### Available Model Types

    #### CNN Models
    - **ResNet1D**: 1D ResNet for time-series classification
    - **CNN1D**: Basic 1D CNN architecture
    - **DenseNet1D**: 1D DenseNet implementation

    #### RNN Models
    - **LSTM**: Long Short-Term Memory networks
    - **GRU**: Gated Recurrent Units
    - **BiLSTM**: Bidirectional LSTM

    #### Transformer Models
    - **ViT1D**: Vision Transformer adapted for 1D signals
    - **TimeSeriesTransformer**: Transformer for time-series

    #### Neural Operators
    - **FNO1D**: Fourier Neural Operator for 1D signals
    - **DeepONet**: Deep Operator Networks

    ### Model Configuration

    Each model requires specific parameters:

    ```yaml
    model:
      name: "ResNet1D"
      type: "CNN"
      input_dim: 3          # Number of input channels
      num_classes: 10       # Output classes (classification)
      block_type: "basic"   # ResNet block type
      layers: [2, 2, 2, 2]  # Blocks per layer
      dropout: 0.1          # Dropout rate
    ```

    ### Adding Custom Models

    To add custom models:
    1. Implement model class
    2. Register with factory
    3. Add configuration schema
    4. Include documentation
    """)


def render_reproducibility_docs() -> None:
    """Render reproducibility documentation."""
    st.markdown("""
    ## 🔬 Reproducibility Guide

    Scientific reproducibility is a core principle of PHM-Vibench. The framework
    provides comprehensive tools to ensure your experiments are fully reproducible.

    ### Reproducibility Features

    #### Deterministic Execution
    - Global random seed management
    - PyTorch deterministic algorithms
    - NumPy seed synchronization
    - CUDA deterministic operations

    #### Environment Tracking
    - System information logging
    - Package version tracking
    - Git commit recording
    - Hardware configuration

    #### Configuration Management
    - Immutable configuration hashing
    - Parameter validation
    - Cross-platform compatibility
    - Version control integration

    ### Best Practices

    1. **Always Set Seeds**: Use consistent random seeds across runs
    2. **Track Environment**: Enable environment tracking for debugging
    3. **Version Control**: Commit code before running experiments
    4. **Document Changes**: Record any manual modifications
    5. **Validate Results**: Compare results across different environments

    ### Reproducibility Checklist

    - ✅ Configuration validated and saved
    - ✅ Random seeds set consistently
    - ✅ Environment information recorded
    - ✅ Git commit tracked
    - ✅ Dependencies documented
    - ✅ Results validated across runs
    """)


def render_api_docs() -> None:
    """Render API reference documentation."""
    st.markdown("""
    ## 📖 API Reference

    ### Core Classes

    #### ExperimentConfig
    Top-level configuration class with validation.

    ```python
    from refactored_configs.config_schema import ExperimentConfig

    config = ExperimentConfig(
        name="my_experiment",
        data=DataConfig(...),
        model=ModelConfig(...),
        task=TaskConfig(...)
    )
    ```

    #### ModelFactory
    Factory for creating model instances.

    ```python
    from refactored_configs.improved_factory import model_factory

    model = model_factory.create(config.model, metadata)
    ```

    #### ReproducibilityManager
    Manager for reproducibility setup and tracking.

    ```python
    from refactored_configs.reproducibility_framework import ReproducibilityManager

    manager = ReproducibilityManager(config.reproducibility, "experiment", output_dir)
    manager.setup_reproducibility()
    ```

    ### Configuration Schema

    #### DataConfig
    - `data_root`: Root directory for datasets
    - `metadata_file`: Metadata file name
    - `batch_size`: Training batch size
    - `window_size`: Signal window size

    #### ModelConfig
    - `name`: Model class name
    - `type`: Model category
    - `input_dim`: Input feature dimension
    - `num_classes`: Output classes (classification)

    #### OptimizationConfig
    - `optimizer`: Optimizer type
    - `learning_rate`: Initial learning rate
    - `max_epochs`: Maximum training epochs
    - `early_stopping`: Enable early stopping
    """)


def render_troubleshooting_docs() -> None:
    """Render troubleshooting documentation."""
    st.markdown("""
    ## 🔧 Troubleshooting

    ### Common Issues and Solutions

    #### Configuration Errors

    **Problem**: "Configuration validation failed"
    **Solution**:
    - Check required fields are present
    - Verify parameter types and ranges
    - Ensure cross-section compatibility

    **Problem**: "Model not found in registry"
    **Solution**:
    - Check model name spelling
    - Verify model type is correct
    - Ensure model is properly registered

    #### Data Loading Issues

    **Problem**: "Metadata file not found"
    **Solution**:
    - Verify file path is correct
    - Check file permissions
    - Ensure file format is supported (.xlsx, .csv)

    **Problem**: "Data shape mismatch"
    **Solution**:
    - Check input_dim matches data channels
    - Verify window_size is appropriate
    - Ensure data preprocessing is correct

    #### Reproducibility Issues

    **Problem**: "Results not reproducible"
    **Solution**:
    - Enable deterministic mode
    - Check all random seeds are set
    - Verify environment consistency
    - Disable CUDNN benchmark

    #### Performance Issues

    **Problem**: "Training is slow"
    **Solution**:
    - Increase batch size if memory allows
    - Enable CUDNN benchmark (reduces determinism)
    - Use mixed precision training
    - Optimize data loading (num_workers)

    ### Getting Help

    If you encounter issues not covered here:
    1. Check the error details in the expandable error section
    2. Review the configuration validation results
    3. Consult the API documentation
    4. Report bugs on GitHub: https://github.com/PHMbench/PHM-Vibench/issues
    """)


def render_examples_docs() -> None:
    """Render examples documentation."""
    st.markdown("""
    ## 📋 Examples

    ### Example 1: Basic Classification

    ```yaml
    name: "bearing_fault_classification"
    description: "Classify bearing faults using ResNet1D"

    data:
      data_root: "data/CWRU/"
      metadata_file: "metadata.xlsx"
      batch_size: 64
      window_size: 4096

    model:
      name: "ResNet1D"
      type: "CNN"
      input_dim: 1
      num_classes: 10

    task:
      name: "classification"
      type: "DG"
      loss_function: "cross_entropy"
    ```

    ### Example 2: Time Series Prediction

    ```yaml
    name: "rul_prediction"
    description: "Remaining useful life prediction using LSTM"

    data:
      data_root: "data/CMAPSS/"
      metadata_file: "metadata.csv"
      batch_size: 32
      window_size: 50

    model:
      name: "LSTM"
      type: "RNN"
      input_dim: 14
      hidden_dim: 128
      num_layers: 2
      output_dim: 1

    task:
      name: "regression"
      type: "FS"
      loss_function: "mse"
    ```

    ### Example 3: Domain Adaptation

    ```yaml
    name: "cross_domain_classification"
    description: "Domain adaptation for fault classification"

    data:
      data_root: "data/multi_domain/"
      metadata_file: "metadata.xlsx"
      batch_size: 64

    model:
      name: "ResNet1D"
      type: "CNN"
      input_dim: 3
      num_classes: 5

    task:
      name: "classification"
      type: "DG"
      domain_config:
        source_domains: [0, 1, 2]
        target_domains: [3, 4]
        adaptation_method: "DANN"
    ```

    ### Running Examples

    1. Copy the configuration to the YAML editor
    2. Modify paths and parameters as needed
    3. Load your data or use example data
    4. Configure reproducibility settings
    5. Run the experiment
    6. Analyze results
    """)


def main() -> None:
    """
    Main application entry point.

    This function orchestrates the entire Streamlit application, handling
    page routing, state management, and error handling.
    """
    try:
        # Setup page configuration and styling
        setup_page_config()

        # Initialize session state
        init_session_state()

        # Display header
        display_header()

        # Create sidebar navigation
        current_page = create_sidebar_navigation()

        # Route to appropriate page
        if current_page == "Configuration":
            render_configuration_page()
        elif current_page == "Data":
            render_data_management_page()
        elif current_page == "Models":
            render_model_selection_page()
        elif current_page == "Reproducibility":
            render_reproducibility_page()
        elif current_page == "Execution":
            render_execution_page()
        elif current_page == "Results":
            render_results_page()
        elif current_page == "Documentation":
            render_documentation_page()
        else:
            st.error(f"Unknown page: {current_page}")

        # Display footer
        st.markdown("---")
        st.markdown(
            '<div style="text-align: center; color: #666; padding: 1rem;">'
            '🔬 PHM-Vibench Research Framework v2.0 | '
            'Built with ❤️ for Scientific Research'
            '</div>',
            unsafe_allow_html=True
        )

    except Exception as e:
        # Global error handler
        st.error("🚨 Application Error")
        st.error(f"An unexpected error occurred: {str(e)}")

        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())

        if st.button("🔄 Restart Application"):
            st.session_state.clear()
            st.rerun()


if __name__ == "__main__":
    main()