# App Module

This package provides the Streamlit interface for running PHM-Vibench pipelines.
The web UI guides users through metadata loading, parameter configuration, signal
preview, experiment launch, and live process output.

## Main app

The maintained modular app is assembled from:

- `state.py`: `st.session_state` defaults.
- `layout.py`: UI sections and data helpers.
- `pipeline.py`: training subprocess launch and output streaming.
- `gui.py`: final Streamlit application.

Launch it with:

```bash
streamlit run app/gui.py
```

## Refactored prototype

`app/gui_refactored.py` is a larger prototype UI with additional workflow pages
and design notes. See `app/README_GUI_Refactored.md` before changing it.
