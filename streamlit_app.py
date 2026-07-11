"""Backward-compatible launcher for the maintained Streamlit workspace.

The implementation lives in ``apps/streamlit/app.py``. Existing commands using
``streamlit run streamlit_app.py`` therefore continue to open the same workspace.
"""

from apps.streamlit.app import main


if __name__ == "__main__":
    main()
