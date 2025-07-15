# App Module

This package provides a Streamlit interface for PHM-Vibench.  The code is split
into small modules:

- `state.py` manages `st.session_state` defaults.
- `layout.py` defines UI sections and data helpers.
- `pipeline.py` launches the training subprocess and streams output.
- `gui.py` assembles the above pieces into the final application.

Launch the app with:

```bash
streamlit run app/gui.py
```
