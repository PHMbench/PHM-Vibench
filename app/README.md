# App Module

<<<<<<< HEAD
This package provides a Streamlit interface for PHM-Vibench.  The code is split
into small modules:

- `state.py` manages `st.session_state` defaults.
- `layout.py` defines UI sections and data helpers.
- `pipeline.py` launches the training subprocess and streams output.
- `gui.py` assembles the above pieces into the final application.

Launch the app with:
=======
>>>>>>> cbe3fed574e34e411cbe74923297c7d0a77a5393
This directory contains a Streamlit interface for running PHM-Vibench pipelines.
The web UI guides users through loading metadata, configuring parameters and
starting experiments. Terminal output is streamed in real time and the process
can be paused or resumed.

Run the application with:

```bash
streamlit run app/gui.py
```
