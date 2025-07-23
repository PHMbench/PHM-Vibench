# GUI Usage

A Streamlit application is provided in `app/gui.py`. The interface streams
process output and allows pausing/resuming the run. Parameters for the data,
model, task and trainer are organized in expandable sections, each with an
"Update" button to confirm changes. The page guides you through three steps:
1) loading metadata, 2) configuring parameters and 3) running the experiment.

Launch the GUI with:

```bash
streamlit run app/gui.py
```
