# GUI Usage

<<<<<<< HEAD
A Streamlit application is provided in `app/gui.py`. The interface streams
process output and allows pausing/resuming the run. Parameters for the data,
model, task and trainer are organized in expandable sections, each with an
"Update" button to confirm changes.

=======
A Streamlit application is provided in `app/gui.py`. It organizes data, model,
task and trainer parameters in expandable sections. After loading metadata you
can preview it directly and visualize raw signals for a selected sample.
Process output is streamed live and you may pause or resume the run.
>>>>>>> cbe3fed574e34e411cbe74923297c7d0a77a5393

Launch the GUI with:

```bash
streamlit run app/gui.py
```
