# Legacy Streamlit Prototype Package

The modules in this directory are historical Streamlit prototypes retained
temporarily for compatibility and provenance. New features are not developed here.

The maintained configuration-first workspace lives at:

```text
apps/streamlit/
```

Launch it from the repository root:

```bash
streamlit run apps/streamlit/app.py
```

The historical root command remains compatible and routes to the same workspace:

```bash
streamlit run streamlit_app.py
```

The maintained UI invokes experiments only through:

```bash
python main.py --config <yaml> [--override key=value ...]
```

It does not import Pipeline functions directly and keeps Streamlit optional for
the core CLI workflow.
