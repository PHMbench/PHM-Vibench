# Frontend

The repository's frontend surface is consolidated here.

The Streamlit console is experimental and not a validation gate. Run it with:

```bash
streamlit run frontend/streamlit_app.py
```

Layout:

- `streamlit_app.py`: primary Streamlit entrypoint
- `console/`: maintained theme, state, adapters, and page renderers
- `pages/`: Streamlit multipage wrappers
- `legacy/`: archived GUI experiments kept for reference only
- `.streamlit/`: frontend-local Streamlit theme defaults

Main path:

- `Workbench`: current protocol, recent runs, evidence status, quick actions
- `Compose`: current-input preflight gating before launch
- `Runs` / `Artifacts`: shared selected run and shared run filters
- `Compare`: baseline-first protocol-aware comparison
