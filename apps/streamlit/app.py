"""Maintained Streamlit entry point for PHM-Vibench."""

try:
    from .workspace import main
except ImportError:  # pragma: no cover - Streamlit executes this file as a script.
    from workspace import main  # type: ignore


if __name__ == "__main__":
    main()
