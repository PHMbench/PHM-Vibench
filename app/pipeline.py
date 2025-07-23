"""Helpers to run PHM-Vibench pipelines as subprocesses."""
from __future__ import annotations

import os
import signal
import subprocess
import threading
from typing import List

import streamlit as st


def _append_output(line: bytes) -> None:
    """Append a decoded output ``line`` and rerun the Streamlit app."""
    st.session_state.output_lines.append(line.decode())
    st.experimental_rerun()


def _reader_thread(process: subprocess.Popen) -> None:
    """Collect ``process`` output and store lines in the session state."""
    for line in iter(process.stdout.readline, b""):
        _append_output(line)
    process.wait()
    st.session_state.process = None
    st.session_state.paused = False


def start_pipeline(config_path: str) -> None:
    """Start the main training script as a subprocess."""

    if st.session_state.process:
        st.warning("\u5df2\u5b58\u5728\u6b63\u5728\u8fd0\u884c\u7684\u5b50\u8fdb\u7a0b")
        return

    cmd = ["python", "main.py", "--config_path", config_path]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    st.session_state.process = process
    st.session_state.output_lines = []
    st.session_state.experiment_run = True
    thread = threading.Thread(target=_reader_thread, args=(process,), daemon=True)
    thread.start()


def toggle_pause() -> None:
    """Pause or resume the active subprocess using POSIX signals."""
    process = st.session_state.process
    if not process:
        return
    if st.session_state.paused:
        os.kill(process.pid, signal.SIGCONT)
        st.session_state.paused = False
    else:
        os.kill(process.pid, signal.SIGSTOP)
        st.session_state.paused = True


def display_output() -> None:
    """Render captured terminal output in a text area widget."""
    if st.session_state.output_lines:
        st.text_area(
            "\u8fd0\u884c\u8f93\u51fa", "".join(st.session_state.output_lines), height=200
        )
