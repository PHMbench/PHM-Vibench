# Recheck detached Streamlit runs

A run whose service restarted no longer reserves the worker forever after its process
exits. Viewing the run or submitting another experiment rechecks the recorded PID. When
absence can be established safely, the record becomes `orphaned`, with an unknown final
exit status. It is not marked successful and no files are removed.

A present or unverifiable PID remains reserved. Windows does not use `os.kill(pid, 0)` as
a probe. The page provides **Recheck process** and a separately confirmed **Release
finished run** for cases that require OS-level verification. Neither action adopts,
terminates or retries an unmanaged process. A saved cancellation request alone is not
proof that a process exited.

Changes are limited to the optional frontend, its existing tests and this documentation.
Training, configuration, split, objective, metrics and checkpoint selection are unchanged.
