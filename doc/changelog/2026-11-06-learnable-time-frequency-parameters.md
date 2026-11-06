# Optional bounded parameters for finite time–frequency maps

`TimeFrequencyBranch` accepts per-transform `learnable: true` with explicit bounds for a continuous STFT taper exponent, CWT scales, selected Stockwell frequencies, or chirplet frequencies and rate. Sampling counts, FFT size, stride and the predeclared valid-center geometry stay fixed within the parameterized family. Omission preserves the legacy calculation and checkpoint keys.

Use `learnable: true, trainable: false` for the same parameterized geometry with its new parameters frozen. Use the saved constructor configuration with strict state restoration. Read effective frequencies/scales/rate and current-window ENBW from the restored module; persisted initialization buffers need not equal the current learned coordinates. Frequencies are cycles/sample; chirp rate is cycles/sample².

The coefficients remain differentiable with respect to the input and the enabled continuous parameters. None of these switches fits from evaluation data or generates a different filter per input.
