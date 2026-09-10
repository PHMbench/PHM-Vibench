# TSPN_fusion research component

The original TSPN is unchanged. `X_model/TSPN_fusion.py` retains its complete frozen function and assembles configured envelope, STFT, Morlet/Mexican-hat CWT, selected-frequency Stockwell and chirplet branches. `TSPN_tf_operators.py` owns the finite analysis definitions. Frequencies are cycles/sample and the current task must have a fixed sampling convention.

Train through `forward_details` with `Components/tspn_fusion_loss.py`. Ordinary `forward` is deployment-only and returns log probabilities; it deliberately rejects use as an ordinary CE training model. The CE+Brier domain-relative loss and correction consistency do not set the independent deployment coefficient alpha or guarantee unseen-domain accuracy/F1.

The P01 project owns model YAML, H5 experiment scripts, coefficient assessment and manuscript proofs. No reverse import from this runtime into the paper repository is required. The existing operator-bias model remains a separate baseline, not an alias for fusion.

`python -m pytest -q test/test_tspn_fusion_v2.py` tests the transforms, gradients, freeze behavior and loss with an explicit reference fixture. The P01 integration suite additionally tests the actual original TSPN, installed Model Factory and constructed H5 execution. None of these tests establishes real industrial improvement.
