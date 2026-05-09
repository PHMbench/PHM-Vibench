# DDPM

Status: exploratory runtime baseline.

The DDPM component provides epsilon-prediction loss, a finite beta/alpha
scheduler, and a stateless reverse-process sampler. It does not add FFT or
envelope-spectrum training losses. Conditions remain `fault_label` and
`domain_id`.

Synthetic outputs are not benchmark-valid by default. They require the same
manifest, split, and leakage evidence as the V0 CFM pipeline.
