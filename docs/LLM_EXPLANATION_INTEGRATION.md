# LLM explanation integration for interpretable PHM models

## Scope

PHMFactory does not ask an LLM to infer a diagnosis or a causal mechanism from raw tensors. The supported path is:

```text
interpretable PHM model
→ same-forward model-native trace
→ PHM Explanation Intermediate Representation (PHM-EIR)
→ provider-specific LLM callback
→ validated public explanation
```

This separates four objects that are often conflated:

1. the model prediction;
2. the model-native evidence trace;
3. an externally supplied mechanism mapping;
4. the natural-language explanation.

A model operator, fuzzy rule, attention weight, or selected feature is not called a physical mechanism unless a separate mapping supplies that status.

## Public API

```python
from phmfactory.explanation import (
    export_xoan_state,
    export_tspn_uxfd_fuzzy_state,
    explain_with_llm,
)
```

The package contains no provider SDK and reads no API key. A caller supplies one explicit `generate(packet)` function. There is no retry, provider substitution, parser repair, or fallback.

## PHM-EIR

The common state contains:

- prediction label, class index, confidence, and logits;
- typed evidence atoms with identifiers, values, units, and source;
- model-native evidence paths;
- same-forward class contributions when the model exports them;
- mechanism relations only when explicitly supplied;
- uncertainty and trace-reconstruction quantities;
- operating conditions;
- capabilities and limitations.

The capability list bounds what the LLM may say. For example, a structural operator path does not license a per-operator causal-contribution claim.

## XOAN operator-path model

`XOANOperatorPath.forward_evidence` already exposes a typed executable path, relaxed/discrete agreement, predictive entropy, and dictionary insufficiency. Export one batch element:

```python
state = export_xoan_state(
    model,
    x,
    sample_id="bearing-run-017",
    class_names=("normal", "inner", "outer", "ball"),
    operating_conditions={"speed_rpm": 1772, "load_nm": 3.1},
)
```

The adapter reports the selected path as model-native structure. It does not invent a bearing mechanism relation.

## TSPN-UXFD fuzzy model

`TSPN_UXFD.forward_with_fuzzy_trace` exposes reduced features, rule firing, rule consequents, additive rule contributions, and a same-forward reconstruction. Export it with optional expert-supplied rule semantics:

```python
state = export_tspn_uxfd_fuzzy_state(
    model,
    x,
    sample_id="bearing-run-018",
    class_names=("normal", "inner", "outer", "ball"),
    rule_names=("rule 0", "rule 1", "rule 2"),
    rule_relations={
        0: ("supports under the admitted bearing rule map", "outer-race fault"),
    },
)
```

Without `rule_relations`, the exported fuzzy rules remain model-native constructs rather than mechanically validated claims.

## Provider boundary

```python
def generate(packet):
    # Convert `packet` to the selected provider's structured-output request.
    # Return a Python mapping with exactly the output_contract fields.
    ...

explanation = explain_with_llm(
    state,
    generate,
    audience="maintenance engineer",
    detail="standard",
)
```

Every substantive explanation claim must cite supplied evidence or path IDs. Unknown evidence, path, and relation IDs are rejected. This deterministic check prevents unsupported references among accepted outputs; it does not prove that the natural-language wording is semantically faithful. That remains an intervention and annotation question.

## Current model coverage

| Model path | Native evidence | Current PHM-EIR capability |
|---|---|---|
| `XOANOperatorPath` | typed executable operator path, relaxed/discrete fidelity, uncertainty | structural path, reconstruction audit, interventions |
| `TSPN_UXFD` fuzzy branch | rule firing, consequents, additive logit contributions | rule path, contribution reconstruction, rule intervention |
| plain `TSPN` | operator architecture and final logits only | not certified for case-level evidence verbalization |

Plain TSPN needs a same-forward trace exporter before it can support evidence-faithful language. A second forward or a post-hoc plot is not silently treated as the trace that produced the prediction.

## Evaluation

The engineering contract supports three separate measurements:

1. **adapter fidelity:** does PHM-EIR preserve the model-native trace and reconstructable decision quantities?
2. **verbalization fidelity:** does independently parsing the public explanation recover the supplied PHM-EIR semantics?
3. **mechanism compatibility:** do the disclosed model-native paths belong to an independently defined PHM mechanism-admissible set?

These measurements must remain separate. Fluent wording or valid identifiers cannot compensate for an incorrect mechanism relation.
