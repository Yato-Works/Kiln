# KIR v1 compatibility policy

KIR v1 is the stable, language-neutral interchange contract for Kiln. The
canonical transport is UTF-8 JSON with the following envelope:

```json
{
  "schema": "kiln.kir",
  "version": 1,
  "inputs": [],
  "outputs": [],
  "tensors": [],
  "nodes": []
}
```

The machine-readable contract is in [`schemas/kir-v1.schema.json`](schemas/kir-v1.schema.json).
The portable graph-plus-weights envelope is specified by
[`schemas/kir-bundle-v1.schema.json`](schemas/kir-bundle-v1.schema.json).

## Frozen guarantees

KIR v1 has no dynamic dimensions, implicit broadcasting, executable values,
Python class names, or frontend-specific objects. A conforming reader must
reject unknown top-level, tensor, node, and quantization fields. Integer fields
must be JSON integers, not booleans or floating-point values.

The seven operation names, dtype names, tensor layout, quantization semantics,
tie-breaking rules, and attribute meanings are defined in [`KIR.md`](KIR.md).
The JSON encoding is deterministic: keys are sorted by the reference Python
codec and graph order is preserved.

## Versioning rules

- A v1 reader must accept every valid v1 document and must reject future major
  versions rather than guessing their meaning.
- New required semantics, operation changes, dtype changes, or layout changes
  require a new major KIR version.
- Additive optional metadata may be introduced only in a new version; v1 does
  not permit unknown fields, so readers cannot silently ignore it.
- A transport such as MessagePack may be added later, but it must encode the
  same object model and preserve `schema` and `version` exactly.
- `kiln validate` is the compatibility gate. CI should validate every checked-in
  bundle and reject a schema change without an explicit versioning decision.

The current status is **frozen**. Compatibility changes require a documented
decision, updated schemas, golden fixtures, and a migration note.
