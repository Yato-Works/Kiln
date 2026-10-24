# Security policy

## Scope

Kiln is a compiler and verification tool for small integer ML graphs. The
generated model is intended to run in a constrained target environment; it is
not a sandbox for executing untrusted Python, C, or model files.

## Reporting a vulnerability

Please do not disclose an exploitable issue in a public issue first. Use the
repository's private security reporting channel when available. If private
reporting is not configured yet, contact the maintainers listed by the hosting
repository and include:

- the affected commit or release;
- a minimal reproduction or malformed input;
- the expected safety boundary and the observed behavior;
- whether code execution, memory corruption, or a false safety certificate is
  possible.

Do not include secrets or private user data in a report. Safety-certificate
false positives and false negatives are both release-significant issues.

## Design expectations

KIR loading, static validation, interval analysis, and target runners should
fail closed. A target runner must not turn a host result into hardware evidence,
and a model that cannot establish int32 safety must not emit deployable C.
