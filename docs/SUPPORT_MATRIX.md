# Support matrix

This table is intentionally conservative. “Supported” means covered by the
KIR v1 contract and tests, not merely accepted by a parser.

| Area | Supported now | Evidence level | Next expansion |
| --- | --- | --- | --- |
| KIR | JSON KIR v1 and graph-plus-weights bundles | Frozen schema, round-trip tests, fail-closed validation | MessagePack transport preserving v1 semantics |
| PyTorch | Supported KIR subset through `torch.fx`: embedding, Conv1d, Linear/MatMul, Add, LUT activations, MaxPool1d, Argmax | Differential and frontend diagnostics tests | Broader shape-polymorphic frontend |
| ONNX | Not yet a public frontend | Planned interface only | ONNX-to-KIR importer with strict diagnostics |
| TFLite | Not yet a public frontend | Planned interface only | TFLite-to-KIR importer and quantization audit |
| C99 backend | Supported | Host compile/link, integer reference parity, static certificates | More kernels after KIR compatibility review |
| Host target | Supported | Native linked replay; first-class KIR VM with deterministic trace and metrics | — |
| Cortex-M3 QEMU | Experimental external backend | Freestanding image, vector table, linker map, semihosting replay | CI evidence retention and broader fixtures |
| Cortex-M0+/M4 | Cross-compile supported; RP2040 board adapter scaffold available | Compile/link inspection; physical replay still pending | Verified RP2040 evidence and additional boards |
| RISC-V 32 | Cross-compile profile | Toolchain path and static inspection contract | Bootable emulator and board profile |
| CMake/embedded SDK | Generated C can be integrated manually | Clean generated artifacts and strict flags | First-class project templates |
| Runtime allocation | Zero dynamic allocation in generated model | Undefined-symbol inspection | Keep as a release blocker |

## Compatibility levels

- **Stable**: frozen KIR v1, C99 semantics, KIR VM execution, reference
  interpreter, validation and safety diagnostics.
- **Experimental**: QEMU replay, hardware runner contract, target profiles,
  and generated integration glue.
- **Planned**: ONNX/TFLite import, turnkey SDK projects, multiple verified
  boards, and formal/bounded checking.

If a feature is not listed as supported, Kiln should reject it with a useful
diagnostic rather than silently lower it.
