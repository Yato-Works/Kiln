# Kiln Intermediate Representation (KIR) Specification

**Status**: Tier 0 / Phase 0 Frozen  
**Target Environment**: Bare-metal MCUs / Embedded Systems (Cortex-M0+/M3/M4/M7, RISC-V, POSIX)  
**Compiler Contract**: Strictly Zero-malloc, Deterministic C99 Output, Bit-exact Differential Parity

---

## 1. Architectural Philosophy & Invariants

Kiln compiles quantized deep learning computation graphs into standalone, dependency-free C99 code. Every model compiled by Kiln obeys the following non-negotiable guarantees:

1. **Zero Dynamic Memory Allocation**:
   - No `malloc`, `free`, or heap usage.
   - All intermediate buffers are placed inside a single static arena union (`_arena.data`) sized at compile time.
2. **Compile-Time Liveness & Memory Reuse**:
   - Tensors whose lifetimes do not overlap reuse the same byte offsets via a First-Fit allocator (`allocate_arena`).
   - Tensors designated as `graph_outputs` have their liveness extended to the final op step, preventing memory aliasing with downstream operations.
   - **In-place elementwise fusion**: an elementwise op whose emitted loop reads and writes one element at a time in the same order (currently `LUT`) aliases its output tensor onto its input buffer (`plan_inplace_aliases`). The alias is applied only when the input is a scratch tensor with exactly one consumer, both tensors have identical sizes, and neither is a graph input/output. The shared block spans the union of both lifetimes, so a fused `CONV1D -> LUT` stage costs zero extra arena bytes.
   - **Zero-copy graph inputs**: graph inputs are never written by any node, so `kiln_predict` reads them directly from the caller's `input_bytes` buffer (`const` pointer) instead of copying them into the arena. Only intermediate tensors occupy arena bytes.
3. **Integer-Only Determinism**:
   - No floating-point instructions or soft-float libraries.
   - Fixed-point arithmetic uses Q31 scaling and bit-shifting with rounding nudges.

---

## 2. Data Types, Memory Layout & Graph Input Contracts

### 2.1 Supported Data Types
| DType | Bitwidth | Representation | Use Cases |
| :--- | :--- | :--- | :--- |
| `UINT8` | 8 bits | Unsigned `[0, 255]` | Graph input discrete token indices (exclusively for `EMBEDDING`) |
| `INT8` | 8 bits | Signed two's complement `[-128, 127]` | Dense activations, raw feature inputs (e.g. Spectrogram), weights, LUT tables |
| `INT32` | 32 bits | Signed two's complement `[-2^31, 2^31 - 1]` | Accumulators for CONV1D, MATMUL, ADD |

### 2.2 Tensor Memory Layout Contract (Row-Major Canonical Order)
All multidimensional tensors in Kiln are strictly flattened and laid out in memory in **C-style row-major order** matching the exact sequence of declared axes:
$$\text{shape} = (D_0, D_1, \dots, D_{k-1})$$
The flat 1D memory offset for element $(i_0, i_1, \dots, i_{k-1})$ is computed as:
$$\text{offset} = \sum_{m=0}^{k-1} \left( i_m \times \prod_{n=m+1}^{k-1} D_n \right)$$
- The first axis ($D_0$) is the most outer / slowest-varying dimension.
- The final axis ($D_{k-1}$) is the most inner / fastest-varying dimension (contiguous in memory).
- **CONV1D Layout**: A tensor of shape $(C_{in}, L)$ is stored with channel as the outer dimension and spatial length as the inner dimension:
  $$\text{index}(c, t) = c \times L + t$$
- External inputs and frontend preprocessors **must conform strictly to this row-major order**.

### 2.3 Raw Feature Input Quantization Contract (EMBEDDING-Free Inputs)
When a graph receives raw continuous feature tensors directly into `CONV1D` or other layers without an `EMBEDDING` lookup:

1. **Graph Input DType**: Must be `INT8`. (`UINT8` is reserved strictly for discrete token IDs feeding `EMBEDDING`).
2. **Symmetric Centered Quantization (Zero-Point-Free)**:
   To avoid introducing zero-point offset correction terms into the hot MAC loops while maximizing the dynamic range across the full signed 8-bit scale $[-127, 127]$, input features with clipping bounds $[V_{min}, V_{max}]$ (where $V_{min} < V_{max}$, e.g. $[-80.0\text{ dB}, 0.0\text{ dB}]$ for log-mel spectrograms) are normalized and quantized at the frontend via:
   $$V_{mid} = \frac{V_{min} + V_{max}}{2}$$
   $$\text{half\_range} = \frac{V_{max} - V_{min}}{2}$$
   $$\text{scale} = \frac{\text{half\_range}}{127}$$
   $$\text{int8\_feature} = \text{clamp\_int8}\left( \text{round}\left( \frac{V - V_{mid}}{\text{scale}} \right) \right)$$
   - This mapping guarantees that $V_{min} \mapsto -127$, $V_{mid} \mapsto 0$, and $V_{max} \mapsto +127$, utilizing the full 8-bit resolution without modifying the kernel's arithmetic core.
   - **Contract Immutability**: The preprocessor constants $(V_{min}, V_{max}, V_{mid}, \text{scale})$ determined during Phase 1a training are frozen and serve as the exact specification that the Phase 1b C fixed-point frontend must reproduce bit-for-bit.

---

## 3. Quantization Semantics & Invariants (`sem.py`)

### 3.1 Quantization Parameter Contract (`QuantParam`)
Requantization transforms a 32-bit accumulator into an 8-bit activation:
$$\text{activation} = \text{clamp\_int8}\left( \text{round}\left( \text{accum} \times \frac{\text{multiplier}}{2^{31}} \times 2^{-\text{shift}} \right) \right)$$

- **`multiplier`**: `int32`, strictly within $[2^{30}, 2^{31})$ ($[1073741824, 2147483647]$).
- **`shift`**: Non-negative integer, strictly within $[0, 31]$.
- **Effective Scale Constraint ($S \in (0, 1.0)$)**:
  - In Tier 0, **effective scaling factor must be strictly less than 1.0**.
  - Any scale $S \ge 1.0$ (which would mathematically require a left-shift $\text{shift} < 0$) is **explicitly rejected at compile time** (`ValueError`).
  - Scales $\le 0.0$, `NaN`, and `Inf` are rejected immediately.

### 3.2 Fixed-Point Arithmetic Primitives

#### `saturating_rounding_doubling_high_mul(a, b)`
- Multiplies two 32-bit signed integers and returns the high 32 bits of their doubled 64-bit product with rounding:
  $$\text{product} = a \times b$$
  $$\text{nudge} = \begin{cases} 2^{30} & (\text{product} \ge 0) \\ 1 - 2^{30} & (\text{product} < 0) \end{cases}$$
  $$\text{result} = \text{truncate\_to\_zero}\left( \frac{\text{product} + \text{nudge}}{2^{31}} \right)$$
- **Special Saturation Case**: If $a = -2^{31}$ and $b = -2^{31}$, product would overflow $2^{62}$; returns $2^{31} - 1$ (`INT32_MAX`).

#### `rounding_divide_by_pot(value, exponent)`
- Symmetric round-to-nearest division by power-of-two ($2^{\text{exponent}}$):
  $$\text{magnitude} = \left( |value| + 2^{\text{exponent} - 1} \right) \gg \text{exponent}$$
  $$\text{result} = \begin{cases} -\text{magnitude} & (value < 0) \\ \text{magnitude} & (value \ge 0) \end{cases}$$
- For $\text{exponent} = 0$, returns `value` unmodified.

#### `requantize(accum, multiplier, shift)`
- Computes `scaled = saturating_rounding_doubling_high_mul(accum, multiplier)`.
- Computes `shifted = rounding_divide_by_pot(scaled, shift)`.
- Clamps `shifted` to $[-128, 127]$.

---

## 4. Operation Specifications (The 7 Supported Ops)

### 1. `EMBEDDING`
- **Inputs**: `[tokens]` (Shape: `(N,)`, DType: `UINT8`)
- **Outputs**: `[output]` (Shape: `(N, D)`, DType: `INT8`)
- **Attributes**: `weight` (Name of weight matrix of shape `(256, D)`)
- **Semantics**: Table row lookup without multiplication: `output[i, j] = weight[tokens[i], j]`.

### 2. `CONV1D`
- **Inputs**: `[input]` (Shape: `(C_in, L)`, DType: `INT8`)
- **Outputs**: `[output]` (Shape: `(C_out, L_out)`, DType: `INT8`, requires `QuantParam`)
- **Attributes**: `weight` (`(C_out, C_in, K)`), `stride` ($\ge 1$), `padding` ($\ge 0$)
- **Output Length**: $L_{out} = \lfloor \frac{L + 2 \times \text{padding} - K}{\text{stride}} \rfloor + 1$
- **Semantics**: Cross-channel 1D sliding window MAC accumulated in `int32`, followed by `requantize`. Zero-padding fills out-of-bounds inputs.

### 3. `MATMUL`
- **Inputs**: `[input]` (Shape: `(Rows, Features_in)`, DType: `INT8`)
- **Outputs**: `[output]` (Shape: `(Rows, Features_out)`, DType: `INT8`, requires `QuantParam`)
- **Attributes**: `weight` (Shape: `(Features_out, Features_in)`)
- **Semantics**: Matrix multiplication accumulated in `int32` over `Features_in`, followed by `requantize`.

### 4. `ADD`
- **Inputs**: `[left, right]` (Shape: matching `(D1, ...)`, DType: `INT8`)
- **Outputs**: `[output]` (Shape: matching `(D1, ...)`, DType: `INT8`, requires `QuantParam`)
- **Semantics**: Elementwise addition $acc = left_i + right_i$, followed by `requantize`.

### 5. `LUT` (Lookup Table)
- **Inputs**: `[input]` (Any shape, DType: `INT8`)
- **Outputs**: `[output]` (Same shape, DType: `INT8`)
- **Attributes**: `table` (Sequence of exactly 256 `int8` entries)
- **Semantics**: Direct index mapping for nonlinearities (ReLU, GELU, Tanh, Sigmoid):  
  `output[i] = table[input[i] + 128]`.

### 6. `MAXPOOL1D`
- **Inputs**: `[input]` (Shape: `(..., Length)`, DType: `INT8`)
- **Outputs**: `[output]` (Shape: `(..., L_out)`, DType: `INT8`)
- **Attributes**: `kernel_size` ($\ge 1$), `stride` ($\ge 1$), `padding` ($\ge 0$)
- **Semantics**: Sliding window max-reduction along the final axis. Padding positions do not contribute.

### 7. `ARGMAX`
- **Inputs**: `[input]` (Shape: `(..., Classes)`, total elements $\le 128$, DType: `INT8`)
- **Outputs**: `[output]` (Shape: `()`, scalar DType: `INT8`)
- **Semantics**: Returns the integer index of the maximum element in the flattened tensor. Tie-break: earliest index.

---

## 5. Phase 0 C Function Signature

All graphs emitted by Kiln in Phase 0 expose a single deterministic entry point:

```c
int8_t kiln_predict(const uint8_t *input_bytes, int len);
```

- **Precondition**: `input_bytes != NULL` and `len == graph.tensors[input_name].num_elements`. Otherwise calls `abort()`.
  *(Note for bare-metal targets: On freestanding embedded systems without POSIX `abort()`, `abort()` can be replaced or mapped to a weak trap handler / infinite spin / fault hook `kiln_trap()`)*
- **Return Value**: Scalar `int8_t` representing the predicted class index from the final `ARGMAX` operation.
- **Side Effects**: Modifies only `_arena.data` in `.bss`; `input_bytes` is read-only because graph inputs are consumed in place (zero-copy). Zero heap allocation, zero non-local side effects.
# KIR v1 serialization

KIR v1 is a versioned, language-neutral JSON representation of the validated
graph. The canonical object has `schema: "kiln.kir"` and `version: 1`, followed
by `inputs`, `outputs`, `tensors`, and `nodes`. Tensor dtypes use the enum values
`uint8`, `int8`, and `int32`; quantization is either `null` or an object with a
Q31 `multiplier` and non-negative `shift`. Node attributes contain JSON values
only. The Python codec is available through `kiln.ir.serialization` and always
re-validates a graph after decoding.

KIR v1 intentionally has no dynamic dimensions, executable attributes, Python
class names, or implicit broadcasting. A frontend must reject unsupported
operations, dynamic control flow, and shape-dependent behavior with a structured
diagnostic before serialization. MessagePack may be added as a transport later;
it must preserve this schema and version contract.

# Static safety certificate assumptions

`kiln.analysis.interval` performs interval boundary propagation over the full
declared input domain (`[0, 255]` for `uint8`, `[-128, 127]` for `int8`). The
certificate is sound for the emitted program when the graph, weights, and
integer semantics remain unchanged. It checks accumulator bounds before
requantization, then records pre-clamp ranges and saturation. This is a static
analysis result, not a formal proof of the analyzer implementation; CBMC/SMT
validation is a planned follow-up.
