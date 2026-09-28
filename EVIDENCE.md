# Kiln -- Evidence

Every number below is the verbatim output of a command executed by
`scripts/reproduce_evidence.py` against this repository on
2026-09-28 17:25 (Windows 10,
Python 3.11.15, torch 2.5.1+cu121, gcc (Rev3, Built by MSYS2 project) 16.2.0).
Re-run the script to regenerate this file. The numbers are verbatim run output;
the section headings, the one-liner and the Honest caveats are authored.

    python scripts/reproduce_evidence.py

Submission one-liner (verified by sections 2-13 below):

> **Kiln compiles a PyTorch model into a single zero-malloc, floating-point-free,
> dependency-free C99 file, and refuses a dangerous model at build time.**
> The accumulator-safety gate is fail-closed: a graph whose int32 accumulator
> could overflow, or whose interval audit cannot run, produces a certificate
> that says so -- and *no* C file.

### 1. Test suite (every guarantee)

```text
$ python -m pytest tests/ -q
........................................................................ [ 26%]
........................................................................ [ 53%]
........................................................................ [ 80%]
......................................................                   [100%]
270 passed in 69.37s (0:01:09)
# exit code: 0
```

### 2. Showcase 1 -- one C file, budget gates green

```text
$ python -m kiln build examples/hazard/kiln.yaml --runtime-header None
/*
 * =========================================================================
 *                  Kiln Model Static Budget Certificate
 * =========================================================================
 *   Arena (.bss) RAM      :    1,344 bytes
 *   Peak Concurrent RAM   :    1,344 bytes
 *   Weights ROM           :    2,096 bytes
 *   LUT Tables ROM        :        0 bytes
 *   Total Constant ROM    :    2,096 bytes
 * -------------------------------------------------------------------------
 *   Total MAC Operations  :    4,608 ops
 *   Total Add Operations  :        0 ops
 *   Estimated Latency     :   24,618 cycles (upper bound)
 * -------------------------------------------------------------------------
 *   Dynamic Allocation    :        0 bytes (strictly zero-malloc guarantee)
 *   Determinism           : Bit-exact verified (C99 fixed-point integer)
 *   Accumulator Safety    : Verified int32 safe [-67640, +61845]
 * =========================================================================
 */
[BUDGET] arena RAM  1,344 <= 2,048 ... OK
[BUDGET] total ROM  2,096 <= 40,960 ... OK
[SUCCESS] Built hazard-sniffer -> C:\Users\smily\Kiln\examples\hazard\build\model.c (Arena RAM: 1,344 B, Total ROM: 2,096 B, MACs: 4,608)
# exit code: 0
```

### 3. Showcase 1 -- clean compile under -Werror -pedantic, nm -u

```text
$ gcc -std=c99 -O2 -Wall -Wextra -Werror -pedantic -c model.c -o model.o && nm -u model.o
nm -u:
                 U abort
# exit code: 0
```

`model.c`: 14,656 B; unresolved symbols: abort; heap symbols: none; only-abort: True


### 4. Showcase 2 -- the dangerous model is refused, nothing written

```text
$ python -m kiln compile examples/hazard/dangerous_model.py -o dangerous.c
/*
 * =========================================================================
 *                  Kiln Model Static Budget Certificate
 * =========================================================================
 *   Arena (.bss) RAM      :       16 bytes
 *   Peak Concurrent RAM   :       16 bytes
 *   Weights ROM           : 1,400,000 bytes
 *   LUT Tables ROM        :        0 bytes
 *   Total Constant ROM    : 1,400,000 bytes
 * -------------------------------------------------------------------------
 *   Total MAC Operations  : 1,400,000 ops
 *   Total Add Operations  :        0 ops
 *   Estimated Latency     : 7,000,030 cycles (upper bound)
 * -------------------------------------------------------------------------
 *   Dynamic Allocation    :        0 bytes (strictly zero-malloc guarantee)
 *   Determinism           : Bit-exact verified (C99 fixed-point integer)
 *   Accumulator Safety    : WARNING: Potential int32 overflow detected
 * =========================================================================
 */

# -- stderr --
[SAFETY VIOLATION] int32 accumulator overflow detected in examples/hazard/dangerous_model.py; compilation aborted, no C emitted
# exit code: 1
```

exit code 1 (non-zero = refused); `[SAFETY VIOLATION]` on stderr: True; `dangerous.c` written: False


### 5. KWS -- shipped checkpoint through the emitted C

```text
$ python -m examples.kws.qat_eval
QAT (shipped) checkpoint: C99 int8 91.01% | FP32 90.74%
  quantization delta : +0.27%
  checkpoint         : C:\Users\smily\Kiln\examples\kws\kws_qat_weights.pt
  params             : 38,590
  evaluated          : 734 held-out samples
# exit code: 0
```

### 6. KWS -- PTQ baseline (what skipping QAT costs)

```text
$ python -m examples.kws.ptq_test
PTQ baseline (dB-domain FP32 training -> quantized C): 50.54%
  checkpoint : C:\Users\smily\Kiln\examples\kws\kws_trained_weights.pt
  evaluated  : 734 held-out samples through the emitted C99 graph
# exit code: 0
```

### 7. KWS -- static footprint certificate

```text
$ python -m kiln audit examples/kws/model.py
/*
 * =========================================================================
 *                  Kiln Model Static Budget Certificate
 * =========================================================================
 *   Arena (.bss) RAM      :    1,992 bytes
 *   Peak Concurrent RAM   :    1,992 bytes
 *   Weights ROM           :   38,590 bytes
 *   LUT Tables ROM        :    1,024 bytes
 *   Total Constant ROM    :   39,614 bytes
 * -------------------------------------------------------------------------
 *   Total MAC Operations  :  314,342 ops
 *   Total Add Operations  :        0 ops
 *   Estimated Latency     : 1,587,715 cycles (upper bound)
 * -------------------------------------------------------------------------
 *   Dynamic Allocation    :        0 bytes (strictly zero-malloc guarantee)
 *   Determinism           : Bit-exact verified (C99 fixed-point integer)
 *   Accumulator Safety    : Verified int32 safe [-385191, +220602]
 * =========================================================================
 */
# exit code: 0
```

### 8. KWS -- Phase 1a benchmark table (full parity)

```text
$ python -m examples.kws.evaluate
=================================================================
      PHASE 1a BENCHMARK RESULTS: SPEECH COMMANDS v2 (12-CLASS)
=================================================================
  Target Dataset               : Speech Commands v2 (12 Classes)
  Test Samples Evaluated       : 734 held-out samples
  PyTorch FP32 Top-1 Accuracy  : 90.74%
  Kiln C99 int8 Top-1 Accuracy : 91.01% (668/734)
  Quantization Accuracy Delta  : +0.27%
  FP32 vs C99 Prediction Match : 98.50% agreement
  Validation Split (selection) : FP32 91.54% / C99 91.68%
  Int8 Weights ROM             : 38590 B (1024 B of LUT tables)
  Arena (.bss) RAM             : 1992 B (target < 2048 B)
  Constant ROM (weights+LUTs)  : 39614 B (target < 40960 B)
  MACs per inference           : 314,342
=================================================================
# exit code: 0
```

### 9. KWS -- fresh training run (CUDA; documented ~1-point spread)

```text
$ python -m examples.kws.train_aug --out <tmp>/kws_qat_weights_repro.pt
architecture=shipped params=38,590 train=3423 val=733 test=734 device=cuda
epoch   50/300 | loss 0.3192 | train 99.91% | val 90.31% | best 90.86%
epoch  100/300 | loss 0.3084 | train 100.00% | val 90.86% | best 91.00%
epoch  150/300 | loss 0.2994 | train 100.00% | val 91.00% | best 91.54%
epoch  200/300 | loss 0.2980 | train 100.00% | val 91.27% | best 91.81%
epoch  250/300 | loss 0.2971 | train 100.00% | val 91.00% | best 91.81%
epoch  300/300 | loss 0.2969 | train 100.00% | val 91.00% | best 91.81%
RESULT {"architecture": "shipped", "parameters": 38590, "epochs": 300, "val_accuracy": 91.81, "test_accuracy": 89.51, "test_samples": 734, "seconds": 76.4, "output": "C:\\Users\\smily\\AppData\\Local\\Temp\\tmpczoe0qpn\\kws_qat_weights_repro.pt"}
# exit code: 0
```

### 10. KWS -- the retrained checkpoint through the emitted C

```text
$ python -m examples.kws.qat_eval --weights <tmp>/kws_qat_weights_repro.pt
QAT (shipped) checkpoint: C99 int8 89.24% | FP32 89.51%
  quantization delta : -0.27%
  checkpoint         : C:\Users\smily\AppData\Local\Temp\tmpczoe0qpn\kws_qat_weights_repro.pt
  params             : 38,590
  evaluated          : 734 held-out samples
# exit code: 0
```

### 11. Hazard -- char n-gram + LogisticRegression baseline (provenance, split, confusion)

```text
$ python -m examples.hazard.baseline_eval
================================================================================
HAZARD DETECTION BASELINE: char n-gram (2-5) + Logistic Regression
================================================================================

Data split (seed 1337):
  Train: 210 (70%)
  Val:   45 (15%) - not used for baseline
  Test:  45 (15%)
  Held-out: 60 (real snippets)

Extracting character n-grams (2-5)...
  Vocabulary size: 13061 n-grams
  Train+Val shape: (255, 13061)
  Test shape:      (45, 13061)
  Held-out shape:  (60, 13061)

Training Logistic Regression (C=1.0, multinomial)...

================================================================================
RESULTS
================================================================================
Test accuracy (n=45):       100.00%
Held-out accuracy (n=60): 100.00%

Test Confusion Matrix
================================================================================
                            benigndynamic_execdeserializatshell_inject    tempfileunsafe_valid
benign                           5           0           0           0           0           0
dynamic_exec                     0           5           0           0           0           0
deserialization                  0           0           7           0           0           0
shell_injection                  0           0           0          14           0           0
tempfile                         0           0           0           0           6           0
unsafe_validation                0           0           0           0           0           8

Held-out Confusion Matrix
================================================================================
                            benigndynamic_execdeserializatshell_inject    tempfileunsafe_valid
benign                          10           0           0           0           0           0
dynamic_exec                     0          10           0           0           0           0
deserialization                  0           0          10           0           0           0
shell_injection                  0           0           0          10           0           0
tempfile                         0           0           0           0          10           0
unsafe_validation                0           0           0           0           0          10

Per-Class Test Accuracy
----------------------------------------
  benign                : 100.0%
  dynamic_exec          : 100.0%
  deserialization       : 100.0%
  shell_injection       : 100.0%
  tempfile              : 100.0%
  unsafe_validation     : 100.0%

Per-Class Held-out Accuracy
----------------------------------------
  benign                : 100.0%
  dynamic_exec          : 100.0%
  deserialization       : 100.0%
  shell_injection       : 100.0%
  tempfile              : 100.0%
  unsafe_validation     : 100.0%

Top 10 n-grams per class (by coefficient magnitude):
--------------------------------------------------------------------------------

benign:
  mp                   coef=-0.2223
   =                   coef=+0.2060
   =                   coef=+0.1920
  =                    coef=+0.1860
  pi                   coef=-0.1417
  al                   coef=-0.1383
  tmp                  coef=-0.1231
  tm                   coef=-0.1231
  ser                  coef=-0.1220
  ev                   coef=-0.1199

dynamic_exec:
  ev                   coef=+0.2553
  eval                 coef=+0.2461
  eva                  coef=+0.2461
  al(                  coef=+0.2461
  val(                 coef=+0.2461
  eval(                coef=+0.2461
  va                   coef=+0.2453
  ex                   coef=+0.2358
  al                   coef=+0.2282
  l(                   coef=+0.2259

deserialization:
  e.                   coef=+0.2006
  .load                coef=+0.1847
  .loa                 coef=+0.1847
  oa                   coef=+0.1818
  oad                  coef=+0.1818
  loa                  coef=+0.1792
  load                 coef=+0.1792
  .lo                  coef=+0.1721
  .l                   coef=+0.1687
  ad                   coef=+0.1652

shell_injection:
  s.                   coef=+0.2391
  os                   coef=+0.2091
  os.                  coef=+0.2022
   {                   coef=+0.1798
  (f'                  coef=+0.1678
  f'                   coef=+0.1593
  (f                   coef=+0.1557
  }'                   coef=+0.1393
  =                    coef=-0.1378
   =                   coef=-0.1366

tempfile:
  mp                   coef=+0.4557
  tm                   coef=+0.3438
  tmp                  coef=+0.3438
  /t                   coef=+0.2549
  /tm                  coef=+0.2522
  /tmp                 coef=+0.2522
  mp/                  coef=+0.2346
  /tmp/                coef=+0.2346
  tmp/                 coef=+0.2346
  p/                   coef=+0.2346

unsafe_validation:
  er                   coef=+0.3006
  se                   coef=+0.2758
  ser                  coef=+0.2688
  ert                  coef=+0.2660
  ss                   coef=+0.2610
  sert                 coef=+0.2588
  sert                 coef=+0.2588
  asser                coef=+0.2588
  ssert                coef=+0.2588
  sser                 coef=+0.2588

Results saved to C:\Users\smily\Kiln\examples\hazard\baseline_results.json
# exit code: 0
```

### 12. Hazard -- equal-budget showdown (M2.3)

```text
$ python -m examples.hazard.benchmark
train 210 (val 45 for the student only) / test 45 / held-out 60

============================================================================================================================
PHASE 2 M2.3 HAZARD SHOWDOWN (frozen test n=45, real held-out n=60, Wilson 95% CI)
============================================================================================================================
model                             footprint                                     test                  held-out             
---------------------------------------------------------------------------------------------------------------------------
Kiln int8 student (emitted C)     ROM 2,096 B; arena 1,344 B                    73.3% [59.0, 84.0]    73.3% [61.0, 82.9]   
fastText (dim 8, no subwords)     461 words x dim 8: 21.2 KB f32, 10.2 KB int8  77.8% [63.7, 87.5]    68.3% [55.8, 78.7]   
Rules baseline (regex control)    377 B of regex source                         86.7% [73.8, 93.7]    76.7% [64.6, 85.6]   
LogisticRegression (char 2-5)     11123 char n-grams: 304 KB f32 OVER BUDGET    97.8% [88.4, 99.6]    98.3% [91.1, 99.7]   

RESULT {"models": {"Kiln int8 student (emitted C)": {"footprint": "ROM 2,096 B; arena 1,344 B", "held_hits": "44", "held_out": "73.3% [61.0, 82.9]", "test": "73.3% [59.0, 84.0]", "test_hits": "33"}, "LogisticRegression (char 2-5)": {"footprint": "11123 char n-grams: 304 KB f32 OVER BUDGET", "held_hits": "59", "held_out": "98.3% [91.1, 99.7]", "test": "97.8% [88.4, 99.6]", "test_hits": "44"}, "Rules baseline (regex control)": {"footprint": "377 B of regex source", "held_hits": "46", "held_out": "76.7% [64.6, 85.6]", "test": "86.7% [73.8, 93.7]", "test_hits": "39"}, "fastText (dim 8, no subwords)": {"footprint": "461 words x dim 8: 21.2 KB f32, 10.2 KB int8", "held_hits": "41", "held_out": "68.3% [55.8, 78.7]", "test": "77.8% [63.7, 87.5]", "test_hits": "35"}}, "n_held_out": 60, "n_test": 45}
# exit code: 0
```

### 13. Hazard data provenance (manifest-pinned)

- teacher model: `meta-llama/llama-3.3-70b-instruct`
- corpus: 300 synthetic snippets, 6 classes, split seed 1337, token count 96, window 48 B
- held-out: 60 real snippets (never synthesized)
- hazard_corpus.json: sha256 MATCHES manifest (`5d00a3a25e2e1286...`)
- held_out_real.json: sha256 MATCHES manifest (`bdc5be967d22a155...`)


## Summary (values extracted mechanically from the sections above)

| check | result | source |
| :--- | :--- | :--- |
| tests | 270 passed in 69.37s (0:01:09) | exit 0 |
| showcase 1 | model.c 14,656 B, nm -u=['abort'] | gcc exit 0 |
| showcase 2 | refused (exit 1), file written: False | fail-closed |
| KWS shipped (734 clips) | QAT (shipped) checkpoint: C99 int8 91.01% \| FP32 90.74% | exit 0 |
| KWS PTQ baseline | PTQ baseline (dB-domain FP32 training -> quantized C): 50.54% | exit 0 |
| KWS certificate (audit) | *   Arena (.bss) RAM      :    1,992 bytes / *   Total Constant ROM    :   39,614 bytes / *   Total MAC Operations  :  314,342 ops / *   Accumulator Safety    : Verified int32 safe [-385191, +220602] | exit 0 |
| KWS Phase 1 table | PyTorch FP32 Top-1 Accuracy  : 90.74% / Kiln C99 int8 Top-1 Accuracy : 91.01% (668/734) / FP32 vs C99 Prediction Match : 98.50% agreement / Arena (.bss) RAM             : 1992 B (target < 2048 B) / MACs per inference           : 314,342 | exit 0 |
| KWS retrain (CUDA) | {"architecture": "shipped", "epochs": 300, "output": "C:\\Users\\smily\\AppData\\Local\\Temp\\tmpczoe0qpn\\kws_qat_weights_repro.pt", "parameters": 38590, "seconds": 76.4, "test_accuracy": 89.51, "test_samples": 734, "val_accuracy": 91.81} | eval exit 0 |
| hazard baseline (sklearn) | Test accuracy (n=45):       100.00% / Held-out accuracy (n=60): 100.00% | exit 0 |
| hazard showdown | Kiln int8 student: test 73.3% / held-out 73.3% | exit 0 |
| hazard provenance | hazard_corpus.json: OK / held_out_real.json: OK | manifest.json |

## Honest caveats

* The Kiln int8 hazard student (73.3%) does **not** beat the sklearn char n-gram baseline (97.8-100%) or even the 377 B regex control (86.7%): the claim is *certification* (2,096 B ROM / 1,344 B arena, int32-safe, bit-exact C, fail-closed refusal), not accuracy.
* CUDA training is not bit-exact (documented spread); the *evaluation* is deterministic -- the emitted C reproduces the checkpoint bit-for-bit.
* `kiln search` currently only has a reference spec for the hazard example; KWS search is not yet available.
* Showcase 3 (QEMU) is skipped honestly when the ARM toolchain is absent.

GIFs: `scripts/render_demo_gif.py` renders `demo_showcase1.gif` / `demo_showcase2.gif` from the same real runs.
