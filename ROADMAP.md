# Kiln Architectural Roadmap: From Core to Universal Micro-Compiler

**Goal**: Build the "QEMU of TinyML" — an open, zero-malloc, bit-exact compiler framework that allows anyone to forge deterministic, specialized C99 inference engines from a simple task specification (`kiln.yaml`).

---

## Roadmap Overview

| Phase | Title | Core Objective | Key Deliverables & Milestones | Status |
| :---: | :--- | :--- | :--- | :---: |
| **0** | **核の確立 (Core Invariants)** | 決定論的整数コンパイラ核の完成 | 5核ファイル、KIR.md凍結、差分・極値・タイブレーク検証 (全50件グリーン) | **DONE** |
| **1** | **バックエンド単体の実証 (Backend Validation)** | 量子化・C生成・予算証明書の実効性証明 | MLPerf Tiny KWS (Speech Commands v2, 12クラス, 38,173 params, 89.10%) + PCM→C E2E parity | **DONE** |
| **2** | **データ生成パイプラインの実証 (Data Synthesis)** | 教師LLM合成データ→蒸留の実効性証明 | Python危険API検知タスク、メタプロンプト生成、差分ファジング蒸留、fastText対決 | UPCOMING |
| **3** | **一般化 (The "QEMU Moment")** | 任意タスクを宣言的仕様から自動導出 | `kiln.yaml`、自動タスク振り分け、アーキテクチャ探索 (RAM/ROM/レイテンシ パレート解) | PLANNED |
| **4** | **信頼レイヤー (Certified Trust)** | 組み込み運用に耐えうる実証的保証 | 実機ファーム検証 (Pi/Cortex-M)、残存能力監査 (Linear Probing)、区間境界伝播、`kiln link` | PLANNED |
| **5** | **ムーンショット (Moonshots)** | 核を汚さない実験ブランチでの賭け | 論理ゲート網バックエンド、CBMC/Frama-C 形式検証、決定木/DNF 抽出、オンデバイス整数SGD | EXPERIMENTAL |
| **6** | **公開・エコシステム (Ecosystem)** | 研究者・エンジニアに愛されるOSS化 | `qemu-user` クロス検証、仕様書先出し、再現可能ビルド、3コマンドのHello World | FUTURE |

---

## Phase 1 Detail: Keyword Spotting (KWS) Backend Showdown

### Motivation & Strategic Scope
Phase 1 intentionally **decouples** the compiler backend from LLM synthetic data generation.
By utilizing the standard **Speech Commands v2** dataset and comparing directly against the **MLPerf Tiny DS-CNN small benchmark (38.6K params, 12 classes, 92.2% accuracy)**, we provide irrefutable proof that Kiln's integer arithmetic, memory planning, and C99 generation achieve state-of-the-art efficiency on microcontrollers.

### Two-Stage Execution Strategy
1. **Phase 1a: Classifier Core Proof**
   - Offline computed log-mel spectrograms (e.g. 10x49 int8 tensors).
   - Classifier network built strictly using Kiln's 7 operations: `CONV1D` + `LUT` + `MAXPOOL1D` + `MATMUL` + `ARGMAX(12)`.
   - Verification: FP32 PyTorch vs Kiln Bit-Exact C parity, static budget audit (RAM < 2KB, ROM < 40KB).
2. **Phase 1b: Fixed-Point C Audio Frontend**
   - Hand-written, audited C module for integer FFT and log-mel filterbanks.
   - Separate from the compiler core, connected as a modular runtime preprocessor.
   - End-to-end: Raw WAV audio -> C frontend -> Kiln C classifier -> 12-class prediction.

---

## Phase 1 Close-Out (2026-09-26)

**Status**: 1a DONE, 1b DONE.  Phase 2 may start; the open items are listed
under "Known gaps" below.

### 1. Phase 1a certificate (shipped artifact, frozen split seed 1337)

Reproduce with `python -m examples.kws.train_aug` (300 epochs, no augmentation,
`examples/kws/kws_qat_weights.pt`) and `python -m examples.kws.evaluate`
(734 held-out clips, never seen in training or model selection).  Training is
only close-to-reproducible — three runs of that command span ~1 point of test
accuracy (finding 5); the emitted C and every number below re-derive exactly
from the shipped `.pt` file.

| Metric | Target | Measured |
| :--- | :--- | :--- |
| Arena RAM (`.bss`, zero-malloc) | < 2,048 B | **1,972 B** |
| Constant ROM (int8 weights + LUT tables) | < 40,960 B | **39,197 B** (38,173 + 1,024) |
| Parameters | ~38.6K (MLPerf DS-CNN small) | **38,173** |
| MACs per inference | — | **310,382** |
| Kiln C99 int8 top-1 accuracy | ~92.2% (MLPerf reference) | **89.10%** (654/734) |
| PyTorch FP32 top-1 (same int8 inputs) | — | 88.69% |
| FP32 vs C99 prediction agreement | — | **99.18%** |
| Quantization accuracy delta | >= 0 | **+0.41%** |
| Ops used | Kiln's 7 ops only | 5x `CONV1D`, 4x `LUT`, 4x `MAXPOOL1D`, 1x `ARGMAX` |

Validation-split accuracy of the same checkpoint: FP32 91.27% / C99 91.00%.
The accuracy column carries the +/-2.2 point binomial spread of a 734-clip test
set.  Four runs of the shipped recipe (two seeds, 300 epochs, no augmentation)
produced FP32 test accuracy 90.05 / 89.78 / 89.51 / 88.69 with validation
90.59 / 91.13 / 91.00 / 91.27, so the shipped checkpoint is the
*highest-validation* run even though it is not the highest-test one —
re-selecting on test accuracy would be test-set leakage (finding 5 below).

### 2. Where the sub-2 KB file comes from

`kiln.ir.memory` applies two backend-wide reductions, and `kiln.ir.arena` places
the resulting blocks; both are verified in `tests/test_inplace.py` and
`tests/test_arena.py`:

| Arena plan | Bytes |
| :--- | ---: |
| First-Fit over every tensor, graph input copied into the arena | 3,288 |
| + in-place `LUT` fusion (4 aliases: `t_relu0..3 -> t_conv0..3`) | 2,464 |
| + zero-copy graph inputs (`const uint8_t *input_bytes`) | **1,972** |

1,972 B is that graph's floor rather than a First-Fit artefact: the first stage's
1,324 B output block is still live while its 648 B pooled tensor is written
(live peak 1,972 B), so no placement can beat it, and the allocator rewrite in
section 6 finding 2 leaves the shipped layout byte-identical.

That is **-1,316 B (-40%)** without changing the arithmetic: the emitted C is
still bit-exact against `kiln.ref.run_ref` (25/25 samples in `tests/test_kws.py`,
24/24 utterances in `tests/test_kws_e2e.py`).

### 3. Architecture search (all measured, frozen split, no augmentation)

`train_aug.py --architecture <preset>` lists the layouts; `arena`/`ROM` are the
audited numbers of the emitted C, `test` is the script's FP32 top-1 on the 734
held-out clips (the emitted-C figure for the shipped row is section 1's 89.10%).

| Preset | Pooled frames | Params | Arena | ROM | Test accuracy (seeds) |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `wide_second` (26/64), 120 ep | 24 | 38,548 | 2,160 B | 39,572 B | 88.42 / 88.01 |
| `wide_second`, 300 ep | 24 | 38,548 | 2,160 B | 39,572 B | **90.60** |
| `narrow_second` (27/56), 300 ep | 24 | 38,590 | **1,992 B** | 39,614 B | **90.74** / 89.92 |
| `shipped` (27/55), 300 ep | 24 | 38,173 | **1,972 B** | 39,197 B | 90.05 / 89.78 / 89.51 / 88.69 |
| `stage3` (31/64), 300 ep | 16 | 36,302 | 2,016 B | 37,326 B | 89.65 / 88.69 |
| `pool5` (35/128), 300 ep | 9 | 33,110 | 2,032 B | 33,878 B | 89.24 |
| `pool4` (33/128), 120 ep | 12 | 32,242 | 2,016 B | 33,010 B | 87.74 |
| `small` (32/48), 120 ep | 24 | 6,144 | 2,336 B | 6,656 B | 90.87 |

The arena column is re-audited with `kiln.ir.arena` (section 6 finding 2);
only `narrow_second` and `wide_second` move (3,316 -> 1,992 B and
3,436 -> 2,160 B), everything else was already at its live peak.

The arena is bounded by **two** live-peak terms, not one: while a stage's pooled
tensor is still live the next convolution writes its own output, so a first
stage with `c0` channels, `L0` pooled frames and a `c1`-channel successor needs
`max(c0 * (49 + L0), L0 * (c0 + c1))` bytes.  `kiln.ir.arena` reaches that bound
for every layout in the table above, so the budget reduces to two linear
constraints: with 24 pooled frames, `73 * c0 <= 2048` and
`24 * (c0 + c1) <= 2048` before alignment, i.e. `c0 <= 28` and `c0 + c1 <= 85`
(sweep: 27/58 -> 2,040 B, 28/57 -> 2,044 B, 27/59 -> 2,064 B is over).  The old
First-Fit-only planner needed the harsher `c1 * L0 <= c0 * 49` instead, which
capped a 27-channel first stage at 55 channels: `c1 = 56` needs 1,344 B for a
1,324 B freed block, so First-Fit appended the whole block after the live pooled
tensor and the arena jumped from 1,972 B to 3,316 B -- 1,344 B of RAM for a 20 B
shortfall.  That cliff is what forced the best layout measured on this corpus
(`narrow_second`: val 91.54 / test 90.74 FP32, 91.68 / 91.01 through the emitted
C, seed 777) over budget; the interval-aware allocator packs it into 1,992 B,
exactly its own live peak, and raises the compliant ceiling to `c1 <= 58` at
`c0 = 27` (2,040 B) or `c1 <= 57` at `c0 = 28` (2,044 B).  Whether the
certificate should therefore ship `narrow_second` instead of the 27/55 layout is
a separate decision -- section 6 finding 2 records the allocator change itself.
Both layouts stay reproducible:
`python -m examples.kws.qat_eval --architecture narrow_second --weights examples/kws/kws_narrow_second_weights.pt`
reprints 91.01% (C99) / 90.74% (FP32) on the frozen split.

### 4. Findings that shaped the shipped recipe

1. **Training budget beats architecture here.** Going from 120 to 300 epochs
   moved the reference layout from 88.42% to 90.60% test, and from 88.96% to
   89.65% (best of two seeds) for `stage3`; no compliant layout beat the long
   schedule.  Every compliant candidate lands in 87.7-90.1%.
2. **Augmentation hurt.** With identical architecture and seed, the int8-domain
   augmentation preset scored 85.69% vs 88.42% test on the reference layout and
   87.06% vs 90.87% on the small layout (120 epochs), so the shipped checkpoint
   uses none (the switches remain behind `--augment` for larger corpora).  An
   earlier bug fed `/127`-scaled values into the int8-domain augmenter (a ~127x
   scale error, visible as 11.04% accuracy) and had to be fixed first.
3. **The dB domain is not deployable.** Training the same architecture on raw dB
   inputs (`train.py`) and then lowering it scores **50.54%** through the emitted
   C (86.38% in FP32), because Kiln's symmetric zero-point-free int8 contract
   cannot represent the `V_mid = -40 dB` offset.  Training on `q / 127` is
   therefore not a trick but a requirement; `python -m examples.kws.ptq_test` and
   `python -m examples.kws.qat_eval` print both numbers.
4. **Selection noise.** With 734 test clips the 95% confidence interval is about
   +/-2.2 points, so the 1-point gaps between compliant layouts are not
   significant.  The shipped layout was selected by *validation* accuracy
   (91.27% vs 90.59% across seeds) and is the closest match to the MLPerf
   DS-CNN footprint.
5. **Training is close-to-reproducible; the compiler is exactly reproducible.**
   `python -m examples.kws.train_aug` re-creates the shipped recipe from scratch
   (same split, same seeds, same schedule), but three runs of the identical
   command gave FP32 val 91.13 / 91.27 / 91.00 and FP32 test 89.78 / 88.69 /
   89.51 (a second seed adds 90.59 / 90.05) — CUDA convolution reductions
   (TF32, non-deterministic cuDNN kernels)
   are not bit-stable, so a re-run lands within about a point instead of
   bit-identically, and validation-driven selection can therefore pick a run
   whose test score is not the best of the three (as it did here).  Everything
   downstream of the checkpoint *is* deterministic: `emit_c`, `audit_graph`, the
   gcc-built benchmark and `kiln.ref.run_ref` take no random input, and every
   test in `tests/` is an equality or bit-exactness check — re-emitting the C
   from the shipped checkpoint reproduces 39,197 B of ROM and 1,972 B of arena
   on every run.  A bit-identical *checkpoint* would need a deterministic
   training setup (CPU training or `torch.use_deterministic_algorithms(True)`);
   that was not needed for Phase 1, whose artifact is the frozen `.pt` file.

### 5. Phase 1b certificate (raw PCM -> label, all C)

`tests/test_kws_frontend.py` proves the fixed-point frontend reproduces the
training log-mel within one int8 step (usually bit-exact, >= 90% of cells exact).
`tests/test_kws_e2e.py` (new) links `kws_frontend.c` + `kws_pipeline.c` + the
*emitted* graph for the shipped checkpoint and checks 24 real clips (two per
class, taken across the archive):

| Check | Result |
| :--- | :--- |
| Generated graph vs `kiln.ref.run_ref` on the frontend's own features | 24/24 bit-exact |
| PCM-to-label path vs the PyTorch pipeline | 24/24 identical labels |
| PCM-to-label top-1 accuracy against ground truth | 23/24 = 95.8% (asserted >= 75%) |

### 6. Known gaps / follow-ups

1. **92.2% is not reached (89.10% int8 C for the shipped checkpoint; the best
   measured int8 C run of this layout is 90.05%).** The
   Phase 1a corpus is 4,890 clips with
   3,423 training clips, while the MLPerf DS-CNN reference trains on the full
   ~105K-clip set.  Training accuracy saturates at 99.85% while validation
   plateaus near 91%, i.e. the model is data-limited, not compiler-limited; the
   2 KB arena cap costs at most a few tenths of a validation point (the
   over-budget twins in section 3), so it is not the binding constraint either.
   Acquiring the full corpus is the cheapest next gain (the frontend pipeline is
   already in place for it).
2. **First-Fit left memory on the table -- fixed.** Placement now lives in
   `kiln.ir.arena`: it evaluates the historical First-Fit insertion layout plus
   size-descending first-fit/best-fit candidates and keeps the smallest, so it
   can only ever shrink the arena and never changes a layout that was already
   minimal.  Measured on section 3's presets: shipped 1,972 B (unchanged, it is
   the live peak), `narrow_second` 3,316 -> **1,992 B** (compliant again),
   `wide_second` 3,436 -> 2,160 B (still over: its live peak is 2,160 B).  The
   compliant search space therefore grows from `c1 <= 55` to `c1 <= 58` at
   `c0 = 27`; re-searching architectures inside the wider bound, and deciding
   which checkpoint the certificate ships, are Phase 3 items.

3. `MATMUL` is part of the op set but unused by the shipped graph; Phase 2
   (fastText-shaped distillation) is the natural place to exercise it.

### 7. Repository hygiene (done before Phase 2)

Phase 1 was developed without version control; the tree is now a git repository
(initial commit on `main`) with:

- **`.gitignore`/`.gitattributes`** that keep the 107 MB Speech Commands archive
  and all build scratch out of history, while tracking the frozen certificate
  inputs: the 9.2 MB precomputed corpus and both 0.1 MB checkpoints (verified
  byte-identical to the working tree after staging, and `.pt` blobs are marked
  binary so nothing rewrites them).
- **`README.md`** (install, three-command quick start, the Phase 1 certificate
  with its caveats, dataset policy), **`LICENSE`** (MIT) and **`pyproject.toml`**
  licence/classifier metadata.
- **`.github/workflows/ci.yml`**: CPU-only PyTorch, `pip install -e ".[test]"`,
  a CLI smoke test, the certificate command and the full suite on every push.
  The archive-dependent tests report `skipped` on CI; the KWS
  accuracy/budget cheque runs, because the corpus and checkpoint are tracked.
- **Two robustness fixes** found by simulating CI (running the suite with the
  data files moved away): `tests/test_kws_e2e.py` skipped on a missing
  checkpoint but then raised `FileNotFoundError` on a missing corpus, and
  `tests/test_kws.py` resolved its data paths relative to the current working
  directory instead of the repository.
- **`kiln/cli.py`**: `--runtime-header None` used to emit `#include "None"`; it
  now produces a self-contained translation unit (regression-tested, compiles
  with `gcc -std=c99 -Wall -Wextra -Werror -pedantic`), which is what the
  README's two-line build example relies on.

---

## Phase 2 Detail (DRAFT): Synthetic-Data Distillation Pipeline

**Status**: DRAFT, not started.  Phase 1 proved the *back half* of the framework on a
public dataset with no LLM in the loop; Phase 2 has to prove the *front half* — the half
that motivated Kiln in the first place.  The tasks Kiln is meant to serve (detecting a
dangerous API call, a policy violation, a suspicious byte pattern) are the ones **without**
an off-the-shelf labeled corpus, so the labels have to be synthesized by a teacher LLM and
compressed into a <2 KB int8 student.

### Target task

**Python dangerous-API detection**: classify a short source snippet by the hazard it
contains (e.g. `eval`/`exec`, `pickle.loads`, `subprocess` with `shell=True`, `os.system`,
`yaml.load` without a safe loader, `tempfile.mktemp`, `assert`-based validation ...).
The output is a small label set; the input is text, so the graph is
`EMBEDDING`/`MATMUL`-shaped — deliberately the opposite of Phase 1's convolution stack,
which is what makes it a *generality* proof rather than a second benchmark.

### Design gates (each one is a Phase 1 lesson, applied in advance)

| Gate | Rule | Phase 1 precedent |
| :--- | :--- | :--- |
| **G1 Feature contract first** | Freeze the exact integer tokenizer/feature contract *before* any training run; the C runtime sees `uint8` tokens and int8 features, never floats or dB | §4 finding 3 — dB-domain training scored 50.54 % through the emitted C |
| **G2 Frozen corpus** | The synthesized dataset is a checked-in artifact with a manifest (teacher model + version, prompt hash, decoding params, seed, corpus hash). No live API calls inside tests or pipelines | the frozen split seed (1337) and the tracked 9.2 MB corpus |
| **G3 Real held-out set** | "Synthetic data works" is only meaningful against *real, never-synthesized* code, and against a naive baseline (regex/allow-list) and fastText | §3's `narrow_second` twin: accuracy claims need a control |
| **G4 Equal-budget showdown** | fastText is compared at the *same* footprint (<= 2 KB arena / <= 40 KB ROM), not at equal accuracy | §3's arena cliff (closed by `kiln.ir.arena`) |
| **G5 Selection by validation** | Test accuracy is reported, never used to pick a checkpoint; report confidence intervals | §4 finding 4 — 734 clips is +/-2.2 points |

### Milestones

| # | Deliverable | Acceptance test |
| :---: | :--- | :--- |
| M2.0 | Feature-contract spec (tokenizer, vocabulary size, int8 packing, unknown-token policy) | written in this file + round-trip unit test |
| M2.1 | Meta-prompt generator + synthesis harness producing the frozen corpus | manifest hash reproduces the corpus bit-exactly |
| M2.2 | int8 student (EMBEDDING + MATMUL + LUT + ARGMAX) trained by distillation, lowered to Kiln IR | arena/ROM audited; `MATMUL` and `EMBEDDING` leave the "unexercised" list |
| M2.3 | Showdown report: Kiln student vs fastText vs rules baseline, with CIs | numbers in this section, reproducible by one command |
| M2.4 | PCM→C equivalent: raw source text -> fixed-point tokenizer -> emitted graph, all C | a Phase 1b-style `tests/test_*.py` bit-exactness check |
| M2.5 | Differential-fuzzing distillation loop (unlabeled code as a fuzzing oracle) | documented as an experiment, success or failure |

### Open questions (to settle before M2.1)

1. **Teacher**: which model, and is the generated corpus shippable (licence/quality)?  A
   frozen corpus makes the pipeline reproducible but hides prompt-iteration cost; live calls
   make it cheap to re-run but non-reproducible.  Proposal: freeze the corpus, keep the
   generator script + manifest in-tree.
2. **Label space**: one hazard class per snippet (single-label) or multi-label?  Single-label
   keeps `ARGMAX` as the graph exit and matches Phase 1's contract.
3. **Real held-out source**: which public, licence-compatible code supplies the *real*
   evaluation set (and how to guarantee it never enters synthesis)?
4. **Distillation signal**: hard labels only, or teacher logits (soft targets)?  Only hard
   labels survive Kiln's integer contract without a new "soft-label" training path.
5. **Budget target**: is <= 2 KB the right student size for a vocabulary-sized `MATMUL`, or
   does this task need a different (ROM-heavy, RAM-light) envelope?


