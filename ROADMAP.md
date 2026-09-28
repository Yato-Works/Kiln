# Kiln Architectural Roadmap: From Core to Universal Micro-Compiler

**Goal**: Build the "QEMU of TinyML" — an open, zero-malloc, bit-exact compiler framework that allows anyone to forge deterministic, specialized C99 inference engines from a simple task specification (`kiln.yaml`).

---

## Roadmap Overview

| Phase | Title | Core Objective | Key Deliverables & Milestones | Status |
| :---: | :--- | :--- | :--- | :---: |
| **0** | **核の確立 (Core Invariants)** | 決定論的整数コンパイラ核の完成 | 5核ファイル、KIR.md凍結、差分・極値・タイブレーク検証 (全50件グリーン) | **DONE** |
| **1** | **バックエンド単体の実証 (Backend Validation)** | 量子化・C生成・予算証明書の実効性証明 | MLPerf Tiny KWS (Speech Commands v2, 12クラス, 38,590 params, 91.01%) + PCM→C E2E parity | **DONE** |
| **2** | **データ生成パイプラインの実証 (Data Synthesis)** | 教師LLM合成データ→蒸留の実効性証明 | Python危険API検知タスク、メタプロンプト生成、差分ファジング蒸留、fastText対決 | DONE (M2.5 open) |
| **3** | **一般化 (The "QEMU Moment")** | 任意タスクを宣言的仕様から自動導出 | `kiln.yaml` + `kiln build` (予算ゲート)、コンパイル時アーキテクチャ探索 `kiln search` (RAM/ROM/容量パレート) 完了; 自動タスク振り分けは未着手 | IN PROGRESS |
| **4** | **信頼レイヤー (Certified Trust)** | 組み込み運用に耐えうる実証的保証 | `kiln link` (厳格コンパイル + リンク + イメージ計測 + リプレイ) 完了、区間境界伝播 M4.2 (`kiln audit --intervals` / 証明書の Accumulator Safety 行) 完了; 残存能力監査 (Linear Probing)、実機ファーム検証は未着手 | IN PROGRESS |
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

Reproduce with `python -m examples.kws.train_aug --seed 777` (300 epochs, no
augmentation, `examples/kws/kws_qat_weights.pt`) and `python -m examples.kws.evaluate`
(734 held-out clips, never seen in training or model selection); the layout is
the 27/56 one.  Training is only close-to-reproducible — runs of that command
span ~1 point of test accuracy (finding 5) — while the emitted C and every
number below re-derive exactly from the shipped `.pt` file.

| Metric | Target | Measured |
| :--- | :--- | :--- |
| Arena RAM (`.bss`, zero-malloc) | < 2,048 B | **1,992 B** |
| Constant ROM (int8 weights + LUT tables) | < 40,960 B | **39,614 B** (38,590 + 1,024) |
| Parameters | ~38.6K (MLPerf DS-CNN small) | **38,590** |
| MACs per inference | — | **314,342** |
| Kiln C99 int8 top-1 accuracy | ~92.2% (MLPerf reference) | **91.01%** (668/734) |
| PyTorch FP32 top-1 (same int8 inputs) | — | 90.74% |
| FP32 vs C99 prediction agreement | — | **98.50%** |
| Quantization accuracy delta | >= 0 | **+0.27%** |
| Ops used | Kiln's 7 ops only | 5x `CONV1D`, 4x `LUT`, 4x `MAXPOOL1D`, 1x `ARGMAX` |

Validation-split accuracy of the same checkpoint: FP32 91.54% / C99 91.68%.
The accuracy column carries the +/-2.2 point binomial spread of a 734-clip test
set.  The checkpoint (seed 777, 300 epochs, no augmentation) is both the
*highest-validation* and the highest-test run measured on this corpus, so
validation-driven selection (finding 4 below) no longer trades test accuracy
away.  Its layout was over budget at 3,316 B until the allocator learned to
reuse the block a finished stage frees (finding 2 in section 6); the narrower
27/55 layout it replaced stays available as the `previous` preset, and its four
300-epoch runs spanned FP32 test 88.69-90.05 with validation 90.59-91.27
(`python -m examples.kws.qat_eval --architecture previous --weights
examples/kws/kws_qat_weights_27_55.pt` reprints its v1 certificate).

### 2. Where the sub-2 KB file comes from

`kiln.ir.memory` applies two backend-wide reductions, and `kiln.ir.arena` places
the resulting blocks; both are verified in `tests/test_inplace.py` and
`tests/test_arena.py`:

| Arena plan | Bytes |
| :--- | ---: |
| Every tensor placed, graph input copied into the arena | 3,296 |
| + in-place `LUT` fusion (4 aliases: `t_relu0..3 -> t_conv0..3`) | **1,992** |
| + zero-copy graph inputs (`const uint8_t *input_bytes`) | **1,992** |

The LUT fusion alone already reaches this graph's floor: the first stage's
1,324 B output block stays live while its 648 B pooled tensor is written, and the
second convolution's 1,344 B has to sit next to that pooled tensor, i.e.
648 + 1,344 = 1,992 B, so no placement can beat it.  Dropping the input copy no
longer moves the total (a graph input is only live at op 0, so the allocator
packs it into a gap); it is what lets the C API read the caller's buffer, and it
was worth 492 B under the First-Fit-only planner.

That is **-1,304 B (-40%)** without changing the arithmetic: the emitted C is
still bit-exact against `kiln.ref.run_ref` (25/25 samples in `tests/test_kws.py`,
24/24 utterances in `tests/test_kws_e2e.py`).

### 3. Architecture search (all measured, frozen split, no augmentation)

`train_aug.py --architecture <preset>` lists the layouts; `arena`/`ROM` are the
audited numbers of the emitted C, `test` is the script's FP32 top-1 on the 734
held-out clips (the emitted-C figure for the shipped row is section 1's 91.01%).

| Preset | Pooled frames | Params | Arena | ROM | Test accuracy (seeds) |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `shipped` (27/56), 300 ep | 24 | 38,590 | **1,992 B** | 39,614 B | **90.74** / 89.92 |
| `previous` (27/55), 300 ep | 24 | 38,173 | **1,972 B** | 39,197 B | 90.05 / 89.78 / 89.51 / 88.69 |
| `wide_second` (26/64), 120 ep | 24 | 38,548 | 2,160 B | 39,572 B | 88.42 / 88.01 |
| `wide_second`, 300 ep | 24 | 38,548 | 2,160 B | 39,572 B | **90.60** |
| `stage3` (31/64), 300 ep | 16 | 36,302 | 2,016 B | 37,326 B | 89.65 / 88.69 |
| `pool5` (35/128), 300 ep | 9 | 33,110 | 2,032 B | 33,878 B | 89.24 |
| `pool4` (33/128), 120 ep | 12 | 32,242 | 2,016 B | 33,010 B | 87.74 |
| `small` (32/48), 120 ep | 24 | 6,144 | 2,336 B | 6,656 B | 90.87 |

The arena column is re-audited with `kiln.ir.arena` (section 6 finding 2): the
promoted `shipped` layout moves 3,316 -> 1,992 B and `wide_second`
3,436 -> 2,160 B, while `previous` was already at its live peak (1,972 B) and
stays byte-identical.

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
shortfall.  That cliff is what kept the best layout measured on this corpus
(val 91.54 / test 90.74 FP32, 91.68 / 91.01 through the emitted C, seed 777) out
of the budget; the interval-aware allocator packs it into 1,992 B, exactly its
own live peak, and raises the compliant ceiling to `c1 <= 58` at `c0 = 27`
(2,040 B) or `c1 <= 57` at `c0 = 28` (2,044 B).  The certificate therefore ships
it as `shipped` (section 1), and the 27/55 layout it replaced stays reproducible
as the `previous` preset:
`python -m examples.kws.qat_eval --architecture previous --weights examples/kws/kws_qat_weights_27_55.pt`
reprints v1's 89.10% (C99) / 88.69% (FP32) on the frozen split.

### 4. Findings that shaped the shipped recipe

1. **Training budget beats architecture here.** Going from 120 to 300 epochs
   moved the reference layout from 88.42% to 90.60% test, and from 88.96% to
   89.65% (best of two seeds) for `stage3`; no compliant layout beat the long
   schedule.  Every layout measured under the First-Fit-only planner landed in
   87.7-90.1%; the wider arena ceiling the allocator unlocked bought the best
   layout of all back (91.01% int8 C, section 1).
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
   +/-2.2 points, so sub-point gaps between compliant layouts are not
   significant.  Selection therefore uses *validation* accuracy only: the
   shipped checkpoint is the highest-validation run measured here (91.54% FP32 /
   91.68% C99), and its layout is the closest fit to the MLPerf DS-CNN footprint
   that satisfies both budgets.
5. **Training is close-to-reproducible; the compiler is exactly reproducible.**
   `python -m examples.kws.train_aug --seed 777` re-creates the shipped recipe
   from scratch (same split, same schedule), but CUDA convolution reductions
   (TF32, non-deterministic cuDNN kernels) are not bit-stable, so a re-run
   lands within about a point instead of bit-identically: four 300-epoch runs
   of the previous 27/55 layout spread FP32 validation 90.59-91.27 and test
   88.69-90.05, which is why selection uses the validation split.  Everything
   downstream of the checkpoint *is* deterministic: `emit_c`, `audit_graph`, the
   gcc-built benchmark and `kiln.ref.run_ref` take no random input, and every
   test in `tests/` is an equality or bit-exactness check — re-emitting the C
   from the shipped checkpoint reproduces 39,614 B of ROM and 1,992 B of arena
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

`pytest tests/test_kws_e2e.py -s` reprints all three counts (the test reports
`skipped` when the 107 MB archive is absent, as on CI).

### 6. Known gaps / follow-ups

1. **92.2% is not reached (91.01% int8 C for the shipped checkpoint; the best
   measured int8 C run of the previous 27/55 layout was 90.05%).** The Phase 1a
   corpus is 4,890 clips with 3,423 training clips, while the MLPerf DS-CNN
   reference trains on the full ~105K-clip set.  Training accuracy saturates at
   99.85% while validation plateaus near 91.5%, i.e. the model is data-limited,
   not compiler-limited, and the 2 KB arena cap no longer excludes the best
   layout either (finding 2 brought the promoted one from 3,316 B to 1,992 B).
   Acquiring the full corpus is the cheapest next gain (the frontend pipeline is
   already in place for it).
2. **First-Fit left memory on the table -- fixed.** Placement now lives in
   `kiln.ir.arena`: it evaluates the historical First-Fit insertion layout plus
   size-descending first-fit/best-fit candidates and keeps the smallest, so it
   can only ever shrink the arena and never changes a layout that was already
   minimal.  Measured on section 3's presets: `previous` 1,972 B (unchanged -- it
   is its live peak, so its layout does not move), the promoted `shipped` layout
   3,316 -> **1,992 B** (compliant, and now the certificate artifact), and
   `wide_second` 3,436 -> 2,160 B (still over: its live peak is 2,160 B).  The
   compliant search space therefore grows from `c1 <= 55` to `c1 <= 58` at
   `c0 = 27`; re-searching architectures inside the wider bound (`c0 = 28` and
   `c1 = 58` were never trained) stays a Phase 3 item.

3. `MATMUL` is part of the op set but unused by the shipped graph; Phase 2
   (fastText-shaped distillation) is the natural place to exercise it.

### 7. Repository hygiene (done before Phase 2)

Phase 1 was developed without version control; the tree is now a git repository
(initial commit on `main`) with:

- **`.gitignore`/`.gitattributes`** that keep the 107 MB Speech Commands archive
  and all build scratch out of history, while tracking the frozen certificate
  inputs: the 9.2 MB precomputed corpus and all three 0.1 MB checkpoints (verified
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

**Status**: M2.0-M2.4 **done** (feature contract, frozen corpus, int8 student,
showdown, C tokenizer — see the Phase 2 Close-Out at the end of this section);
M2.5 experiment open.
Phase 1 proved the *back half* of the framework on a
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

#### M2.0 feature contract (FROZEN 2026-09-26)

The normative implementation is `examples/hazard/tokenizer.py` -- the spec *is*
the code -- and `tests/test_hazard_tokenizer.py` pins it.  The C frontend
`examples/hazard/frontend/hazard_tokenizer.c` is bit-exact against it
(`tests/test_hazard_frontend.py`, M2.4):

| Item | Contract |
| :--- | :--- |
| Normal form | UTF-8 bytes; `A-Z` -> lowercase; control bytes and `0x7F` -> space; bytes `>= 0x80` -> `?`; truncated and space-padded to exactly `WINDOW_BYTES = 48` |
| Vocabulary | 256 ids, all `EMBEDDING` can address: id `0` unused, `1..96` printable ASCII (`byte - 0x1F`), `97..255` hashed bigrams |
| Bigrams | `97 + fnv1a32(byte, successor) % 159`, with a space sentinel after the last byte, so there are exactly `WINDOW_BYTES` pairs |
| Token vector | `TOKEN_COUNT = 96` uint8 ids: the 48 byte ids then the 48 bigram ids, i.e. one flat tensor and one `kiln_predict(tokens, 96)` call |
| Unknown tokens | none: the map is total over the 256 byte values, so there is no `[UNK]` path; non-ASCII folds to `?` one byte at a time |
| Round trip | `detokenize(tokenize(text)[:48]) == normalize_bytes(text)` and `tokenize(normalize_bytes(text)) == tokenize(text)` |
| Window budget | arena `= TOKEN_COUNT * (dim + labels)`: 96 tokens -> 1,344 B at `dim` 8 / 6 labels (asserted); `WINDOW_BYTES` may be raised to 96 only at `dim` 4 (1,920 B) |

`tests/test_hazard_tokenizer.py` also runs the loop end to end: text -> contract
ids -> *emitted* C == `kiln.ref` bit for bit, compiled with
`-std=c99 -O3 -Wall -Wextra -Werror -pedantic`, with every label inside the class
set.  That is M2.4 minus the C tokenizer itself.  Left open for M2.1/M2.2: the
label taxonomy (question 2 below), the teacher and the corpus; the head in
`examples/hazard/graph.py` assumes 6 labels.

#### IR amendment: `MAXPOOL1D(axis=...)` (2026-09-26)

Kiln's frozen op set could not express a token bag, which is what this task
needs: `EMBEDDING` (rank-1 uint8 tokens) and `MATMUL` always put the token axis
*first*, while `MAXPOOL1D` only reduced the *last* axis.  "Pool over the tokens"
therefore failed validation (`maxpool1d node 'op_pool': kernel 24 exceeds
padded input length 3`), and the closest expressible graph silently pooled the
*class* axis instead: it returned label 5 for a 3-class problem, because the IR
checks tensor shapes, not intent.

`MAXPOOL1D` therefore gained an integer `axis` attribute whose default is
`rank - 1`, so **every existing graph emits byte-identical C**: hashing the
shipped KWS translation unit and the toy test graph before and after the change
gave `1441f1de…` and `79d20551…` both times, and the certificate still reads
1,992 B / 39,614 B.  With `axis = 0` the window runs over the leading axis and
keeps the trailing axes as an inner block, i.e. `pooled[0, c] = max over tokens`
-- the bag a hazardous-API classifier needs.  `tests/test_diff.py` pins the new
shape (`EMBEDDING -> MATMUL -> MAXPOOL1D(axis=0) -> ARGMAX`, bit-exact against
`kiln.ref` through gcc-compiled C) and the failure modes it replaces
(out-of-range axis, and the wrong-axis kernel that used to pass validation).

Measured footprint of that shape (audited with `kiln.budget.report`, emitted C
bit-exact against the reference in every row):

| Tokens (bag) | `dim` | Labels | Arena | ROM | MACs | Fits 2 KB? |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 96 | 8 | 6 | 1,344 B | 2,096 B | 4,608 | yes |
| 128 | 4 | 6 | 1,280 B | 1,048 B | 3,072 | yes |
| 192 | 4 | 6 | 1,920 B | 1,048 B | 4,608 | yes |
| 128 | 8 | 12 | 2,560 B | 2,144 B | 12,288 | no |
| 256 | 4 | 12 | 4,096 B | 1,072 B | 12,288 | no |

The arena is exactly `tokens * (dim + labels)`, so the 2 KB cap binds the
*feature window*, not the weights: `dim` 4-8 and a handful of labels are
effectively free, while a 256-token window costs 4,096 B.  Open question 5 below
is therefore a question about the window, not about the model size.

### Design gates (each one is a Phase 1 lesson, applied in advance)

| Gate | Rule | Phase 1 precedent |
| :--- | :--- | :--- |
| **G1 Feature contract first** | Freeze the exact integer tokenizer/feature contract *before* any training run; the C runtime sees `uint8` tokens and int8 features, never floats or dB | §4 finding 3 — dB-domain training scored 50.54 % through the emitted C |
| **G2 Frozen corpus** | The synthesized dataset is a checked-in artifact with a manifest (teacher model + version, prompt hash, decoding params, seed, corpus hash). No live API calls inside tests or pipelines | the frozen split seed (1337) and the tracked 9.2 MB corpus |
| **G3 Real held-out set** | "Synthetic data works" is only meaningful against *real, never-synthesized* code, and against a naive baseline (regex/allow-list) and fastText | §3's `previous` twin (27/55): accuracy claims need a control |
| **G4 Equal-budget showdown** | fastText is compared at the *same* footprint (<= 2 KB arena / <= 40 KB ROM), not at equal accuracy | §3's arena cliff (closed by `kiln.ir.arena`) |
| **G5 Selection by validation** | Test accuracy is reported, never used to pick a checkpoint; report confidence intervals | §4 finding 4 — 734 clips is +/-2.2 points |

### Milestones

| # | Deliverable | Acceptance test |
| :---: | :--- | :--- |
| M2.0 | Feature-contract spec (tokenizer, vocabulary size, int8 packing, unknown-token policy) | **DONE**: `examples/hazard/tokenizer.py` + the contract table below + `tests/test_hazard_tokenizer.py` (54 checks, including text -> emitted-C parity); the `MAXPOOL1D(axis=0)` prerequisite is done as well |
| M2.1 | Meta-prompt generator + synthesis harness producing the frozen corpus | **DONE**: `examples/hazard/synthesize.py` + `manifest.json` (corpus 300 + real held-out 60, seed 1337); `tests/test_hazard_dataset.py` verifies the digests and the zero-leakage gate |
| M2.2 | int8 student (`EMBEDDING` + `MATMUL` + `MAXPOOL1D(axis=0)` + `ARGMAX`) trained by distillation, lowered to Kiln IR | **DONE**: `examples/hazard/train.py`; arena **1,344 B** / ROM **2,096 B** audited on trained weights (budget table above); bit-exact PyTorch int <-> `kiln.ref` <-> emitted C on all 360 samples (`tests/test_hazard_model.py`); `MATMUL` and `EMBEDDING` leave the "unexercised" list |
| M2.3 | Showdown report: Kiln student vs fastText vs rules baseline, with CIs | **DONE**: showdown table in the Phase 2 Close-Out section below, reproduced by `python -m examples.hazard.benchmark` |
| M2.4 | Raw source text -> fixed-point tokenizer -> emitted graph, all C | **DONE**: `examples/hazard/frontend/hazard_tokenizer.c/h`, checked by `tests/test_hazard_frontend.py` (`-std=c99 -O3 -Wall -Wextra -Werror -pedantic`) |
| M2.5 | Differential-fuzzing distillation loop (unlabeled code as a fuzzing oracle) | documented as an experiment, success or failure |

### Open questions (to settle before M2.1)

1. **Teacher**: which model, and is the generated corpus shippable (licence/quality)?  A
   frozen corpus makes the pipeline reproducible but hides prompt-iteration cost; live calls
   make it cheap to re-run but non-reproducible.  Proposal: freeze the corpus, keep the
   generator script + manifest in-tree.
2. **Label space**: one hazard class per snippet (single-label) or multi-label?  Single-label
   keeps `ARGMAX` as the graph exit and matches Phase 1's contract, so
   `examples/hazard/graph.py` assumes a 6-label single-label head; the taxonomy itself
   (benign, dynamic code execution, deserialization, shell injection, insecure tempfile,
   unsafe validation) is a proposal and only has to be frozen before M2.1's corpus exists.
3. **Real held-out source**: which public, licence-compatible code supplies the *real*
   evaluation set (and how to guarantee it never enters synthesis)?
4. **Distillation signal**: hard labels only, or teacher logits (soft targets)?  Only hard
   labels survive Kiln's integer contract without a new "soft-label" training path.
5. **Budget target**: is <= 2 KB the right student size for a vocabulary-sized `MATMUL`, or
   does this task need a different (ROM-heavy, RAM-light) envelope?  Measured: the arena is
   `tokens * (dim + labels)`, so the current cap allows a 96-token window at `dim` 8 / 6
   labels (1,344 B) or 192 tokens at `dim` 4 (1,920 B), while a 256-token, 12-label model
   needs 4,096 B -- keeping 2 KB decides the snippet window, not the model size.

## Phase 2 Close-Out (2026-09-26)

**Status**: M2.0-M2.4 DONE, M2.5 open (documented-as-an-experiment).  The
teacher record, the leakage gate and the corpus digests are in
`examples/hazard/manifest.json`.

### 1. M2.2 certificate: the int8 student

Reproduce with `python -m examples.hazard.train` (seed 1337: 180 smooth-max
annealed float epochs, then 40 QAT epochs through the exact integer pipeline;
`examples/hazard/hazard_int8_weights.json` is the shipped artifact) and check
with `python -m pytest tests/test_hazard_model.py`.  The label requantisation
scale is calibrated from the integer accumulators over train+val
(127 / 37,772 = 0.00336228) and lives in `examples/hazard/graph.py` as
`_LABEL_SCALE`, so a plain `build_bag_graph()` ships the calibrated graph.

| Metric | Target | Measured |
| :--- | :--- | :--- |
| Arena RAM (`.bss`, zero-malloc) | < 2,048 B | **1,344 B** |
| Constant ROM (int8 weights) | < 40,960 B | **2,096 B** (256 x 8 embedding + 6 x 8 head) |
| MACs per inference | — | **4,608** |
| int8 top-1, frozen test (n=45) | — | **73.3%** [59.0, 84.0] |
| int8 top-1, real held-out (n=60) | — | **73.3%** [61.0, 82.9] |
| float vs int agreement (test) | — | **100%**, 0.00% saturating logits |
| PyTorch int <-> `kiln.ref` <-> emitted C | bit-exact | **360/360 samples** (gcc) |

The float stage is a `BagClassifier` trained with a smooth-max (logsumexp)
surrogate whose tau anneals 1.0 -> 0.08, always evaluated with the hard
max-pool the graph deploys; QAT then fine-tunes through an int64 port of
`kiln.ir.sem.requantize` with straight-through gradients.  Checkpoints are
selected on validation accuracy only (G5).  `tests/test_hazard_model.py`
(10 tests) pins the artifact contract, the budget, the frozen split and a
differential check of the requantise port against `kiln.ir.sem.requantize`
over boundary accumulators.

### 2. M2.3 showdown (Wilson 95% CIs; one command: `python -m examples.hazard.benchmark`)

| Model | Footprint | Frozen test (n=45) | Real held-out (n=60) |
| :--- | :--- | :--- | :--- |
| Kiln int8 student (emitted C) | ROM 2,096 B; arena 1,344 B | 73.3% [59.0, 84.0] | **73.3%** [61.0, 82.9] |
| fastText (dim 8, no subwords) | 461 words x dim 8: 10.2 KB int8 | 77.8% [63.7, 87.5] | 68.3% [55.8, 78.7] |
| Rules baseline (regex control, G3) | 377 B of regex source | 86.7% [73.8, 93.7] | 76.7% [64.6, 85.6] |
| LogisticRegression (char 2-5, context) | 304 KB f32 -- over budget | 97.8% [88.4, 99.6] | 98.3% [91.1, 99.7] |

Every fitted model sees only the same 210-sample train split (the 45-sample
validation split exists solely for the student's checkpoint selection), and
fastText runs at the student's `dim` 8 with subword n-grams off, so G4's
comparison is between two linear bag-of-token models whose int8 footprints
both sit under the 40 KB envelope (10.2 KB vs 2.1 KB).  Reading: the CIs
overlap, fastText wins the frozen test set while the student wins the real
held-out set; the regex control's strength on the frozen test (86.7%) shows
how much of the *synthetic* test distribution its generation patterns
explain -- which is what the real held-out set (G3) is for; the 304 KB char
model prices the headroom an unconstrained footprint buys.  The student is
the only entry that is also bit-exact, arena-bounded and emitted as C99 from
the same graph the reference VM executes.


## Phase 3 Progress (2026-09-27)

**Status**: the declarative front half is in place.  `kiln.yaml` + `kiln build`
(M3.1) turn a task into `model.c` plus a budget certificate behind a hard budget
gate (joined by M4.2's accumulator-safety gate), and `kiln search` (M3.2)
enumerates a declared parameter grid and reports the RAM/ROM/capacity Pareto front
*without training anything*.
自動タスク振り分け (automatic task dispatch) is the remaining Phase 3
deliverable.

### 1. M3.1 declarative spec and `kiln build`

`kiln/spec/schema.py` is a plain-dataclass schema (`input`, `output`, `budget`,
`model`, `frontend`, `dataset`, `search`; the budget defaults to arena 2,048 B /
ROM 40,960 B) and `kiln/spec/loader.py` resolves `module:attr` and
`file.py:attr` entry points, passing `weights_path` / `architecture` when the
factory asks for them and exporting PyTorch `nn.Module` targets through
`kiln.frontend.pytorch`.  `kiln build <spec>` then emits `model.c`,
`budget_certificate.txt` and the declared C frontend files, prints one
`[BUDGET]` verdict line per limit and returns 1 if any limit is exceeded -- while
still writing the artifacts, so the failure can be diagnosed.  M4.2 added the one
exception: an int32 accumulator overflow aborts *before* emission, so no artifact
survives a model that cannot be accumulated safely.

Two checked-in specs pin the pipeline to the earlier certificates:

| Spec | Reproduces | Arena | ROM | MACs |
| :--- | :--- | ---: | ---: | ---: |
| `examples/hazard/kiln.yaml` | M2.2 (Phase 2 close-out §1) | 1,344 B | 2,096 B | 4,608 |
| `examples/kws/kiln.yaml` | the Phase 1 certificate | 1,992 B | 39,614 B | 314,342 |

`tests/test_spec.py` (26 tests) covers the schema, the resolver and both
reference specs; `tests/test_cli_build.py` (8) covers the budget gate, the
frontend copy, the soft input-shape check, the full hazard reference build and the
two M4.2 overflow-gate tests.

### 2. M3.2 compile-time architecture search (`kiln search`)

The premise is the one Phase 1 and Phase 2 both leaned on: **a Kiln footprint is
a compile-time quantity**.  `audit_graph` derives the arena, the constant ROM and
the MAC count from tensor shapes and weight extents, so a candidate's cost is
known the instant its graph is built -- no training run, no floating point, no
silicon.  A whole grid of architectures is therefore measurable in milliseconds,
and the same grid always produces the same report bytes.

`kiln search <spec>` reads the spec's `search:` block (`params` = factory kwargs
to try, `entry_point` = an optional shape-parameterized factory that replaces
`model.entry_point` for the search only, `max_candidates` = grid guard), builds
and audits every candidate with the same `audit_graph` the build uses, gates each
one on the spec's budget and reports the front over:

| Objective | Direction | Why |
| :--- | :--- | :--- |
| `arena_ram_bytes` | minimise | the `.bss` arena is the scarce resource on an MCU |
| `total_rom_bytes` | minimise | flash is the second constraint, and the RAM/ROM trade is real |
| `total_macs` | maximise | inside one frozen task the MAC count is the capacity knob; the audit's `estimated_worst_cycles` grows with it |

A candidate that cannot be built (a kernel longer than its input, a window its
frontend rejects) becomes an *error row* instead of an abort: the point of the
search is to map the boundary, and the boundary is exactly where candidates stop
being legal.  Setting `budget.macs_max` turns the MAC axis into a hard constraint,
which recovers the pure RAM/ROM front.

The shipped reference grid is `examples/hazard/kiln-search.yaml`: 15 candidates =
`tokens` {64, 96, 128, 192, 256} x `dim` {4, 6, 8}, with the shipped *trained*
factory kept as `model.entry_point` and the shape-parameterized
`examples.hazard.graph:build_bag_candidate` as `search.entry_point` (placeholder
weights are enough -- footprints follow shapes, not values).  Every row matches
the documented relations `arena = tokens * (dim + labels)`,
`rom = 256 * dim + labels * dim` and `macs = tokens * labels * dim`:

| dim | tokens | Arena | ROM | MACs | MACs/B | Verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 4 | 64 | 640 B | 1,048 B | 1,536 | 2.40 | fits (Pareto) |
| 4 | 96 | 960 B | 1,048 B | 2,304 | 2.40 | fits (Pareto) |
| 4 | 128 | 1,280 B | 1,048 B | 3,072 | 2.40 | fits (Pareto) |
| 4 | 192 | 1,920 B | 1,048 B | 4,608 | 2.40 | fits (Pareto) |
| 6 | 64 | 768 B | 1,572 B | 2,304 | 3.00 | fits (Pareto) |
| 6 | 96 | 1,152 B | 1,572 B | 3,456 | 3.00 | fits (Pareto) |
| 6 | 128 | 1,536 B | 1,572 B | 4,608 | 3.00 | fits (Pareto) |
| 8 | 64 | 896 B | 2,096 B | 3,072 | 3.43 | fits (Pareto) |
| 8 | 96 | 1,344 B | 2,096 B | 4,608 | 3.43 | fits (Pareto) |
| 8 | 128 | 1,792 B | 2,096 B | 6,144 | 3.43 | fits (Pareto) |
| 4/6/8 | 192, 256 | 2,304 - 3,584 B | 1,048 - 2,096 B | up to 12,288 | 2.40 - 3.43 | over arena |

Reading.  Ten candidates fit the 2 KB arena and **all ten are non-dominated**:
this grid is a tight RAM/capacity ladder, where widening the window or the
embedding always buys MACs for bytes, so no feasible shape is wasteful.  The
actionable form of that finding is the report's *capacity classes* table (the
RAM-cheapest shape at each reachable MAC count): at 4,608 MACs the widest
embedding (`dim` 8 / 96 tokens) costs 1,344 B against 1,920 B for `dim` 4 / 192
tokens -- 576 B of RAM saved for identical capacity, paid for with 1,048 B of
extra ROM.  That is the RAM/ROM trade the objective set is meant to expose, and
it is what the CLI/`search_report.md`/`search_results.csv` triple hands to a
designer, with the over-budget rows named rather than hidden.

`tests/test_search.py` (36 tests) pins the `search:` schema and its guards, the
grid order, the front arithmetic (`dominates` / `pareto_front`, ties, the
maximise direction), the loader's override path (including the refusal to
override an `nn.Module`), the error rows, the reference grid's exact footprints
and the determinism of the report bytes.

Left open: re-searching the KWS stage widths (`c0 = 28` and `c1 = 58` from
Phase 1 finding 2) is a *training* question, not a search-only one -- the current
lowering calibrates every activation on the corpus, and the never-trained widths
have no checkpoint -- so it stays a Phase 3 follow-up behind a shape-only
lowering path in `examples/kws/kws_graph.py`.

---

## Phase 4 Progress (2026-09-27)

**Status**: M4.1 (`kiln link`) is done -- the emitted C is compiled, linked,
measured and replayed, and the zero-malloc claim is now an *observation*.  M4.2
(interval boundary propagation) is done -- the certificate now carries a static
proof of int32 accumulator safety.  The linear-probing capability audit and
on-device firmware verification remain open.

### 1. M4.1 `kiln link`: from certificate to measured image

Phase 1-3 verified the emitted *translation unit*; this verifies a *program*.  One
command now performs six steps: resolve the spec and write `model.c` + certificate
+ the declared frontend (the same helper `kiln build` uses), generate a replay
harness, compile every translation unit with the strict flags
(`-std=c99 -O3 -Wall -Wextra -Werror -pedantic`), link a real executable, measure
the artifact (section sizes, undefined symbols of the model object) and optionally
run it.

Measured on this repository's own references (host gcc 16.2.0, MSYS2 mingw64; the
`.bss`/read-only columns include the frontend's own static state):

| Spec | Translation units | Audited arena / ROM | Linked `.bss` | Linked read-only | Allocation refs |
| :--- | :--- | ---: | ---: | ---: | :--- |
| `examples/hazard/kiln.yaml` | model.c + tokenizer + pipeline + harness | 1,344 B / 2,096 B | 4,152 B (arena + 2,808 B) | 6,888 B | none |
| `examples/kws/kiln.yaml` | model.c + audio frontend + pipeline + harness | 1,992 B / 39,614 B | 21,240 B (arena + 19,248 B) | 52,904 B | none |

Both linked binaries replay the harness vectors with labels **bit-identical to
`kiln.ref`** -- hazard (uint8 tokens) `[4, 3, 3, 2, 0]`, KWS (int8 log-mel)
`[0, 0, 0, 9, 1]`.  One harness covers both dtypes because the C entry point always
takes `const uint8_t *`: `to_input_domain` reinterprets the bytes
two's-complement for an int8 graph input, and the same five vectors therefore
exercise a token bag and a convolutional stack without a per-model fixture.

Why the extra `.bss` is *reported* instead of hidden: it is the frontend's own
zero-init scratch (tokenizer/pipeline buffers, the FFT workspace), so the number
that has to fit a 2 KB MCU is `.bss − arena` versus the frontend's declared static
state -- the model's own share is exactly the audited arena, which the report
states as `exact match` when nothing else claims `.bss`.

The zero-malloc guarantee is now checked where it matters: `nm -u` on the *model
object* must show no reference to `malloc`/`calloc`/`realloc`/`free`
(`kiln/link/toolchain.py`), and both references report
`no allocation entry point referenced`.  Measured on this host the model object's
entire undefined-symbol set is `{abort}` -- the entry point's own length guard --
which is a stronger statement than "no malloc": the translation unit's only
external dependency is an abort when the caller violates the input contract, so
there is nothing left to link against a runtime.  The harness itself is a pure
function of the input length -- zeros, an alternating 0x00/0xFF pattern, both
uniform extremes and a `7i + 1 mod 256` ramp -- which targets the three ways a
fixed-point graph goes wrong (saturation at the extremes, the max pool's
tie-breaking on a flat input, and an input spread over the whole quantisation
range) while keeping the linked image reproducible.

The strict flags are load-bearing rather than decorative: a frontend with an
unused variable fails the link under `-Werror`, so a warning that would otherwise
be discovered on hardware is discovered here, with the compiler's own message and
command line in the error.  A declared frontend file that is missing from the tree
is skipped (with `kiln build` warning about it), because the model can still be
linked and measured on its own.

`tests/test_link.py` adds 22 tests: the harness/prediction contract, the uint8 vs
int8 input domains, toolchain discovery (`--cc`, `$KILN_CC`, PATH) and its error
paths, the strict-flag gate, a missing frontend, a compile error surfacing the
compiler's diagnosis, the budget gate on `kiln link`, the M4.2 accumulator gate
(refused before the toolchain is discovered) and the two bit-exactness tests.  The
module skips itself when the host has no C compiler, and the zero-malloc section
reports `unavailable` instead of guessing when binutils are absent -- the same
"evidence or an explicit unknown" rule the rest of the repository follows.

### 2. M4.2 interval boundary propagation: from claim to proof

Phase 1-3 *asserted* that the int32 accumulator never overflows; M4.2 proves it.
`kiln/analysis/interval.py` runs Interval Boundary Propagation (IBP) over the
resolved graph: every tensor gets an inclusive `[lo, hi]` value interval, every
accumulating op (CONV1D, MATMUL, ADD) gets an accumulator interval, and the
audit reports whether both stay inside their contracts.

The propagation is exact where it matters and conservative where it must be:

* **Input bounds** come from the declared dtype (`UINT8` -> `[0, 255]`,
  `INT8` -> `[-128, 127]`) or explicit per-tensor overrides.
* **Weight bounds** use the sign-split trick: for each output filter/row,
  `acc_lo = x.lo * pos_sum + x.hi * neg_sum` and
  `acc_hi = x.hi * pos_sum + x.lo * neg_sum`, where `pos_sum`/`neg_sum` are the
  sums of the positive and non-positive weights -- the standard interval-arithmetic
  dot-product bound, computed per row and min/max-ed across rows.
* **Requantization** (`requantize(a, m, s)` = SRDHM + rounding divide) is strictly
  monotonic in `a`, which was verified empirically over 100K+ random triples in
  development, so evaluating it at the accumulator endpoints gives the *exact*
  output interval -- not an approximation.
* **Saturation** is detected *pre-clamp*: if the preclamp value falls below
  `INT8_MIN` or above `INT8_MAX`, the node is flagged `saturates=True` even
  though the emitted output is clamped.  This is reported per node, so a model
  whose worst case touches the clamp is visible rather than silently accepted.
* **LUT** nodes slice the 256-entry table at `idx = value - INT8_MIN` for the
  input range and take the min/max of the slice; **MAXPOOL1D** preserves its
  input interval; **ARGMAX** reports the index space `[0, num_elements - 1]`.

The results reach every surface a user already touches:

* the **certificate** gains one line --
  `Accumulator Safety : Verified int32 safe [-67640, +61845]` (or a WARNING when
  the bound fails) -- written to `budget_certificate.txt` by `kiln build`, and a
  hard gate for `kiln build` / `kiln compile` / `kiln link` (see below);
* `kiln audit --intervals` prints the full report (per-tensor bounds and the
  per-node accumulator/preclamp/saturation table);
* `kiln audit --json` ships an `interval_analysis` object
  (`int32_overflow_free`, `total_accum_ops`, `saturating_ops`, `min_accum`,
  `max_accum`, `tensor_intervals`, `node_audits`);
* `BudgetReport.interval_audit` carries the structured
  `IntervalAudit` for programmatic consumers.

An overflow is *reported* by the audit and *enforced* by the build.  Propagation
itself never raises: a synthetic 200K-weight matmul whose bound reaches
3,225,800,000 (> `INT32_MAX`) sets `int32_overflow_free=False` while the walk
continues (the requantize helpers clamp the accumulator into range first, so the
audit cannot itself crash on the very case it exists to catch).  `audit_graph`
wraps the whole analysis in a best-effort guard, so a custom graph with missing
quant parameters degrades to `interval_audit=None` instead of failing the budget
calculation.

`kiln build` and `kiln compile` turn that flag into a hard gate.  When
`BudgetReport.int32_overflow_free` is False, `emit_spec_artifacts` raises
`SafetyViolationError` *before the first byte is written*; the CLI prints the
certificate (whose `Accumulator Safety` line carries the WARNING), prints
`[SAFETY VIOLATION]` on stderr and exits 1.  A refused build therefore leaves no
`model.c` for someone to flash by accident -- the deliberate exception to the
"artifacts stay for diagnosis" rule that budget violations follow, because a
budget violation is a trade-off to revisit while a miscompiled accumulator is not.
`kiln link` inherits the gate (it emits through the same function), and
`kiln audit` stays read-only: reporting is its whole job.  `kiln search` is the one
surface that does not consult the bound yet -- it gates candidates on RAM/ROM/MACs
alone, so an architecture that would overflow still reports as feasible and is only
refused at build time (a recorded follow-up, not a Phase 4 deliverable).

Measured on the two in-tree models, both int32-safe:

| Spec | Accumulating ops | Accumulator range | int32 safe | Saturating (worst case) |
| :--- | ---: | :---: | :---: | ---: |
| `examples/hazard/kiln.yaml` | 1 | `[-67,640, +61,845]` | yes | 1 / 1 |
| `examples/kws/kiln.yaml` | 5 | `[-385,191, +220,602]` | yes | 5 / 5 |

The KWS `5 / 5` is the conservative worst case (full-range int8 inputs through
every conv), reported rather than hidden: the clamp exists precisely because the
worst case can reach it, and the certificate states where.  Both numbers come
from `python -m kiln audit <spec> --intervals`.

`tests/test_interval.py` adds 7 tests: interval algebra invariants, int8/int32
bounds, EMBEDDING+MATMUL propagation with exact requantized endpoints,
CONV1D+LUT+MAXPOOL1D+ADD chains, saturation detection, int32 overflow detection
on the synthetic matmul, and both real models pinned to their measured ranges.
`tests/test_budget.py` and `tests/test_cli.py` assert the certificate line, the
`--intervals` output and the `interval_analysis` JSON payload.  The gate itself is
pinned by four more tests: `tests/test_cli_build.py` checks that the 200K-weight
spec makes `kiln build` exit 1 while writing neither `model.c` nor
`budget_certificate.txt` (and that `emit_spec_artifacts` raises
`SafetyViolationError` without even creating the output directory);
`tests/test_link.py` checks that `kiln link` is refused for the same spec before a
compiler is even looked for; and `tests/test_cli.py` compiles a 133,200-tap
`nn.Conv1d` whose bound reaches 2,148,382,800 to check that `kiln compile` refuses
it with no C emitted.

Open in Phase 4: the linear-probing capability audit and on-device firmware
verification -- which needs a cross toolchain (`arm-none-eabi-gcc`, absent on
this host, so the firmware path stays unverified) plus real hardware.  `kiln
link` already walks one code path for both: the compiler is discovered through
`--cc`/`$KILN_CC`/PATH, so a cross build becomes a matter of a target profile
(linking script, `-mcpu`, memory map) rather than new plumbing.

