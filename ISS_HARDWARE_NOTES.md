Implementing online AuxIVA-ISS on hardware — what to watch for
==============================================================

Practical notes for porting the online AuxIVA-ISS in this repo to embedded
C++ (ARM+NEON, a DSP core, or fixed point). Every item below came out of
debugging the three builds here, and the numbers quoted are measured on this
repo's 60 s two-speaker test signal, not estimates.

Reference builds:

| folder | arithmetic | notes |
|--------|-----------|-------|
| `online_iva_cpp` | `double` | reference |
| `online_iva_cpp_floating` | `float` (fp32) | bit-comparable to double, 1.4–1.7x faster |
| `online_iva_cpp_fixed_point` | `int32`/`int64` | partial; see its README |

Start here: the two tests that catch most bugs
----------------------------------------------

**1. Test anechoic first (RT60 = 0).** An anechoic 2-source / 2-mic mixture is
the *easiest* possible case for AuxIVA — it is exactly the instantaneous
mixing model the algorithm assumes. If that does not separate cleanly, you
have a code bug, not an acoustics problem. Running this is what finally
exposed a pipeline bug that had been misattributed to reverberation for
several rounds of investigation.

**2. Test with deliberate silence.** Continuous speech hides the `V`-decay
failure described below completely. Insert 30 s of true digital silence
between two speech segments and watch what happens to the covariance.

1. The denominator
------------------

`denom = w_k^H V_m w_k` in the per-bin update. This is the single most common
failure point in an ISS implementation.

**Know which branch is dangerous.** The two cases behave completely
differently as `V_m` shrinks:

```cpp
v[m] = (m == k) ? (1 - 1/sqrt(denom))   // NOT scale-invariant  <-- danger
                : numer / denom;         // scale-invariant, safe
```

For `m != k`, `numer` and `denom` are both linear in `V_m`, so the quotient is
invariant to `V_m`'s scale and rides out a collapse fine. For `m == k`,
`1/sqrt(denom)` scales as `1/sqrt(c)`. That branch is where the bug lives —
don't spend effort hardening the other one.

**Skip, never clamp.** A floor like

```cpp
denom = std::max(denom, eps);   // WRONG
```

looks protective and is the opposite. At `eps = 1e-10` it yields
`v_k = 1 - 1e5`, and the following `w_m -= v_m * w_k` scales the pivot row by
~1e5. The clamp does not rescue the bin; it converts an ill-posed
normalization into a catastrophic update that then lives in `W` permanently
and contaminates everything downstream. Use `continue` and leave `W` alone —
which is what AuxIVA-IP already does when its linear solve reports
singularity. ISS as commonly written has no equivalent escape.

**Make the degeneracy test scale-relative**, e.g. `denom > eps * trace(V_m)`.
An absolute threshold has to be retuned for every input level.

2. What makes the denominator collapse
--------------------------------------

**`V` decays geometrically during silence.** The recursion

```
V <- alpha*V + weight*xxH
```

degenerates to `V <- alpha*V` when `xxH ~ 0`, i.e. pure `alpha^t` decay with
**nothing flooring it**. Measured on 10 s speech / 30 s silence / 10 s speech:

| alpha | min denom observed |
|-------|--------------------|
| 0.96  | 5.03e-4 (20x below the `V` initialization) |
| 0.99  | 9.90e-3 |
| 0.995 | 9.95e-3 |

At alpha = 0.96 the decay is roughly 2x per second of silence, so ~50 s of
silence reaches `eps = 1e-10` and the floor starts firing. Lower alpha
collapses faster.

**The initialization masks this.** `V` starts at `0.01*I`, and the measured
minimum `denom` during healthy continuous speech is **0.00995** — the init
value. That looks like a floor but is only an initial condition. Once the
recursion decays past it, it is gone. A test signal without long silences
will never show you this.

**Fix: gate on activity.** If a frame carries no energy there is nothing to
learn from it — freeze `W` and `V` and just demix with the existing `W`. One
check removes both this failure and the `phi()` one below. Make the threshold
relative to a slowly decaying running peak, not absolute, so it does not need
retuning per input gain.

Measured effect (30 s silence, alpha = 0.96), with the gate:

| segment | target corr | interference |
|---------|-------------|--------------|
| before silence | 0.623 | 0.162 |
| after silence  | **0.773** | 0.156 |

It comes out of the gap *better* than it went in, because `W` is preserved
across the silence instead of being destroyed by a decaying covariance.

**Also consider trace-normalizing `V`.** Rewriting the recursion as

```
V_new = (A*V_prev + xxH) / (A + trace(xxH)),    A = alpha/weight
```

makes `trace(V_new) = 1` identically whenever `V_prev` has unit trace — the
normalization is free, it is just how the recursion is written. `V` then lives
in [0, 1] and `denom` becomes a bounded Rayleigh quotient. This is
**mandatory** for a fixed-point port and harmless in float. It is safe
because scaling `V_k` only rescales row *k* of `W`, and the projection-back
step sets the output level by minimum distortion anyway.

3. `phi()` is the same disease
------------------------------

```cpp
phi(r) = 1.0 / std::max(eps, 2.0 * r);
```

returns `1/eps` when `r -> 0`. Two separate bugs in this repo traced back to
this. Anything of the form "reciprocal of a quantity that legitimately goes to
zero when the input is silent" needs the same treatment: gate or skip, not
clamp.

**Keep `r2` and `xxH` describing the same data.** If you delay, blend or
pipeline frames, the auxiliary variable and the covariance sample must come
from the same frames. In this repo `r2` was computed from one frame while
`xxH` was averaged over two — so at every speech onset (`r2` still small,
`xxH` already large) `phi()` went large *while* `xxH` was large, injecting an
oversized sample. Fixing only that cut steady-state interference from 0.238 to
0.190 and raised target correlation from ~0.67 to ~0.75.

**Never feed an all-zero frame.** Prime pipeline registers with real data
rather than leaving them zero-initialized. A zero frame drives `r2` to exactly
0, `phi()` to `1/eps`, and poisons `V` for hundreds of frames afterwards
(it only decays at `alpha` per frame). This cost a full debugging cycle here:
separation looked broken even on perfectly anechoic input.

4. Numerical precision
----------------------

**`alpha` is a per-frame constant, so its wall-clock memory scales with
`hop`.** Halving the hop halves the time constant. After an `n_fft` change
this repo was silently running ~1 s of covariance memory where ~2 s was
intended. Measured effect of correcting it (60 s input, n_fft 2048/hop 1024):

| alpha | target | interference | converges in |
|-------|--------|--------------|--------------|
| 0.96  | 0.787  | 0.172        | ~5 s  |
| 0.99  | 0.813  | 0.136        | ~12 s |
| 0.995 | 0.844  | **0.069**    | ~25 s |
| 0.997 | 0.802  | 0.001        | ~50 s |
| 0.999 | 0.302  | —            | never (within 60 s) |

Higher alpha means cleaner steady state but slower warm-up, and past ~0.997 it
cannot converge inside a short recording. `alpha >= 0.995` assumes the sources
do not move.

**`alpha` also costs precision.** You need roughly `log2(1/(1-alpha))` bits
just to represent the per-frame increment — ~7.6 bits at alpha = 0.995, before
spending anything on the signal itself. This is why:

* fp32 (24-bit mantissa) is comfortable
* **fp16 (11-bit mantissa) is a poor fit for `V` and `W`** — the increment is
  only 2–3 bits above the quantization floor, and the accumulator quietly
  stops accumulating
* if you want fp16 for its 8 NEON lanes, use it for the STFT frames and
  `xxH` only, and keep `V` and `W` in fp32 (widening fp16 x fp16 -> fp32
  multiply-accumulate exists on recent ARM cores)

**Widen the accumulators.** `r2` sums over `n_freq` bins (1025 at
n_fft = 2048). This repo's fp32 build keeps `r2` in `double` because fp32
there visibly perturbs the weight. Same for the dual-window sum of squares.

**`W` is not bounded by 1.** The ISS normalization sets
`w_k^H V_k w_k = 1`, so `||w_k||^2 >= 1/lambda_max(V_k)` and grows without
bound as `V_k` becomes ill-conditioned. No fixed Q format works unless `V` is
normalized. Measured in the fixed-point build: Q26 (range +-32) saturated
22287 times; +-512 was needed.

**Round, do not truncate**, on every shift and division in an integer build.
Plain `>>` truncates toward negative infinity, which is a systematic bias
rather than noise, and it compounds across thousands of frames.

5. Structure — ISS's hardware advantages, and keeping them
-----------------------------------------------------------

ISS is the better of the two algorithms for hardware. Preserve why:

* **No matrix solve**, therefore no pivoting, therefore no data-dependent
  branches — the per-frame cost is constant, which is what lets you bound
  worst-case execution time. Do not reintroduce branches.
* At `n_chan == 2`, use the **closed-form 2x2 inverse** for projection back,
  not Gaussian elimination.
* **No heap allocation in the processing path.** This repo's float build
  allocated a `std::vector` per frame inside `StftSynthesis::push()` — fine on
  a desktop, unacceptable in an audio callback.
* **Flip the data layout for SIMD.** `(n_freq, n_chan)` row-major leaves only
  2–4 contiguous elements, too short to vectorize. The hot loops
  (`compute_xxH`, `update_covariance`, `demix`) are independent per frequency
  bin, so put frequency contiguous and process 4 bins per NEON vector.
* Consider `-fno-exceptions` and compile-time algorithm selection so
  `update_demixing()` can inline.

**Latency.** Total group delay is `(n_fft - hop) + 2*hop = n_fft + hop`. At
n_fft = 2048, hop = 1024, 16 kHz that is 192 ms: 64 ms of STFT window overlap
plus **128 ms of pipeline registers**. The registers are the larger half. If
latency binds — and 192 ms is a lip-sync failure on a TV — cut the registers
first, not `n_fft`.

6. Instrumentation
------------------

Every bug found in this repo showed up in a counter before it showed up in the
audio. Log at minimum:

* bins skipped as degenerate
* saturations (integer builds)
* frames frozen by the activity gate
* minimum `denom` observed

Two diagnostic tells worth knowing:

* **IP and ISS producing identical output means something is frozen.** Two
  different update rules cannot agree bit-for-bit unless both are seeing
  static state. This is how a block-exponent sign error was caught in the
  fixed-point build — `V` had frozen at its initial value and neither
  algorithm was really running.
* **A degenerate-skip count near 100% means a scaling error, not degeneracy.**
  A `2^-QW` factor dropped in a complex division made AuxIVA-IP skip 1,927,000
  bins — 99.9% of all updates — because every computed `w` was ~2^26 too small
  and tripped the guard.

7. On fixed point specifically
------------------------------

Short version: **it does not solve the denominator problem, it makes it
worse.** The problem is algorithmic conditioning, not precision.

Q15 resolves 1/32768 ~ 3.05e-5 and has no exponent. Against the measured
values above, a `denom` of 5.03e-4 leaves ~4 bits, and anything below 3.05e-5
is *exactly zero* — so `1/sqrt(denom)` becomes a division by zero rather than
by something small. Roughly 34 s of silence at alpha = 0.96 gets you there.
fp16, by contrast, has an exponent and degrades gracefully; its problem is
precision (see section 4), not range.

If integer arithmetic is genuinely required, trace-normalize `V` first
(section 2) — without it the port cannot work, because no fixed Q format can
hold `w_k` across the resulting range. `online_iva_cpp_fixed_point/` is a
working starting point with that change in place, and its README documents
both the scaling bugs found and the quality gap still open.

Otherwise: the fp32 build in `online_iva_cpp_floating/` measured **waveform
correlation 1.0000 against the double build** at 1.4–1.7x the speed, and
ARM+NEON gives fp32 natively at 4 lanes. On current evidence fixed point buys
nothing and costs accuracy.
