Online AuxIVA-IP and AuxIVA-ISS in C++ — single precision (fp32)
================================================================

A single-precision port of `../online_iva_cpp`. Every sample, coefficient and
accumulator is `float` instead of `double`. This folder is self-contained:
the FFTW single-precision header and DLL are in `third_party/`, the demo
audio is in `audio/`, and nothing outside this directory is needed to build
or run it.

Why fp32
--------

Aimed at an ARM+NEON target. NEON's 128-bit registers hold **4 floats** but
only **2 doubles**, and on most embedded ARM cores fp64 throughput is half of
fp32 or worse. Single precision is the natural width there — this is not a
fixed-point (Q15) port, which would be a much larger change.

Build and run
-------------

    build.bat                       (needs MinGW g++ on PATH)

    separate_online.exe audio\loop60\mixture.wav output\f32_iss iss

    separate_online.exe <input.wav> <output_prefix> <ip|iss> \
        [n_fft=2048] [hop=n_fft/2] [n_iter=1] [alpha=0.995] [model=laplace]

`build.bat` copies `libfftw3f-3.dll` next to the exe, since it is needed at
run time.

Does fp32 cost anything?
------------------------

Measured against the double build on the same 60 s input, over the last four
repetitions (i.e. after convergence):

| build  | algo | target corr. | interference |
|--------|------|--------------|--------------|
| double | IP   | 0.836        | 0.052        |
| double | ISS  | 0.835        | 0.052        |
| fp32   | IP   | 0.836        | 0.052        |
| fp32   | ISS  | 0.835        | 0.052        |

Waveform correlation between the two builds is **1.0000** for both
algorithms. Separation quality is unaffected.

Speed, same machine (32-bit MinGW, GCC 6.3, `-O3`, no explicit SIMD):

| algo | double      | fp32        |
|------|-------------|-------------|
| IP   | 1.25 ms/fr  | 0.72 ms/fr  |
| ISS  | 0.84 ms/fr  | 0.58 ms/fr  |

~1.4–1.7x faster before any vectorization work.

What differs from the double build
----------------------------------

Beyond the type change:

* **`eps` default is `1e-6f`, not `1e-10`.** fp32's relative epsilon is
  ~1.2e-7, so 1e-10 is below the precision at which the guarded quantities are
  computed, and `1/1e-10 = 1e10` is a wildly oversized weight. `phi()` blowing
  up was the root cause of two separate bugs in the double build, so the guard
  is set to something meaningful at this precision.
* **`r2` is accumulated in `double`.** It sums over `n_freq` bins (1025 at
  `n_fft = 2048`); doing that in fp32 loses enough precision to perturb the
  weight. The same applies to `dual_window()`'s sum of squares, which is a
  setup-time computation.
* **No per-frame heap allocation.** `StftSynthesis::push()` in the double
  build allocates a `std::vector` on every call; here it is a preallocated
  member (`new_hop_`). A real-time audio callback must not touch the heap.
* **`FFTW_ESTIMATE`, not `FFTW_MEASURE`.** `MEASURE` runs timing benchmarks at
  construction — slow and non-deterministic startup.
* **Self-contained compat shim.** `include/piva_online/compat.hpp` supplies
  `std::clamp` (MinGW GCC 6.3's libstdc++ predates it) and `M_PI`. The double
  build relies on an external force-included header instead.

Still to do for an actual embedded target
-----------------------------------------

This is a faithful fp32 port, not a finished embedded implementation. What it
does **not** yet do:

* **Structure-of-arrays layout.** Buffers are still `(n_freq, n_chan)`
  row-major, so frequency is the strided axis and the contiguous inner
  dimension is only 2–4 elements — too short to vectorize. The hot loops
  (`compute_xxH`, `update_covariance`, `demix`) are independent per frequency
  bin, so the layout should be flipped to put frequency contiguous and process
  4 bins per NEON vector.
* **Closed-form 2×2 inverse.** At `n_chan == 2`, `solve_inplace()`'s Gaussian
  elimination should be replaced by the closed form. It is branch-free (no
  pivoting), so it both vectorizes and gives a constant per-frame cost — the
  current data-dependent branches make worst-case execution time hard to bound.
* **Real streaming.** `separate_online.cpp` is still a file-based test bench.
  In particular the final peak normalization needs the entire signal before it
  can scale anything, so it has to become an AGC or a fixed gain.
* **Exceptions and virtual dispatch** are still used; an embedded build would
  likely want `-fno-exceptions` and compile-time algorithm selection so
  `update_demixing()` can inline.
* **FFTW** would be swapped for Ne10 / CMSIS-DSP / Arm Performance Libraries.
  Note the library name here is `libfftw3f` — the trailing `f` is the
  single-precision build.

Latency
-------

Unchanged from the double build, and worth knowing before targeting a TV:

    latency = (n_fft - hop) + 2*hop = n_fft + hop

At `n_fft = 2048`, `hop = 1024`, 16 kHz that is **192 ms**: 64 ms of STFT
window overlap plus **128 ms of pipeline registers**. The registers are the
larger half — they exist to supply the second frame to the covariance
estimate. If latency is the binding constraint, dropping them (and the
2-frame blend) returns you to 64 ms.

`alpha = 0.995` assumes **stationary sources**: ~9 s of covariance memory and
~25 s to converge. Use ~0.99 if the sources move. Note `alpha` is a per-frame
constant, so the wall-clock memory it represents scales with `hop`.

Licence
-------

Same GPL terms as the parent project. FFTW is GPL; see
`third_party/FFTW-COPYING.txt`.
