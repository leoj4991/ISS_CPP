Online AuxIVA-IP and AuxIVA-ISS in C++
======================================

Standalone streaming blind source separation with online independent vector
analysis. C++ ports of `auxiva_ip_online.py` and `auxiva_iss_online.py` from
the piva project; the demixing output is identical to the Python versions.

Folder contents
---------------

    include/piva_online/
        common.hpp             shared state, recursive covariance, projection back
        auxiva_ip_online.hpp   online AuxIVA-IP  (per-bin linear solve)
        auxiva_iss_online.hpp  online AuxIVA-ISS (inverse-free)
        stft_online.hpp        streaming STFT / iSTFT with FFTW
                               (periodic Hann analysis, dual synthesis window)
        wav_io.hpp             WAV reading / writing (no audio library needed)
    src/separate_online.cpp    separates a multichannel WAV file hop by hop
    audio/
        mixture.wav            2-mic simulated room mixture of 2 speakers
        reference_src1.wav     clean sources at mic 1, for comparison
        reference_src2.wav
    build.sh, Makefile

Dependencies
------------

Only a C++17 compiler and FFTW3.

| OS            | Install                                           |
|---------------|---------------------------------------------------|
| macOS         | `xcode-select --install` and `brew install fftw`  |
| Ubuntu/Debian | `sudo apt install g++ libfftw3-dev`               |
| Fedora        | `sudo dnf install gcc-c++ fftw-devel`             |

Build
-----

    ./build.sh          (or: make)

Run
---

Separate the bundled mixture with both algorithms (writes to `output/`):

    make run

Or on your own file:

    ./separate_online <input.wav> <output_prefix> <ip|iss> \
        [n_fft=2048] [hop=n_fft/2] [n_iter=1] [alpha=0.995] [model=laplace]

Example:

    mkdir -p output
    ./separate_online audio/mixture.wav output/online_iss iss 1024 512

Writes `<output_prefix>_src1.wav`, `_src2.wav`, ... (one per channel).

* The input must have one channel per source (determined case).
* WAV input can be 8/16/24/32-bit PCM or 32/64-bit float; output is 16-bit.
* `n_fft` / `hop`: smaller frames (e.g. 1024 / 512) sound less echoic on
  short or reverberant recordings.
* `alpha`: forgetting factor, per frame. Closer to 1 is more stable but adapts
  more slowly. Note it is a *per-frame* constant, so the wall-clock memory it
  represents scales with `hop`: the default 0.995 is ~9 s of memory at
  `hop = n_fft/2`, and converges in ~25 s. It assumes the sources do not move;
  for moving sources use ~0.99. Values above ~0.997 suppress interference
  further but may not converge within a short recording.
* Online algorithms need some time to converge, so the first seconds are
  less separated than the end.
* The input/output pipeline is double-buffered (one hop of latency on each
  side, on top of the STFT's own `n_fft - hop` window-overlap latency): the
  frame that gets demixed each cycle is the one that finished analysis on
  the *previous* cycle, and the hop written out each cycle is the one that
  finished synthesis on the *previous* cycle. The covariance estimate takes
  advantage of this by blending the two frames in flight each cycle (the
  one about to be demixed and the one that just arrived) into one
  instantaneous sample, rather than using a single frame.

Using the algorithms in your own code
-------------------------------------

```cpp
#include "piva_online/auxiva_iss_online.hpp"   // or auxiva_ip_online.hpp

piva_online::AuxIVAISSOnline bss(n_freq, n_chan, /*n_iter=*/3, /*alpha=*/0.96);

// for every STFT frame; x_cur, x_next, y, y_out are (n_freq, n_chan), row-major
// x_cur/x_next come from StftAnalysis::push (x_next is one hop ahead of x_cur)
bss.process_frame(x_cur, x_next, y);  // demixed frame, from x_cur
bss.project_back_frame(y, y_out, 0);  // scale fix to mic 1 (optional)
```

The algorithm headers (`common.hpp`, `auxiva_*_online.hpp`) need only the
C++ standard library; FFTW is used only by `stft_online.hpp`.
