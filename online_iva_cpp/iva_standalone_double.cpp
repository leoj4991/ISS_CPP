// =============================================================================
// Online AuxIVA-IP / AuxIVA-ISS  -- single-file standalone, DOUBLE precision
// =============================================================================
//
// Streaming blind source separation of a 2+ channel WAV file. Everything is in
// this one file: WAV I/O, FFT, STFT, and both algorithms. There are NO external
// dependencies -- no FFTW, no Eigen, nothing to install.
//
//   Build:  g++ -std=c++17 -O3 iva_standalone_double.cpp -o iva_standalone_double
//           cl /std:c++17 /O2 /EHsc iva_standalone_double.cpp        (MSVC)
//
//   Run:    iva_standalone_double <input.wav> <output_prefix> <ip|iss>
//                          [n_fft=2048] [hop=n_fft/2] [n_iter=1]
//                          [alpha=0.995] [model=laplace]
//
//   e.g.    iva_standalone_double mixture.wav out iss
//           -> writes out_src1.wav, out_src2.wav
//
// Input must have one channel per source (determined case). WAV input may be
// 8/16/24/32-bit PCM or 32/64-bit float; output is 16-bit PCM.
//
// -----------------------------------------------------------------------------
// Notes that matter if you change parameters
// -----------------------------------------------------------------------------
//
// alpha (forgetting factor) is a PER-FRAME constant, so the wall-clock memory
// it represents scales with hop. The default 0.995 is ~9 s of covariance memory
// at hop = n_fft/2, and takes ~25 s to converge. It assumes the sources do not
// move; use ~0.99 for moving sources. Above ~0.997 it may never converge inside
// a short recording.
//
// Latency is (n_fft - hop) + 2*hop = n_fft + hop samples. At n_fft=2048,
// hop=1024, 16 kHz that is 192 ms: 64 ms of STFT window overlap plus 128 ms of
// pipeline registers. The registers feed the 2-frame covariance blend; if
// latency matters more than that blend, they are the thing to remove.
//
// This is a FILE-BASED test bench that simulates streaming. The final peak
// normalization needs the whole signal before it can scale any of it, so a
// real-time build must replace it with an AGC or a fixed gain.
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -----------------------------------------------------------------------------
// Do not build this with -ffast-math / -Ofast.
//
// KahanSum below depends on `c = (t - sum) - y` NOT being algebraically
// simplified. Under -ffast-math the compiler is licensed to prove that
// expression is zero and delete it, which silently degrades compensated
// summation to naive summation plus three wasted flops per term. Nothing would
// warn you; the numbers would just get quietly worse. -O2 / -O3 are fine.
// -----------------------------------------------------------------------------
#if defined(__FAST_MATH__)
#error "This file must not be compiled with -ffast-math / -Ofast: it would \
delete the compensation in KahanSum. Use -O2 or -O3."
#endif

// MinGW.org GCC 6.3's libstdc++ predates std::clamp even though it accepts
// -std=c++17. Harmless no-op on any compiler that already has it.
#if !defined(__cpp_lib_clamp)
namespace std {
template <class T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi) {
  return v < lo ? lo : (hi < v ? hi : v);
}
}  // namespace std
#endif

namespace piva_online {

// See reset(): opt-in, signal-scaled V initialization. Off by default so the
// default build stays bit-exact with the FFTW reference in this directory.
#ifdef PIVA_SCALED_V_INIT
constexpr bool kScaledVInit = true;
#else
constexpr bool kScaledVInit = false;
#endif

using real = double;
using cplx = std::complex<real>;

// PRECISION
//
// Everything is double. This is the REFERENCE build: it is bit-exact against
// the FFTW folder build in this directory (waveform correlation 1.000000,
// zero sample difference), so use it to validate optimized or
// reduced-precision ports.
//
// NOTE on comparing against ../online_iva_cpp_floating/iva_standalone.cpp:
// that file carries the same PIVA_SCALED_V_INIT switch with the same default
// (off), so the two match out of the box and any difference between them is
// precision alone. Define the macro on both, or neither.
//
// `setup_real` is the same type here; it exists so the two files stay
// line-by-line diffable.
using setup_real = double;

// Kahan compensated summation.
//
// Vestigial in this double build -- double has ample headroom over the 1025
// terms these sums run to. It is retained so this file stays line-by-line
// diffable against the fp32 standalone, where it does real work. Measured
// there, naive fp32 summation of r2 gives bit-identical output anyway, so it
// is insurance rather than a fix.
//
// Requires that -ffast-math is off; see the #error above.
struct KahanSum {
  real sum = 0.0;
  real c = 0.0;
  inline void add(real x) {
    const real y = x - c;
    const real t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  inline real get() const { return sum; }
};

// =============================================================================
// FFT -- iterative radix-2 Cooley-Tukey, replacing FFTW so this file stands
// alone. Twiddle tables are built at setup precision (see PRECISION POLICY).
// =============================================================================

class FFT {
 public:
  explicit FFT(size_t n) : n_(n), tw_(n / 2), rev_(n) {
    size_t levels = 0;
    while ((size_t(1) << levels) < n) ++levels;
    if ((size_t(1) << levels) != n || n < 2)
      throw std::invalid_argument("FFT size must be a power of two >= 2");

    for (size_t k = 0; k < n / 2; k++) {
      const setup_real ang =
          setup_real(-2.0 * M_PI) * setup_real(k) / setup_real(n);
      tw_[k] = cplx(real(std::cos(ang)), real(std::sin(ang)));
    }
    for (size_t i = 0; i < n; i++) {
      size_t r = 0;
      for (size_t b = 0; b < levels; b++) r |= ((i >> b) & size_t(1)) << (levels - 1 - b);
      rev_[i] = r;
    }
  }

  // in-place forward DFT, exp(-2*pi*i*k*n/N) convention (same as FFTW)
  void forward(cplx* buf) const {
    for (size_t i = 0; i < n_; i++)
      if (i < rev_[i]) std::swap(buf[i], buf[rev_[i]]);

    for (size_t size = 2; size <= n_; size <<= 1) {
      const size_t half = size / 2, step = n_ / size;
      for (size_t i = 0; i < n_; i += size)
        for (size_t j = i, k = 0; j < i + half; j++, k += step) {
          const cplx t = buf[j + half] * tw_[k];
          buf[j + half] = buf[j] - t;
          buf[j] = buf[j] + t;
        }
    }
  }

  // in-place inverse DFT, UNNORMALIZED (like FFTW: caller divides by n)
  void inverse(cplx* buf) const {
    for (size_t i = 0; i < n_; i++) buf[i] = std::conj(buf[i]);
    forward(buf);
    for (size_t i = 0; i < n_; i++) buf[i] = std::conj(buf[i]);
  }

  size_t size() const { return n_; }

 private:
  size_t n_;
  std::vector<cplx> tw_;
  std::vector<size_t> rev_;
};

// =============================================================================
// WAV I/O
// =============================================================================

struct WavData {
  int samplerate = 0;
  size_t n_chan = 0;
  size_t n_samples = 0;
  std::vector<real> samples;  // interleaved, in [-1, 1]
};

namespace detail {
inline uint32_t read_u32(const unsigned char* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
inline uint16_t read_u16(const unsigned char* p) { return uint16_t(p[0] | (p[1] << 8)); }
inline void write_u32(std::ofstream& o, uint32_t v) {
  for (int i = 0; i < 4; i++) o.put(char((v >> (8 * i)) & 0xff));
}
inline void write_u16(std::ofstream& o, uint16_t v) {
  o.put(char(v & 0xff));
  o.put(char((v >> 8) & 0xff));
}
}  // namespace detail

inline WavData read_wav(const std::string& filename) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) throw std::runtime_error("Could not open " + filename);
  std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
  if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
      std::memcmp(bytes.data() + 8, "WAVE", 4) != 0)
    throw std::runtime_error(filename + " is not a WAV file");

  uint16_t format = 0, n_chan = 0, bits = 0;
  uint32_t samplerate = 0;
  const unsigned char* data = nullptr;
  size_t data_size = 0, pos = 12;

  while (pos + 8 <= bytes.size()) {
    const unsigned char* chunk = bytes.data() + pos;
    uint32_t size = detail::read_u32(chunk + 4);
    const unsigned char* body = chunk + 8;
    size_t available = bytes.size() - (pos + 8);
    if (std::memcmp(chunk, "fmt ", 4) == 0 && size >= 16) {
      format = detail::read_u16(body);
      n_chan = detail::read_u16(body + 2);
      samplerate = detail::read_u32(body + 4);
      bits = detail::read_u16(body + 14);
      if (format == 0xFFFE && size >= 26) format = detail::read_u16(body + 24);  // extensible
    } else if (std::memcmp(chunk, "data", 4) == 0) {
      data = body;
      data_size = std::min<size_t>(size, available);
    }
    pos += 8 + size + (size & 1);  // chunks are padded to an even size
  }

  if (!data || n_chan == 0) throw std::runtime_error(filename + ": missing fmt or data chunk");
  const bool is_pcm = format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32);
  const bool is_float = format == 3 && (bits == 32 || bits == 64);
  if (!is_pcm && !is_float)
    throw std::runtime_error(filename + ": unsupported WAV format " + std::to_string(format) +
                             " with " + std::to_string(bits) + " bits");

  const size_t bps = bits / 8;
  WavData wav;
  wav.samplerate = int(samplerate);
  wav.n_chan = n_chan;
  wav.n_samples = data_size / (bps * n_chan);
  wav.samples.resize(wav.n_samples * n_chan);

  for (size_t i = 0; i < wav.samples.size(); i++) {
    const unsigned char* p = data + i * bps;
    setup_real v = 0;
    if (is_float && bits == 32) {
      float f;
      std::memcpy(&f, p, 4);
      v = f;
    } else if (is_float) {
      // A 64-bit float WAV stores IEEE-754 binary64 on disk, so the memcpy
      // target must be exactly 8 bytes. Some non-conforming embedded
      // toolchains (AVR-GCC, several TI DSP compilers) make `double` a 32-bit
      // alias for float; copying 8 bytes into that is a stack overwrite, not
      // a precision loss, so the static_assert catches it at compile time.
      static_assert(sizeof(double) == 8,
                    "64-bit float WAV input requires a 64-bit double; this "
                    "toolchain's double is narrower");
      double d64 = 0.0;
      std::memcpy(&d64, p, 8);
      v = setup_real(d64);
    } else if (bits == 8) {
      v = (setup_real(p[0]) - 128) / 128;
    } else if (bits == 16) {
      v = int16_t(detail::read_u16(p)) / 32768.0;
    } else if (bits == 24) {
      int32_t s = int32_t(p[0] | (p[1] << 8) | (p[2] << 16));
      if (s & 0x800000) s -= 0x1000000;
      v = s / 8388608.0;
    } else {
      v = int32_t(detail::read_u32(p)) / 2147483648.0;
    }
    wav.samples[i] = real(v);
  }
  return wav;
}

// samples: interleaved in [-1, 1]; written as 16-bit PCM
inline void write_wav(const std::string& filename, const std::vector<real>& samples,
                      size_t n_chan, int samplerate) {
  std::ofstream out(filename, std::ios::binary);
  if (!out) throw std::runtime_error("Could not write " + filename);

  const uint32_t data_size = uint32_t(samples.size() * 2);
  out.write("RIFF", 4);
  detail::write_u32(out, 36 + data_size);
  out.write("WAVE", 4);
  out.write("fmt ", 4);
  detail::write_u32(out, 16);
  detail::write_u16(out, 1);  // PCM
  detail::write_u16(out, uint16_t(n_chan));
  detail::write_u32(out, uint32_t(samplerate));
  detail::write_u32(out, uint32_t(samplerate * n_chan * 2));  // byte rate
  detail::write_u16(out, uint16_t(n_chan * 2));               // block align
  detail::write_u16(out, 16);
  out.write("data", 4);
  detail::write_u32(out, data_size);
  for (real v : samples) {
    // same scale as reading (x / 32768), so 16-bit files round-trip exactly
    long s = std::clamp(std::lround(setup_real(v) * 32768), -32768L, 32767L);
    detail::write_u16(out, uint16_t(int16_t(s)));
  }
}

// =============================================================================
// Shared algorithm core
//
// Memory layout (row-major):
//   x_cur, x_next, y : (n_freq, n_chan)     one STFT frame each
//   W    : (n_freq, n_chan, n_chan)         row k stores w_k^H, so y = W x_cur
//   V    : (n_src, n_freq, n_chan, n_chan)  per-source weighted covariance
// =============================================================================

enum class Model { Laplace, Gauss };

inline Model model_from_string(const std::string& name) {
  if (name == "laplace") return Model::Laplace;
  if (name == "gauss") return Model::Gauss;
  throw std::invalid_argument("No such model " + name);
}

// Auxiliary-variable weight. `eps` only bites when r is essentially zero, and
// then this returns 1/eps, which is enormous -- that is why the activity gate
// exists, rather than relying on this floor.
//
// `eps` here is an ABSOLUTE floor, matching the reference folder build. It is
// therefore not gain-invariant -- a sufficiently quiet recording could reach
// it. Measured on this project's material it never fires between 0 and -120
// dBFS, so it is left as-is rather than changed speculatively.
//
// If you do make it relative, scale only the FLOOR, never r itself:
// phi(r) = 1/(2r) is the Laplace auxiliary weight, and self-normalizing r
// changes the contrast function and breaks the AuxIVA majorization.
inline real phi(real r, Model model, size_t n_freq, real eps) {
  if (model == Model::Laplace) return 1.0 / std::max(eps, 2.0 * r);
  return 1.0 / std::max(eps, (r * r) / real(n_freq));
}

class OnlineAuxIVABase {
 public:
  // eps       - absolute floor inside phi(), on 2*r (r is summed over all bins)
  // silence_rel - a frame whose mean power is this far below the recent peak is
  //             treated as inactive and frozen. 1e-6 is -60 dB; 0 disables.
  // degen_rel - relative threshold for calling w_k^H V_m w_k degenerate,
  //             compared against trace(V_m), hence dimensionless
  OnlineAuxIVABase(size_t n_freq, size_t n_chan, size_t n_iter = 1,
                   real alpha = 0.995, Model model = Model::Laplace,
                   real eps = 1e-6, real silence_rel = 1e-6, real degen_rel = 1e-6)
      : n_freq_(n_freq),
        n_chan_(n_chan),
        n_src_(n_chan),
        n_iter_(n_iter),
        alpha_(alpha),
        model_(model),
        eps_(eps),
        silence_rel_(silence_rel),
        degen_rel_(degen_rel),
        v_ref_(0.0),
        v_scale_frames_seen_(0),
        peak_power_(0.0),
        frames_gated_(0),
        bins_skipped_(0),
        pb_degenerate_(0),
        W_(n_freq * n_chan * n_chan),
        V_(n_chan * n_freq * n_chan * n_chan),
        V_prev_(V_.size()),
        xxH_(n_freq * n_chan * n_chan),
        xxH_next_(xxH_.size()) {
    reset();
  }

  virtual ~OnlineAuxIVABase() = default;

  // W = identity per bin. V = 0.01 * identity per source/bin by default,
  // matching the reference folder build.
  //
  // -DPIVA_SCALED_V_INIT instead scales V's initialization to the signal (see
  // scale_v_to_signal). That is NOT enabled by default because it is a trade,
  // not a clear win: measured on this project's 60 s material it takes steady
  // interference 0.052 -> 0.016, but target fidelity 0.835 -> 0.731 and
  // roughly doubles convergence time. Enable it to compare like-for-like
  // against the fp32 standalone, which has it on.
  void reset() {
    const size_t n = n_chan_;
    std::fill(W_.begin(), W_.end(), cplx(0.0));
    std::fill(V_.begin(), V_.end(), cplx(0.0));
    for (size_t f = 0; f < n_freq_; f++)
      for (size_t c = 0; c < n; c++) W_[w_idx(f, c, c)] = 1.0;
    v_ref_ = 0.0;
    v_scale_frames_seen_ = 0;
    if (!kScaledVInit)
      for (size_t k = 0; k < n_src_; k++)
        for (size_t f = 0; f < n_freq_; f++)
          for (size_t c = 0; c < n; c++) V_[v_idx(k, f, c, c)] = 0.01;
  }

  // Process one STFT frame. Only x_cur is demixed into y; x_next is one hop
  // ahead and only enriches the instantaneous covariance sample.
  //
  // ACTIVITY GATE: a frame with essentially no energy is skipped entirely (W
  // and V untouched) and just demixed with the existing W. Adapting on silence
  // is actively harmful: V decays as alpha^t with nothing flooring it, and
  // r2 -> 0 makes phi() return 1/eps. Both are reciprocals of something that
  // legitimately goes to zero when the input is silent.
  void process_frame(const cplx* x_cur, const cplx* x_next, cplx* y) {
    if (!frame_is_active(x_cur)) {
      ++frames_gated_;
      demix(x_cur, y);
      return;
    }

    compute_xxH(x_cur, xxH_);
    compute_xxH(x_next, xxH_next_);
    for (size_t i = 0; i < xxH_.size(); i++) xxH_[i] = 0.5 * (xxH_[i] + xxH_next_[i]);

    // snapshot of V_{t-1}; the recursion is anchored to the previous frame
    if (kScaledVInit) scale_v_to_signal();

    std::copy(V_.begin(), V_.end(), V_prev_.begin());

    for (size_t it = 0; it < n_iter_; it++)
      for (size_t k = 0; k < n_src_; k++) {
        update_covariance(x_cur, x_next, k);
        update_demixing(k);
      }

    demix(x_cur, y);
  }

  // det(a*d - b*c) for complex values, with FMA-compensated products.
  //
  // This is the one place where cofactor inversion is weaker than pivoted
  // elimination: ad and bc are formed at full magnitude and subtracted, so a
  // near-singular W loses bits to cancellation. std::fma recovers them --
  // fma(x, y, -x*y) is exactly the rounding error of the product x*y, so
  // adding it back gives a result close to the exactly-rounded difference.
  // (Kahan's 2x2 determinant algorithm, applied per real component.)
  static inline real det2_fma(real x, real y, real u, real v) {
    const real w = u * v;                  // rounded product
    const real e = std::fma(u, v, -w);     // its exact rounding error
    return std::fma(x, y, -w) - e;
  }
  static inline cplx det2c(cplx a, cplx d, cplx b, cplx c) {
    // (a*d - b*c), expanded so each real component uses the compensated form
    const real re = det2_fma(a.real(), d.real(), b.real(), c.real()) -
                    det2_fma(a.imag(), d.imag(), b.imag(), c.imag());
    const real im = det2_fma(a.real(), d.imag(), b.real(), c.imag()) +
                    det2_fma(a.imag(), d.real(), b.imag(), c.real());
    return cplx(re, im);
  }

  // Minimum-distortion scaling: y_k <- A[ref,k] y_k with A = W^{-1}. Uses the
  // current W only, so it stays causal. At n_chan == 2 this is the closed-form
  // (cofactor) inverse: branch-free, and the determinant is FMA-compensated.
  void project_back_frame(const cplx* y, cplx* y_scaled, size_t ref = 0) {
    const size_t n = n_chan_;
    if (n == 2) {
      for (size_t f = 0; f < n_freq_; f++) {
        const cplx a = W_[w_idx(f, 0, 0)], b = W_[w_idx(f, 0, 1)];
        const cplx c = W_[w_idx(f, 1, 0)], d = W_[w_idx(f, 1, 1)];
        const cplx det = det2c(a, d, b, c);

        // RELATIVE degeneracy test -- an exact-zero test would be nearly
        // useless here, and worse, the FMA compensation above makes it more
        // so: compensation exists precisely to stop a near-singular
        // determinant from flushing to zero, so it lands on a tiny non-zero
        // value instead. `det == 0` would then pass, and r0/det would produce
        // an enormous scale factor on that bin -- a single such spike is
        // enough to wreck a whole file once the output is peak-normalized.
        //
        // Compare |det|^2 against the size of the products that formed it.
        // Squared magnitudes throughout, so there is no sqrt in the per-bin
        // loop (std::norm is |z|^2, not |z|).
        const real scale2 = std::norm(a) * std::norm(d) + std::norm(b) * std::norm(c);
        if (!(std::norm(det) > degen_rel_ * degen_rel_ * scale2)) {
          ++pb_degenerate_;
          y_scaled[f * n + 0] = y[f * n + 0];
          y_scaled[f * n + 1] = y[f * n + 1];
          continue;
        }
        // inv(W) = [d -b; -c a]/det, so row `ref` is (d, -b) or (-c, a)
        const cplx r0 = (ref == 0) ? d : -c;
        const cplx r1 = (ref == 0) ? -b : a;
        y_scaled[f * n + 0] = (r0 / det) * y[f * n + 0];
        y_scaled[f * n + 1] = (r1 / det) * y[f * n + 1];
      }
      return;
    }
    // general case: row `ref` of W^{-1} solves W^T z = e_ref
    std::vector<cplx> A(n * n), e(n);
    for (size_t f = 0; f < n_freq_; f++) {
      for (size_t r = 0; r < n; r++)
        for (size_t c = 0; c < n; c++) A[r * n + c] = W_[w_idx(f, c, r)];
      for (size_t c = 0; c < n; c++) e[c] = (c == ref) ? 1.0 : 0.0;
      const bool ok = solve_inplace(A.data(), e.data(), n);
      for (size_t k = 0; k < n; k++)
        y_scaled[f * n + k] = ok ? e[k] * y[f * n + k] : y[f * n + k];
    }
  }

  long frames_gated() const { return frames_gated_; }
  long bins_skipped() const { return bins_skipped_; }
  // bins where W was too near-singular to invert for projection back
  long pb_degenerate() const { return pb_degenerate_; }

 protected:
  // Solve A x = b in place, partial pivoting. Only used for n_chan != 2.
  static bool solve_inplace(cplx* A, cplx* b, size_t n) {
    for (size_t col = 0; col < n; col++) {
      size_t piv = col;
      real best = std::abs(A[col * n + col]);
      for (size_t r = col + 1; r < n; r++) {
        const real v = std::abs(A[r * n + col]);
        if (v > best) { best = v; piv = r; }
      }
      if (best == 0.0) return false;
      if (piv != col) {
        for (size_t c = 0; c < n; c++) std::swap(A[col * n + c], A[piv * n + c]);
        std::swap(b[col], b[piv]);
      }
      for (size_t r = col + 1; r < n; r++) {
        const cplx fac = A[r * n + col] / A[col * n + col];
        if (fac == cplx(0.0)) continue;
        for (size_t c = col; c < n; c++) A[r * n + c] -= fac * A[col * n + c];
        b[r] -= fac * b[col];
      }
    }
    for (size_t i = n; i-- > 0;) {
      cplx s = b[i];
      for (size_t c = i + 1; c < n; c++) s -= A[i * n + c] * b[c];
      b[i] = s / A[i * n + i];
    }
    return true;
  }

  size_t w_idx(size_t f, size_t r, size_t c) const { return (f * n_chan_ + r) * n_chan_ + c; }
  size_t v_idx(size_t k, size_t f, size_t r, size_t c) const {
    return ((k * n_freq_ + f) * n_chan_ + r) * n_chan_ + c;
  }

  // Relative to a slowly decaying peak, so it is invariant to input gain.
  // The sum is Kahan-compensated because it runs over n_freq*n_chan terms.
  bool frame_is_active(const cplx* x_cur) {
    KahanSum acc;
    for (size_t i = 0; i < n_freq_ * n_chan_; i++) acc.add(std::norm(x_cur[i]));
    const real power = acc.get() / real(n_freq_ * n_chan_);
    peak_power_ = std::max(peak_power_ * 0.999, power);  // forgets far slower than alpha
    if (silence_rel_ <= 0.0) return true;
    return power > peak_power_ * silence_rel_;
  }

  // Scale V''s initialization to the signal (enabled by -DPIVA_SCALED_V_INIT).
  //
  // A fixed constant is not gain-invariant. If it is far larger than the
  // per-frame contribution weight*xxH, V stays ~ c*alpha^t*I for EVERY source
  // -- isotropic and identical across m -- and ISS''s m != k update degenerates
  // to (w_m^H w_k)/||w_k||^2 because the scalar cancels exactly, carrying zero
  // information about the input. Scaling the init keeps it commensurate so the
  // isotropy breaks on the first update.
  //
  // WHAT THIS DOES NOT FIX -- measured, do not assume otherwise. c0 below is
  // proportional to the input gain (verified: exactly 1e-6x going from 0 to
  // -120 dBFS), but the recursion's equilibrium is proportional to gain
  // SQUARED. ISS pins w_k^H V_k w_k = 1, an absolute target, so W ~ 1/sqrt(V);
  // then r = ||Wx|| and weight = (1-alpha)/(2r), which makes the data term
  // weight*xxH scale as gain^(a/2+1) when V ~ gain^a. The fixed point is
  // a = 2, not a = 1.
  //
  // So at -120 dBFS the init is ~1e6 too large, and at alpha = 0.995 it decays
  // only 0.009x across a 941-frame file -- it never gets out of the way. V
  // stays isotropic (anisotropy pinned at 0.0045, against 0.13 -> 0.67 at
  // 0 dBFS), W stays diagonal, and the output is the unseparated input.
  // Forcing c0 ~ gain^2 does restore anisotropy (1.86) and mixing (|W01/W00|
  // 0.54) and moves interference 0.65 -> 0.60, but still does not converge:
  // W then overshoots to ~4e6, r grows, weight collapses, and V decays as pure
  // alpha^t with a negligible data term. The real defect is the W <-> phi(r)
  // feedback, which a scaled init only relocates. Fixing it properly means
  // making phi's argument scale-free (normalize r by the frame's own power, or
  // drop the init entirely and bias-correct the EMA as V/(1-alpha^t)) -- that
  // changes convergence behaviour and has not been done here.
  //
  // Rescaling V is always safe: it only rescales rows of W, and
  // project_back_frame() sets the output level regardless. That is why this
  // may keep revising upward over the first few frames, in case the first
  // active frame is room tone rather than speech.
  void scale_v_to_signal() {
    if (v_scale_frames_seen_ >= kVScaleFrames) return;
    ++v_scale_frames_seen_;
    const size_t n = n_chan_;
    KahanSum p;
    for (size_t f = 0; f < n_freq_; f++)
      for (size_t c = 0; c < n; c++) p.add(xxH_[(f * n + c) * n + c].real());
    const real mean_power = p.get() / real(n_freq_ * n);
    if (!(mean_power > 0.0)) return;
    const real r_ref = std::sqrt(mean_power * real(n_freq_ * n));
    const real c0 = mean_power / (2.0 * std::max(r_ref, eps_));
    if (!(c0 > v_ref_)) return;
    if (v_ref_ > 0.0) {
      const real g = c0 / v_ref_;
      for (cplx& v : V_) v *= g;
    } else {
      std::fill(V_.begin(), V_.end(), cplx(0.0));
      for (size_t k = 0; k < n_src_; k++)
        for (size_t f = 0; f < n_freq_; f++)
          for (size_t c = 0; c < n; c++) V_[v_idx(k, f, c, c)] = c0;
    }
    v_ref_ = c0;
  }

  void compute_xxH(const cplx* x, std::vector<cplx>& out) const {
    const size_t n = n_chan_;
    for (size_t f = 0; f < n_freq_; f++)
      for (size_t c = 0; c < n; c++)
        for (size_t d = 0; d < n; d++)
          out[(f * n + c) * n + d] = x[f * n + c] * std::conj(x[f * n + d]);
  }

  // r2 is averaged over BOTH frames, matching xxH_ being the average of
  // x_cur x_cur^H and x_next x_next^H. Taking r2 from x_cur alone would
  // desynchronize them: at a speech onset (x_cur quiet, x_next loud) phi()
  // would go large while xxH_ was large too, injecting an oversized sample.
  // r2 is Kahan-accumulated. In THIS build that is belt-and-braces -- double
  // has ample headroom over 1025 terms -- and it is kept only so the file
  // stays line-by-line diffable against the fp32 version, where the same
  // summation was measured to give bit-identical output anyway.
  void update_covariance(const cplx* x_cur, const cplx* x_next, size_t k) {
    const size_t n = n_chan_;
    KahanSum r2;
    for (size_t f = 0; f < n_freq_; f++) {
      cplx y_cur = 0.0, y_next = 0.0;
      for (size_t c = 0; c < n; c++) {
        y_cur += W_[w_idx(f, k, c)] * x_cur[f * n + c];
        y_next += W_[w_idx(f, k, c)] * x_next[f * n + c];
      }
      r2.add(0.5 * (std::norm(y_cur) + std::norm(y_next)));
    }
    const real weight = (1.0 - alpha_) * phi(std::sqrt(r2.get()), model_, n_freq_, eps_);

    const size_t block = n * n, off = k * n_freq_ * block;
    for (size_t i = 0; i < n_freq_ * block; i++)
      V_[off + i] = alpha_ * V_prev_[off + i] + weight * xxH_[i];
  }

  void demix(const cplx* x_cur, cplx* y) const {
    const size_t n = n_chan_;
    for (size_t f = 0; f < n_freq_; f++)
      for (size_t r = 0; r < n; r++) {
        cplx s = 0.0;
        for (size_t c = 0; c < n; c++) s += W_[w_idx(f, r, c)] * x_cur[f * n + c];
        y[f * n + r] = s;
      }
  }

  virtual void update_demixing(size_t k) = 0;

  size_t n_freq_, n_chan_, n_src_, n_iter_;
  real alpha_;
  Model model_;
  real eps_, silence_rel_, degen_rel_;
  real v_ref_;
  int v_scale_frames_seen_;
  static constexpr int kVScaleFrames = 64;
  real peak_power_;
  long frames_gated_, bins_skipped_, pb_degenerate_;
  std::vector<cplx> W_, V_, V_prev_, xxH_, xxH_next_;
};

// =============================================================================
// AuxIVA-ISS -- iterative source steering (inverse-free)
//
//   v_m = (w_m^H V_m w_k) / (w_k^H V_m w_k)      for m != k
//   v_k = 1 - 1 / sqrt(w_k^H V_k w_k)
//   w_m^H <- w_m^H - v_m w_k^H
//
// No matrix solve, so no pivoting and a constant cost per bin. That makes this
// the better candidate for an embedded / SIMD port.
// =============================================================================

class AuxIVAISSOnline : public OnlineAuxIVABase {
 public:
  AuxIVAISSOnline(size_t n_freq, size_t n_chan, size_t n_iter = 1,
                  real alpha = 0.995, Model model = Model::Laplace,
                  real eps = 1e-6, real silence_rel = 1e-6,
                  real degen_rel = 1e-6)
      : OnlineAuxIVABase(n_freq, n_chan, n_iter, alpha, model, eps, silence_rel, degen_rel),
        wk_(n_chan),
        v_(n_chan) {}

 protected:
  void update_demixing(size_t k) override {
    const size_t n = n_chan_;

    for (size_t f = 0; f < n_freq_; f++) {
      // frozen pivot row (stores w_k^H) -- all v_m must be computed from the
      // W as it was BEFORE this update, so snapshot it first
      for (size_t c = 0; c < n; c++) wk_[c] = W_[w_idx(f, k, c)];

      // denom = w_k^H V_m w_k is a quadratic form in a PSD matrix, so it is
      // >= 0 and goes to 0 as V_m does. The branches differ sharply there:
      //   m != k : numer and denom are BOTH linear in V_m, so the quotient is
      //            scale-invariant and stays well behaved.
      //   m == k : 1/sqrt(denom) is NOT scale-invariant. Clamping denom does
      //            not rescue a degenerate bin, it turns it into a huge update
      //            that then lives in W permanently. So: SKIP the bin instead.
      bool degenerate = false;
      for (size_t m = 0; m < n_src_; m++) {
        const cplx* Vm = &V_[v_idx(m, f, 0, 0)];
        real denom = 0.0, trace = 0.0;
        cplx numer = 0.0;
        for (size_t r = 0; r < n; r++) {
          cplx Vw = 0.0;  // (V_m w_k)_r
          for (size_t c = 0; c < n; c++) Vw += Vm[r * n + c] * std::conj(wk_[c]);
          denom += std::real(wk_[r] * Vw);
          numer += W_[w_idx(f, m, r)] * Vw;
          trace += std::real(Vm[r * n + r]);
        }
        // scale-relative test: compared against V_m's own size, so it holds at
        // any input level (both sides are linear in V_m)
        if (!(trace > 0.0) || !(denom > degen_rel_ * trace)) {
          degenerate = true;
          break;
        }
        v_[m] = (m == k) ? cplx(1.0 - 1.0 / std::sqrt(denom)) : numer / denom;
      }
      if (degenerate) {
        ++bins_skipped_;
        continue;
      }

      for (size_t m = 0; m < n_src_; m++)
        for (size_t c = 0; c < n; c++) W_[w_idx(f, m, c)] -= v_[m] * wk_[c];
    }
  }

 private:
  std::vector<cplx> wk_, v_;
};

// =============================================================================
// AuxIVA-IP -- iterative projection
//   solve (W V_k) w_k = e_k, then w_k <- w_k / sqrt(w_k^H V_k w_k)
// =============================================================================

class AuxIVAIPOnline : public OnlineAuxIVABase {
 public:
  AuxIVAIPOnline(size_t n_freq, size_t n_chan, size_t n_iter = 1,
                 real alpha = 0.995, Model model = Model::Laplace,
                 real eps = 1e-6, real silence_rel = 1e-6,
                 real degen_rel = 1e-6)
      : OnlineAuxIVABase(n_freq, n_chan, n_iter, alpha, model, eps, silence_rel, degen_rel),
        WV_(n_chan * n_chan),
        w_(n_chan) {}

 protected:
  void update_demixing(size_t k) override {
    const size_t n = n_chan_;

    for (size_t f = 0; f < n_freq_; f++) {
      const cplx* Vk = &V_[v_idx(k, f, 0, 0)];

      // WV = W V_k, row m is w_m^H V_k
      for (size_t r = 0; r < n; r++)
        for (size_t c = 0; c < n; c++) {
          cplx s = 0.0;
          for (size_t d = 0; d < n; d++) s += W_[w_idx(f, r, d)] * Vk[d * n + c];
          WV_[r * n + c] = s;
        }

      // w = (W V_k)^{-1} e_k, the true (unconjugated) vector
      for (size_t c = 0; c < n; c++) w_[c] = (c == k) ? 1.0 : 0.0;
      if (!solve_inplace(WV_.data(), w_.data(), n)) {
        ++bins_skipped_;
        continue;
      }

      // normalize by sqrt(w^H V_k w); skip rather than clamp, for the same
      // reason ISS does -- a clamped denominator becomes an oversized scale
      // factor written permanently into W
      real denom = 0.0, trace = 0.0;
      for (size_t r = 0; r < n; r++) {
        cplx Vw = 0.0;
        for (size_t c = 0; c < n; c++) Vw += Vk[r * n + c] * w_[c];
        denom += std::real(std::conj(w_[r]) * Vw);
        trace += std::real(Vk[r * n + r]);
      }
      if (!(trace > 0.0) || !(denom > degen_rel_ * trace)) {
        ++bins_skipped_;
        continue;
      }
      const real scale = 1.0 / std::sqrt(denom);

      for (size_t c = 0; c < n; c++) W_[w_idx(f, k, c)] = std::conj(w_[c] * scale);
    }
  }

 private:
  std::vector<cplx> WV_, w_;
};

// =============================================================================
// Streaming STFT / iSTFT
//
// StftAnalysis returns TWO frames per call: x_cur (the frame computed on the
// previous call) and x_next (computed this call). That 1-hop register is what
// feeds the 2-frame covariance blend. StftSynthesis is symmetric on the output
// side. Each costs one hop of latency on top of the window overlap.
// =============================================================================

inline std::vector<real> hann_window(size_t length) {
  std::vector<real> w(length);
  for (size_t n = 0; n < length; n++)
    w[n] = real(0.5 * (1.0 - std::cos(setup_real(2.0 * M_PI) * setup_real(n) /
                                                  setup_real(length))));
  return w;
}

inline std::vector<real> dual_window(const std::vector<real>& awin, size_t hop) {
  const size_t len = awin.size();
  const long length = static_cast<long>(len);
  const long lhop = static_cast<long>(hop);

  // setup precision: a sum of squares over overlapping windows is exactly the
  // kind of accumulation that loses precision, and this runs once
  std::vector<setup_real> norm(len, setup_real(0));
  for (long shift = -((length - 1) / lhop) * lhop; shift < length; shift += lhop)
    for (long m = std::max(0L, shift); m < std::min(length, length + shift); m++) {
      const setup_real a = setup_real(awin[static_cast<size_t>(m - shift)]);
      norm[static_cast<size_t>(m)] += a * a;
    }

  std::vector<real> swin(len);
  for (size_t m = 0; m < len; m++) {
    // norm == 0 means no analysis window covers this sample, so the dual window
    // is undefined and perfect reconstruction is impossible. At hop == n_fft
    // only shift 0 contributes and a periodic Hann has awin[0] == 0, which
    // would silently produce NaN. Fail loudly instead.
    if (!(norm[m] > setup_real(0)))
      throw std::invalid_argument(
          "hop breaks the COLA condition for this window (no overlap covers "
          "every sample); use hop < n_fft, e.g. n_fft/2");
    swin[m] = real(setup_real(awin[m]) / norm[m]);
  }
  return swin;
}

class StftAnalysis {
 public:
  StftAnalysis(size_t n_fft, size_t hop, size_t n_chan)
      : n_fft_(n_fft),
        hop_(hop),
        n_chan_(n_chan),
        n_freq_(n_fft / 2 + 1),
        fft_(n_fft),
        window_(hann_window(n_fft)),
        buffer_(n_chan * n_fft, 0.0),
        scratch_(n_fft),
        cached_frame_(n_chan * (n_fft / 2 + 1), cplx(0.0)),
        primed_(false) {
    if (hop == 0 || hop > n_fft) throw std::invalid_argument("invalid hop size");
  }

  // samples: (hop, n_chan) interleaved.
  // x_next is computed this call; x_cur is the previous call's x_next. On the
  // FIRST call the register is primed with x_next rather than left zero: an
  // all-zero x_cur would drive r2 to 0, phi() to 1/eps, and poison V for
  // hundreds of frames (it only decays at alpha per frame).
  void push(const real* samples, cplx* x_cur, cplx* x_next) {
    for (size_t c = 0; c < n_chan_; c++) {
      real* buf = &buffer_[c * n_fft_];
      std::copy(buf + hop_, buf + n_fft_, buf);  // slide window, oldest hop falls off
      for (size_t i = 0; i < hop_; i++) buf[n_fft_ - hop_ + i] = samples[i * n_chan_ + c];

      for (size_t i = 0; i < n_fft_; i++) scratch_[i] = cplx(buf[i] * window_[i], 0.0);
      fft_.forward(scratch_.data());
      for (size_t f = 0; f < n_freq_; f++) x_next[f * n_chan_ + c] = scratch_[f];
    }
    if (!primed_) {
      std::copy(x_next, x_next + n_freq_ * n_chan_, cached_frame_.begin());
      primed_ = true;
    }
    std::copy(cached_frame_.begin(), cached_frame_.end(), x_cur);
    std::copy(x_next, x_next + n_freq_ * n_chan_, cached_frame_.begin());
  }

  size_t n_freq() const { return n_freq_; }

 private:
  size_t n_fft_, hop_, n_chan_, n_freq_;
  FFT fft_;
  std::vector<real> window_, buffer_;
  std::vector<cplx> scratch_, cached_frame_;
  bool primed_;
};

class StftSynthesis {
 public:
  StftSynthesis(size_t n_fft, size_t hop, size_t n_chan)
      : n_fft_(n_fft),
        hop_(hop),
        n_chan_(n_chan),
        n_freq_(n_fft / 2 + 1),
        fft_(n_fft),
        window_(dual_window(hann_window(n_fft), hop)),
        accum_(n_chan * n_fft, 0.0),
        cached_hop_(n_chan * hop, 0.0),
        new_hop_(n_chan * hop, 0.0),
        scratch_(n_fft) {}

  // frame: (n_freq, n_chan); samples: (hop, n_chan) interleaved.
  // The samples emitted are the hop finished on the PREVIOUS call (1-hop
  // register). new_hop_ is a preallocated member, not a local vector: this runs
  // once per frame and a real-time audio callback must not touch the heap.
  void push(const cplx* frame, real* samples) {
    const real inv_n = 1.0 / real(n_fft_);  // the inverse FFT is unnormalized
    for (size_t c = 0; c < n_chan_; c++) {
      // rebuild the full spectrum from the half we carry (Hermitian symmetry)
      for (size_t f = 0; f < n_freq_; f++) scratch_[f] = frame[f * n_chan_ + c];
      for (size_t f = n_freq_; f < n_fft_; f++) scratch_[f] = std::conj(scratch_[n_fft_ - f]);
      fft_.inverse(scratch_.data());

      real* acc = &accum_[c * n_fft_];
      for (size_t i = 0; i < n_fft_; i++) acc[i] += scratch_[i].real() * inv_n * window_[i];
      for (size_t i = 0; i < hop_; i++) new_hop_[i * n_chan_ + c] = acc[i];
      std::copy(acc + hop_, acc + n_fft_, acc);
      std::fill(acc + n_fft_ - hop_, acc + n_fft_, 0.0);
    }
    std::copy(cached_hop_.begin(), cached_hop_.end(), samples);
    cached_hop_.swap(new_hop_);
  }

 private:
  size_t n_fft_, hop_, n_chan_, n_freq_;
  FFT fft_;
  std::vector<real> window_, accum_, cached_hop_, new_hop_;
  std::vector<cplx> scratch_;
};

}  // namespace piva_online

// =============================================================================
// main
// =============================================================================

using namespace piva_online;

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <input.wav> <output_prefix> <ip|iss> [n_fft=2048] [hop=n_fft/2]"
                 " [n_iter=1] [alpha=0.995] [model=laplace]\n";
    return 1;
  }
  const std::string input_filename(argv[1]), output_prefix(argv[2]), algo(argv[3]);
  const size_t n_fft = argc > 4 ? std::stoul(argv[4]) : 2048;
  const size_t hop = argc > 5 ? std::stoul(argv[5]) : n_fft / 2;
  const size_t n_iter = argc > 6 ? std::stoul(argv[6]) : 1;
  const real alpha = argc > 7 ? real(std::stod(argv[7])) : 0.995;

  try {
    const Model model = model_from_string(argc > 8 ? argv[8] : "laplace");
    const WavData wav = read_wav(input_filename);
    const size_t n_chan = wav.n_chan, n_samples = wav.n_samples;
    const std::vector<real>& audio = wav.samples;

    std::cout << "Input: " << n_chan << " channels, " << n_samples << " samples at "
              << wav.samplerate << " Hz  [standalone double]\n";
    if (n_chan < 2) {
      std::cerr << "Separation needs a multichannel file (one channel per source)\n";
      return 1;
    }

    std::unique_ptr<OnlineAuxIVABase> bss;
    const size_t n_freq = n_fft / 2 + 1;
    if (algo == "ip")
      bss.reset(new AuxIVAIPOnline(n_freq, n_chan, n_iter, alpha, model));
    else if (algo == "iss")
      bss.reset(new AuxIVAISSOnline(n_freq, n_chan, n_iter, alpha, model));
    else {
      std::cerr << "Unknown algorithm " << algo << " (use ip or iss)\n";
      return 1;
    }

    StftAnalysis analysis(n_fft, hop, n_chan);
    StftSynthesis synthesis(n_fft, hop, n_chan);

    // window-overlap latency plus one hop each for the input and output
    // pipeline registers
    const size_t latency = n_fft - hop + 2 * hop;
    const size_t n_frames = (n_samples + latency + hop - 1) / hop;

    std::vector<real> in_hop(hop * n_chan), out_hop(hop * n_chan);
    std::vector<cplx> X_cur(n_freq * n_chan), X_next(n_freq * n_chan), Y(n_freq * n_chan),
        Y_pb(n_freq * n_chan);
    std::vector<real> output(n_frames * hop * n_chan, 0.0);

    double bss_seconds = 0.0;
    for (size_t t = 0; t < n_frames; t++) {
      for (size_t i = 0; i < hop; i++) {
        const size_t s = t * hop + i;
        for (size_t c = 0; c < n_chan; c++)
          in_hop[i * n_chan + c] = s < n_samples ? audio[s * n_chan + c] : 0.0;
      }

      analysis.push(in_hop.data(), X_cur.data(), X_next.data());

      const auto tic = std::chrono::steady_clock::now();
      bss->process_frame(X_cur.data(), X_next.data(), Y.data());
      bss->project_back_frame(Y.data(), Y_pb.data(), 0);
      bss_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - tic).count();

      synthesis.push(Y_pb.data(), out_hop.data());
      std::copy(out_hop.begin(), out_hop.end(), output.begin() + long(t * hop * n_chan));
    }

    const double frame_ms = 1000.0 * bss_seconds / double(n_frames);
    const double hop_ms = 1000.0 * double(hop) / double(wav.samplerate);
    std::cout << "Online AuxIVA-" << (algo == "ip" ? "IP" : "ISS") << ": " << n_frames
              << " frames, " << frame_ms << " ms/frame (hop is " << hop_ms
              << " ms, real-time factor " << frame_ms / hop_ms << ")\n"
              << "Activity gate froze " << bss->frames_gated() << " / " << n_frames
              << " frames as silent; bins skipped as degenerate: " << bss->bins_skipped()
              << "; project-back singular: " << bss->pb_degenerate() << "\n";

    // one file per source, peak-normalized to 0.9
    for (size_t k = 0; k < n_chan; k++) {
      std::vector<real> src(n_samples);
      real peak = 0.0;
      for (size_t s = 0; s < n_samples; s++) {
        src[s] = output[(s + latency) * n_chan + k];
        peak = std::max(peak, std::abs(src[s]));
      }
      if (peak > 0.0)
        for (real& v : src) v *= 0.9 / peak;

      const std::string filename = output_prefix + "_src" + std::to_string(k + 1) + ".wav";
      write_wav(filename, src, 1, wav.samplerate);
      std::cout << "Wrote " << filename << "\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}










