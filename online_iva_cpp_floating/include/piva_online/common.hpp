// Shared pieces of the online AuxIVA-IP / AuxIVA-ISS C++ implementations.
//
// SINGLE-PRECISION (fp32) build. Every sample, coefficient and accumulator is
// `real` == float, so the whole pipeline is 32-bit. This mirrors what an
// ARM+NEON target wants: NEON holds 4 floats per 128-bit register against only
// 2 doubles, and on most embedded ARM cores fp64 throughput is half of fp32 or
// worse. See ../../README.md for what else differs from the double build.
//
// Memory layout (row-major, same axis order as the Python code):
//   x_cur, x_next, y : (n_freq, n_chan)     one STFT frame each
//   W    : (n_freq, n_chan, n_chan)         row k stores w_k^H, so y = W x_cur
//   V    : (n_src, n_freq, n_chan, n_chan)  per-source weighted covariance
//
// process_frame() takes two frames per call, x_cur and x_next (x_next is
// one hop ahead, see StftAnalysis): only x_cur is demixed into y, but the
// instantaneous covariance sample that feeds V's recursive update is the
// average of x_cur x_cur^H and x_next x_next^H.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace piva_online {

using real = float;
using cplx = std::complex<real>;

enum class Model { Laplace, Gauss };

inline Model model_from_string(const std::string& name) {
  if (name == "laplace") return Model::Laplace;
  if (name == "gauss") return Model::Gauss;
  throw std::invalid_argument("No such model " + name);
}

// Auxiliary-variable weight, identical to _phi in auxiva_iss_online.py.
//
// `eps` only bites when r is essentially zero, and then phi returns 1/eps,
// which is enormous. The double build defaults to 1e-10 (so 1e10 here); that
// value is representable in fp32 but is far below anything meaningful at this
// precision, so the float build defaults to a larger eps -- see
// OnlineAuxIVABase's constructor.
inline real phi(real r, Model model, size_t n_freq, real eps) {
  if (model == Model::Laplace) return 1.0f / std::max(eps, 2.0f * r);
  return 1.0f / std::max(eps, (r * r) / real(n_freq));
}

// Solve A x = b in place for a small dense complex system with partial
// pivoting. A (n x n, row-major) is destroyed; the solution is left in b.
// Returns false if the matrix is numerically singular.
//
// NOTE for an embedded port: at n_chan == 2 this can be replaced by the
// closed-form 2x2 inverse, which is branch-free (no pivoting) and therefore
// both vectorizable across frequency and constant-time per frame.
inline bool solve_inplace(cplx* A, cplx* b, size_t n) {
  for (size_t col = 0; col < n; col++) {
    size_t piv = col;
    real best = std::abs(A[col * n + col]);
    for (size_t r = col + 1; r < n; r++) {
      real v = std::abs(A[r * n + col]);
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (best == 0.0f) return false;
    if (piv != col) {
      for (size_t c = 0; c < n; c++) std::swap(A[col * n + c], A[piv * n + c]);
      std::swap(b[col], b[piv]);
    }
    for (size_t r = col + 1; r < n; r++) {
      cplx f = A[r * n + col] / A[col * n + col];
      if (f == cplx(0.0f)) continue;
      for (size_t c = col; c < n; c++) A[r * n + c] -= f * A[col * n + c];
      b[r] -= f * b[col];
    }
  }
  for (size_t i = n; i-- > 0;) {
    cplx s = b[i];
    for (size_t c = i + 1; c < n; c++) s -= A[i * n + c] * b[c];
    b[i] = s / A[i * n + i];
  }
  return true;
}

class OnlineAuxIVABase {
 public:
  // eps defaults to 1e-6f rather than the double build's 1e-10: fp32 has a
  // ~1.2e-7 relative epsilon, so 1e-10 is well below the precision at which
  // the quantities it guards are computed, and 1/1e-10 = 1e10 is a wildly
  // oversized weight. 1e-6f keeps the guard meaningful at this precision.
  //
  // silence_rel: a frame whose mean power is more than this far below the
  // recent peak is treated as inactive -- W_ and V_ are frozen for it. See
  // process_frame(). Default 1e-6 is -60 dB. Set to 0 to disable the gate.
  //
  // eps_ and degen_rel_ are deliberately separate: they guard different
  // quantities with different units, and one value tuned for either is
  // arbitrary for the other.
  //   eps        - absolute floor inside phi(), on 2*r, where r is summed
  //                over all n_freq bins (so O(1)..O(100) for real audio)
  //   degen_rel  - RELATIVE threshold for declaring w_k^H V_m w_k degenerate,
  //                compared against trace(V_m), hence dimensionless
  OnlineAuxIVABase(size_t n_freq, size_t n_chan, size_t n_iter = 1,
                   real alpha = 0.995f, Model model = Model::Laplace,
                   real eps = 1e-6f, real silence_rel = 1e-6f,
                   real degen_rel = 1e-6f)
      : n_freq_(n_freq),
        n_chan_(n_chan),
        n_src_(n_chan),
        n_iter_(n_iter),
        alpha_(alpha),
        model_(model),
        eps_(eps),
        silence_rel_(silence_rel),
        degen_rel_(degen_rel),
        peak_power_(0.0),
        frames_gated_(0),
        bins_skipped_(0),
        W_(n_freq * n_chan * n_chan),
        V_(n_chan * n_freq * n_chan * n_chan),
        V_prev_(V_.size()),
        xxH_(n_freq * n_chan * n_chan),
        xxH_next_(xxH_.size()),
        a_(n_chan * n_chan),
        e_(n_chan) {
    reset();
  }

  virtual ~OnlineAuxIVABase() = default;

  // W = identity per bin, V = 0.01 * identity per source/bin
  void reset() {
    size_t n = n_chan_;
    std::fill(W_.begin(), W_.end(), cplx(0.0f));
    std::fill(V_.begin(), V_.end(), cplx(0.0f));
    for (size_t f = 0; f < n_freq_; f++)
      for (size_t c = 0; c < n; c++) W_[w_idx(f, c, c)] = 1.0f;
    for (size_t k = 0; k < n_src_; k++)
      for (size_t f = 0; f < n_freq_; f++)
        for (size_t c = 0; c < n; c++) V_[v_idx(k, f, c, c)] = 0.01f;
  }

  // Process one STFT frame. x_cur, x_next and y are (n_freq, n_chan),
  // row-major. y is the demixed frame (from x_cur only), without any scale
  // correction. x_next is one hop ahead of x_cur (see StftAnalysis) and is
  // used only to enrich the instantaneous covariance sample this cycle;
  // it does not affect what gets demixed.
  //
  // ACTIVITY GATE: if the frame carries essentially no energy, the adaptation
  // is skipped entirely (W_ and V_ are left untouched) and the frame is just
  // demixed with the existing W_. There is nothing to learn from silence, and
  // adapting on it is actively harmful in two ways:
  //
  //   1. V_ decays as alpha^t whenever xxH is ~0, with nothing flooring it.
  //      Once V_ has decayed, ISS's normalization denominator w_k^H V_k w_k
  //      collapses with it, and `1 - 1/sqrt(denom)` explodes -- measured on a
  //      silence-padded signal, V_'s scale fell 20x below its initialization
  //      in 30 s at alpha = 0.96, and keeps going.
  //   2. r2 -> 0 makes phi() return 1/eps, so `weight` becomes enormous. Any
  //      non-zero xxH riding along with it is then injected into V_ at a
  //      wildly oversized scale.
  //
  // Both are the same failure: a reciprocal of something that legitimately
  // goes to zero when the input is silent. Freezing is the honest response.
  void process_frame(const cplx* x_cur, const cplx* x_next, cplx* y) {
    if (!frame_is_active(x_cur)) {
      ++frames_gated_;
      demix(x_cur, y);
      return;
    }

    // instantaneous covariance sample: the average of x_cur x_cur^H and
    // x_next x_next^H, i.e. both of the frames in flight this cycle
    compute_xxH(x_cur, xxH_);
    compute_xxH(x_next, xxH_next_);
    for (size_t i = 0; i < xxH_.size(); i++) xxH_[i] = 0.5f * (xxH_[i] + xxH_next_[i]);

    // snapshot of V_{t-1}; the recursion is anchored to the previous frame
    std::copy(V_.begin(), V_.end(), V_prev_.begin());

    for (size_t it = 0; it < n_iter_; it++) {
      for (size_t k = 0; k < n_src_; k++) {
        update_covariance(x_cur, x_next, k);
        update_demixing(k);
      }
    }

    demix(x_cur, y);
  }

  // how many frames the activity gate has frozen (diagnostic / tuning aid)
  long frames_gated() const { return frames_gated_; }

  // how many per-bin demixing updates were skipped as degenerate
  long bins_skipped() const { return bins_skipped_; }

  // Scale-corrected output by the minimum distortion principle:
  // y_k <- A[ref, k] y_k with A = W^{-1}, using the current W only, so it
  // stays causal. Writes (n_freq, n_src) into y_scaled.
  void project_back_frame(const cplx* y, cplx* y_scaled, size_t ref = 0) {
    size_t n = n_chan_;
    if (n == 2) {
      // Closed-form 2x2 (cofactor/adjugate) inverse. Same answer as the
      // Gaussian elimination below, but branch-free -- no pivoting -- so the
      // per-bin cost is constant and the loop can vectorize across frequency.
      //
      // Cofactor is not backward stable the way pivoted elimination is: it
      // forms ad and bc at full magnitude and subtracts. At n == 2 with
      // sanely scaled entries that is a non-issue, and the determinant is
      // accumulated in double here to absorb the cancellation, which is the
      // only place it would bite.
      for (size_t f = 0; f < n_freq_; f++) {
        const cplx a = W_[w_idx(f, 0, 0)], b = W_[w_idx(f, 0, 1)];
        const cplx c = W_[w_idx(f, 1, 0)], d = W_[w_idx(f, 1, 1)];
        const std::complex<double> det =
            std::complex<double>(a) * std::complex<double>(d) -
            std::complex<double>(b) * std::complex<double>(c);
        if (det == std::complex<double>(0.0)) {
          y_scaled[f * n + 0] = y[f * n + 0];
          y_scaled[f * n + 1] = y[f * n + 1];
          continue;
        }
        // inv(W) = [d -b; -c a] / det, so row `ref` is (d, -b) or (-c, a)
        const std::complex<double> r0 = (ref == 0) ? std::complex<double>(d)
                                                   : -std::complex<double>(c);
        const std::complex<double> r1 = (ref == 0) ? -std::complex<double>(b)
                                                   : std::complex<double>(a);
        y_scaled[f * n + 0] = cplx(r0 / det) * y[f * n + 0];
        y_scaled[f * n + 1] = cplx(r1 / det) * y[f * n + 1];
      }
      return;
    }
    for (size_t f = 0; f < n_freq_; f++) {
      // column k of A solves W a = e_k; row `ref` of A solves W^T z = e_ref
      for (size_t r = 0; r < n; r++)
        for (size_t c = 0; c < n; c++) a_[r * n + c] = W_[w_idx(f, c, r)];
      for (size_t c = 0; c < n; c++) e_[c] = (c == ref) ? 1.0f : 0.0f;
      bool ok = solve_inplace(a_.data(), e_.data(), n);
      for (size_t k = 0; k < n; k++)
        y_scaled[f * n + k] = ok ? e_[k] * y[f * n + k] : y[f * n + k];
    }
  }

  size_t n_freq() const { return n_freq_; }
  size_t n_chan() const { return n_chan_; }
  size_t n_src() const { return n_src_; }
  const std::vector<cplx>& demixing() const { return W_; }

 protected:
  size_t w_idx(size_t f, size_t r, size_t c) const {
    return (f * n_chan_ + r) * n_chan_ + c;
  }
  size_t v_idx(size_t k, size_t f, size_t r, size_t c) const {
    return ((k * n_freq_ + f) * n_chan_ + r) * n_chan_ + c;
  }

  // Is this frame worth adapting on? The test is relative to a slowly
  // decaying peak, so it is invariant to input gain -- an absolute threshold
  // would have to be retuned for every recording level. peak_power_ is kept
  // in double: it is a long-running accumulator and underflows in fp32.
  bool frame_is_active(const cplx* x_cur) {
    double power = 0.0;
    for (size_t i = 0; i < n_freq_ * n_chan_; i++) power += double(std::norm(x_cur[i]));
    power /= double(n_freq_ * n_chan_);

    peak_power_ = std::max(peak_power_ * kPeakDecay, power);
    if (silence_rel_ <= 0.0f) return true;  // gate disabled
    return power > peak_power_ * double(silence_rel_);
  }

  // x x^H per frequency bin, into out (n_freq, n_chan, n_chan)
  void compute_xxH(const cplx* x, std::vector<cplx>& out) const {
    size_t n = n_chan_;
    for (size_t f = 0; f < n_freq_; f++)
      for (size_t c = 0; c < n; c++)
        for (size_t d = 0; d < n; d++)
          out[(f * n + c) * n + d] = x[f * n + c] * std::conj(x[f * n + d]);
  }

  // Auxiliary variable and recursive covariance for source k.
  // r2 is averaged over both frames in flight, to match xxH_ being the
  // average of x_cur x_cur^H and x_next x_next^H. Taking r2 from x_cur alone
  // would desynchronize the two: at a speech onset (x_cur still quiet,
  // x_next already loud) phi() would go large while xxH_ was large too,
  // injecting an oversized covariance sample that takes many frames to decay.
  //
  // r2 accumulates over n_freq bins, so it is kept in double even in this
  // fp32 build: at n_fft = 2048 that is 1025 additions, and doing them in
  // fp32 loses enough precision to visibly perturb the weight.
  void update_covariance(const cplx* x_cur, const cplx* x_next, size_t k) {
    size_t n = n_chan_;
    double r2 = 0.0;
    for (size_t f = 0; f < n_freq_; f++) {
      cplx y_cur = 0.0f, y_next = 0.0f;
      for (size_t c = 0; c < n; c++) {
        y_cur += W_[w_idx(f, k, c)] * x_cur[f * n + c];
        y_next += W_[w_idx(f, k, c)] * x_next[f * n + c];
      }
      r2 += 0.5 * (double(std::norm(y_cur)) + double(std::norm(y_next)));
    }
    real weight = (1.0f - alpha_) * phi(real(std::sqrt(r2)), model_, n_freq_, eps_);

    size_t block = n * n;
    size_t off = k * n_freq_ * block;
    for (size_t i = 0; i < n_freq_ * block; i++)
      V_[off + i] = alpha_ * V_prev_[off + i] + weight * xxH_[i];
  }

  void demix(const cplx* x_cur, cplx* y) const {
    size_t n = n_chan_;
    for (size_t f = 0; f < n_freq_; f++)
      for (size_t r = 0; r < n; r++) {
        cplx s = 0.0f;
        for (size_t c = 0; c < n; c++) s += W_[w_idx(f, r, c)] * x_cur[f * n + c];
        y[f * n + r] = s;
      }
  }

  virtual void update_demixing(size_t k) = 0;

  // how fast the reference peak forgets, per frame; deliberately much slower
  // than alpha_ so a long pause does not drag the gate threshold down with it
  static constexpr double kPeakDecay = 0.999;

  size_t n_freq_, n_chan_, n_src_, n_iter_;
  real alpha_;
  Model model_;
  real eps_;
  real silence_rel_;
  real degen_rel_;
  double peak_power_;
  long frames_gated_;
  long bins_skipped_;

  std::vector<cplx> W_, V_, V_prev_, xxH_, xxH_next_;
  std::vector<cplx> a_, e_;  // scratch for projection back
};

}  // namespace piva_online

