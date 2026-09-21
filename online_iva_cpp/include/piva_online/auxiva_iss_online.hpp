// Online (frame-recursive) AuxIVA with iterative source steering (ISS).
// C++ port of piva/auxiva_iss_online.py (arXiv:2209.00937, Algorithm 1).
// DOUBLE-PRECISION build (the reference).
//
// ISS update for pivot source k, per frequency bin, for every source m:
//   v_m = (w_m^H V_m w_k) / (w_k^H V_m w_k)        for m != k
//   v_k = 1 - 1 / sqrt(w_k^H V_k w_k)
//   w_m^H <- w_m^H - v_m w_k^H
// No matrix inversion or linear solve is needed, which also makes this the
// better candidate for an embedded/NEON port: no pivoting branches, so the
// per-frame cost is constant.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
#pragma once

#include "piva_online/common.hpp"

namespace piva_online {

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
    size_t n = n_chan_;

    for (size_t f = 0; f < n_freq_; f++) {
      // frozen pivot row (stores w_k^H)
      for (size_t c = 0; c < n; c++) wk_[c] = W_[w_idx(f, k, c)];

      // all v_m are computed from the W before this update
      //
      // denom is w_k^H V_m w_k, a quadratic form in a positive semi-definite
      // matrix, so it is >= 0 and goes to 0 when V_m does. The two branches
      // below behave very differently as that happens:
      //
      //   m != k : numer and denom are BOTH linear in V_m, so the quotient is
      //            invariant to the scale of V_m and stays well behaved.
      //   m == k : 1/sqrt(denom) is NOT scale-invariant. This is the
      //            normalization that sets the extracted source to unit
      //            variance, and for a silent source that target is
      //            meaningless -- the old `denom = max(denom, eps_)` did not
      //            rescue it, it turned a degenerate denominator into a huge
      //            update (at eps_ = 1e-10, v_k = 1 - 1e5) which then lands
      //            in W_ permanently.
      //
      // So: detect degeneracy and SKIP the bin, leaving W_ untouched, the way
      // AuxIVA-IP already does when its linear solve reports singularity.
      // The activity gate in OnlineAuxIVABase::process_frame() prevents the
      // usual cause (V_ decaying through a silence); this is the backstop for
      // a V_m that is near-singular in the direction of w_k.
      bool degenerate = false;
      for (size_t m = 0; m < n_src_ && !degenerate; m++) {
        const cplx* Vm = &V_[v_idx(m, f, 0, 0)];
        real denom = 0.0;
        cplx numer = 0.0;
        real trace = 0.0;
        for (size_t r = 0; r < n; r++) {
          cplx Vw = 0.0;  // (V_m conj(W[f,k,:]))_r
          for (size_t c = 0; c < n; c++) Vw += Vm[r * n + c] * std::conj(wk_[c]);
          denom += std::real(wk_[r] * Vw);
          numer += W_[w_idx(f, m, r)] * Vw;
          trace += std::real(Vm[r * n + r]);
        }
        // scale-relative test: compare against the size of V_m itself rather
        // than an absolute constant, so it holds for quiet and loud input
        // alike. degen_rel_ is dimensionless and separate from eps_, which
        // guards phi() and has completely different units.
        if (!(denom > degen_rel_ * trace) || !(trace > 0.0)) {
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

}  // namespace piva_online


