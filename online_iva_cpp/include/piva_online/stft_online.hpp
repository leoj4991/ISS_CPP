// Streaming STFT / iSTFT for multichannel audio, built on FFTW.
//
// StftAnalysis takes `hop` new samples per channel and returns two STFT
// frames: x_cur (the frame freshly computed on the *previous* call) and
// x_next (the frame just freshly computed this call). This is a 1-hop
// double-buffered pipeline register on the input side: x_cur is what
// downstream code should actually demix, x_next is only there to enrich
// the covariance estimate one cycle early. StftSynthesis is symmetric on
// the output side: the hop it writes out this call is the one it finished
// computing *last* call, not the one it just computed from this call's
// frame. Each side adds one hop of pure pipeline latency, on top of the
// STFT's own window-overlap latency.
//
// The analysis window is a periodic Hann window and the synthesis window is
// its dual (same construction as piva::windows::make_dual), so a
// pass-through reconstructs the input delayed by n_fft - hop + 2*hop
// samples (the extra 2*hop being the two pipeline registers).
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
#pragma once

#include <fftw3.h>

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include "piva_online/common.hpp"

namespace piva_online {

inline std::vector<double> hann_window(size_t length) {
  std::vector<double> w(length);
  for (size_t n = 0; n < length; n++)
    w[n] = 0.5 * (1.0 - std::cos(2.0 * M_PI * double(n) / double(length)));
  return w;
}

inline std::vector<double> dual_window(const std::vector<double>& awin, size_t hop) {
  long length = long(awin.size());
  std::vector<double> norm(length, 0.0);
  for (long shift = -long((length - 1) / hop) * long(hop); shift < length; shift += long(hop))
    for (long m = std::max(0L, shift); m < std::min(length, length + shift); m++)
      norm[m] += awin[m - shift] * awin[m - shift];
  std::vector<double> swin(length);
  for (long m = 0; m < length; m++) {
    // norm[m] == 0 means no analysis window covers sample m, so the dual
    // window is undefined and perfect reconstruction is impossible. At
    // hop == n_fft only shift 0 contributes and a periodic Hann has
    // awin[0] == 0, so this would silently yield NaN. Fail loudly.
    if (!(norm[m] > 0.0))
      throw std::invalid_argument(
          "hop breaks the COLA condition for this window; use hop < n_fft");
    swin[m] = awin[m] / norm[m];
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
        window_(hann_window(n_fft)),
        buffer_(n_chan * n_fft, 0.0),
        cached_frame_(n_chan * n_freq_, cplx(0.0)),
        primed_(false) {
    if (hop == 0 || hop > n_fft) throw std::invalid_argument("invalid hop size");
    in_ = fftw_alloc_real(n_fft);
    out_ = fftw_alloc_complex(n_freq_);
    plan_ = fftw_plan_dft_r2c_1d(int(n_fft), in_, out_, FFTW_MEASURE);
  }
  ~StftAnalysis() {
    fftw_destroy_plan(plan_);
    fftw_free(in_);
    fftw_free(out_);
  }
  StftAnalysis(const StftAnalysis&) = delete;
  StftAnalysis& operator=(const StftAnalysis&) = delete;

  // samples: (hop, n_chan) interleaved; x_cur, x_next: (n_freq, n_chan)
  // x_next is the frame freshly computed this call; x_cur is the frame that
  // was x_next on the *previous* call (a 1-hop pipeline register), so it's
  // one hop behind x_next. On the very first call the register is primed with
  // x_next rather than left empty: an all-zero x_cur would drive the
  // auxiliary variable r2 to 0, and phi() would return 1/eps (1e10), which
  // injects a wildly oversized covariance sample that poisons V_ for a long
  // time afterwards (it only decays at alpha per frame).
  void push(const double* samples, cplx* x_cur, cplx* x_next) {
    for (size_t c = 0; c < n_chan_; c++) {
      double* buf = &buffer_[c * n_fft_];
      std::copy(buf + hop_, buf + n_fft_, buf);
      for (size_t i = 0; i < hop_; i++) buf[n_fft_ - hop_ + i] = samples[i * n_chan_ + c];

      for (size_t i = 0; i < n_fft_; i++) in_[i] = buf[i] * window_[i];
      fftw_execute(plan_);
      for (size_t f = 0; f < n_freq_; f++)
        x_next[f * n_chan_ + c] = cplx(out_[f][0], out_[f][1]);
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
  std::vector<double> window_, buffer_;
  std::vector<cplx> cached_frame_;
  bool primed_;
  double* in_;
  fftw_complex* out_;
  fftw_plan plan_;
};

class StftSynthesis {
 public:
  StftSynthesis(size_t n_fft, size_t hop, size_t n_chan)
      : n_fft_(n_fft),
        hop_(hop),
        n_chan_(n_chan),
        n_freq_(n_fft / 2 + 1),
        window_(dual_window(hann_window(n_fft), hop)),
        accum_(n_chan * n_fft, 0.0),
        cached_hop_(n_chan * hop, 0.0) {
    in_ = fftw_alloc_complex(n_freq_);
    out_ = fftw_alloc_real(n_fft);
    plan_ = fftw_plan_dft_c2r_1d(int(n_fft), in_, out_, FFTW_MEASURE);
  }
  ~StftSynthesis() {
    fftw_destroy_plan(plan_);
    fftw_free(in_);
    fftw_free(out_);
  }
  StftSynthesis(const StftSynthesis&) = delete;
  StftSynthesis& operator=(const StftSynthesis&) = delete;

  // frame: (n_freq, n_chan); samples: (hop, n_chan) interleaved.
  // samples emitted this call are the hop finished *last* call (a 1-hop
  // pipeline register), one hop behind `frame`. On the very first call the
  // emitted samples are all zero.
  void push(const cplx* frame, double* samples) {
    double inv_n = 1.0 / double(n_fft_);  // FFTW's inverse is unnormalized
    std::vector<double> new_hop(n_chan_ * hop_);
    for (size_t c = 0; c < n_chan_; c++) {
      for (size_t f = 0; f < n_freq_; f++) {
        in_[f][0] = frame[f * n_chan_ + c].real();
        in_[f][1] = frame[f * n_chan_ + c].imag();
      }
      fftw_execute(plan_);

      double* acc = &accum_[c * n_fft_];
      for (size_t i = 0; i < n_fft_; i++) acc[i] += out_[i] * inv_n * window_[i];
      for (size_t i = 0; i < hop_; i++) new_hop[i * n_chan_ + c] = acc[i];
      std::copy(acc + hop_, acc + n_fft_, acc);
      std::fill(acc + n_fft_ - hop_, acc + n_fft_, 0.0);
    }
    std::copy(cached_hop_.begin(), cached_hop_.end(), samples);
    cached_hop_.swap(new_hop);
  }

 private:
  size_t n_fft_, hop_, n_chan_, n_freq_;
  std::vector<double> window_, accum_, cached_hop_;
  fftw_complex* in_;
  double* out_;
  fftw_plan plan_;
};

}  // namespace piva_online

