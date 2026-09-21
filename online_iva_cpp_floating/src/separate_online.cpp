// Streaming blind source separation of a multichannel wav file with online
// AuxIVA-IP or AuxIVA-ISS. SINGLE-PRECISION (fp32) build.
//
// Audio is fed hop by hop, as it would arrive from a sound card:
//   hop samples -> streaming STFT -> online IVA -> projection back
//               -> streaming iSTFT -> hop output samples
//
// Usage:
//   separate_online <input.wav> <output_prefix> <ip|iss> [n_fft=2048]
//                   [hop=n_fft/2] [n_iter=1] [alpha=0.995] [model=laplace]
//
// Writes <output_prefix>_src<k>.wav for every source, time-aligned with
// the input (the n_fft + hop samples of pipeline latency are removed).
//
// NOTE: this is a file-based test bench that *simulates* streaming, not a
// real-time implementation. The peak normalization at the end needs the whole
// signal before it can scale any of it, so a real-time build must replace it
// with an AGC or a fixed gain.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
#include "piva_online/compat.hpp"  // must come first: std::clamp / M_PI shim

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "piva_online/auxiva_ip_online.hpp"
#include "piva_online/auxiva_iss_online.hpp"
#include "piva_online/stft_online.hpp"
#include "piva_online/wav_io.hpp"

using namespace piva_online;

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <input.wav> <output_prefix> <ip|iss> [n_fft=2048] [hop=n_fft/2]"
                 " [n_iter=1] [alpha=0.995] [model=laplace]\n";
    return 1;
  }
  std::string input_filename(argv[1]);
  std::string output_prefix(argv[2]);
  std::string algo(argv[3]);
  size_t n_fft = argc > 4 ? std::stoul(argv[4]) : 2048;
  size_t hop = argc > 5 ? std::stoul(argv[5]) : n_fft / 2;
  size_t n_iter = argc > 6 ? std::stoul(argv[6]) : 1;
  // 0.995 assumes stationary (non-moving) sources: at hop = n_fft/2 it keeps
  // roughly 9 s of covariance memory, which suppresses residual interference
  // far better than 0.96 (~1 s) but takes ~25 s to converge. Lower it toward
  // 0.99 if the sources move.
  real alpha = argc > 7 ? real(std::stod(argv[7])) : 0.995f;
  Model model = model_from_string(argc > 8 ? argv[8] : "laplace");

  // read the input file
  WavData wav;
  try {
    wav = read_wav(input_filename);
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  size_t n_chan = wav.n_chan;
  size_t n_samples = wav.n_samples;
  const std::vector<real>& audio = wav.samples;

  std::cout << "Input: " << n_chan << " channels, " << n_samples << " samples at "
            << wav.samplerate << " Hz  [fp32 build]\n";
  if (n_chan < 2) {
    std::cerr << "Separation needs a multichannel file (one channel per source)\n";
    return 1;
  }

  std::unique_ptr<OnlineAuxIVABase> bss;
  size_t n_freq = n_fft / 2 + 1;
  if (algo == "ip")
    bss = std::make_unique<AuxIVAIPOnline>(n_freq, n_chan, n_iter, alpha, model);
  else if (algo == "iss")
    bss = std::make_unique<AuxIVAISSOnline>(n_freq, n_chan, n_iter, alpha, model);
  else {
    std::cerr << "Unknown algorithm " << algo << " (use ip or iss)\n";
    return 1;
  }

  StftAnalysis analysis(n_fft, hop, n_chan);
  StftSynthesis synthesis(n_fft, hop, n_chan);

  // STFT/iSTFT window-overlap latency, plus one hop each for the input-side
  // and output-side double-buffered pipeline registers (see stft_online.hpp)
  size_t latency = n_fft - hop + 2 * hop;
  size_t n_frames = (n_samples + latency + hop - 1) / hop;

  std::vector<real> in_hop(hop * n_chan), out_hop(hop * n_chan);
  std::vector<cplx> X_cur(n_freq * n_chan), X_next(n_freq * n_chan), Y(n_freq * n_chan),
      Y_pb(n_freq * n_chan);
  std::vector<real> output(n_frames * hop * n_chan, 0.0f);

  double bss_seconds = 0.0;

  for (size_t t = 0; t < n_frames; t++) {
    // next hop of input, zero-padded past the end of the file
    for (size_t i = 0; i < hop; i++) {
      size_t s = t * hop + i;
      for (size_t c = 0; c < n_chan; c++)
        in_hop[i * n_chan + c] = s < n_samples ? audio[s * n_chan + c] : 0.0f;
    }

    analysis.push(in_hop.data(), X_cur.data(), X_next.data());

    auto tic = std::chrono::steady_clock::now();
    bss->process_frame(X_cur.data(), X_next.data(), Y.data());
    bss->project_back_frame(Y.data(), Y_pb.data(), 0);
    bss_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - tic).count();

    synthesis.push(Y_pb.data(), out_hop.data());
    std::copy(out_hop.begin(), out_hop.end(), output.begin() + long(t * hop * n_chan));
  }

  double frame_ms = 1000.0 * bss_seconds / double(n_frames);
  double hop_ms = 1000.0 * double(hop) / double(wav.samplerate);
  std::cout << "Online AuxIVA-" << (algo == "ip" ? "IP" : "ISS") << ": " << n_frames
            << " frames, " << frame_ms << " ms/frame (hop is " << hop_ms
            << " ms, real-time factor " << frame_ms / hop_ms << ")\n";
  std::cout << "Activity gate froze " << bss->frames_gated() << " / " << n_frames
            << " frames as silent; bins skipped as degenerate: " << bss->bins_skipped() << "\n";

  // write one file per source, peak-normalized to 0.9 like the Python examples
  for (size_t k = 0; k < n_chan; k++) {
    std::vector<real> src(n_samples);
    real peak = 0.0f;
    for (size_t s = 0; s < n_samples; s++) {
      src[s] = output[(s + latency) * n_chan + k];
      peak = std::max(peak, std::abs(src[s]));
    }
    if (peak > 0.0f)
      for (auto& v : src) v *= 0.9f / peak;

    std::string filename = output_prefix + "_src" + std::to_string(k + 1) + ".wav";
    try {
      write_wav(filename, src, 1, wav.samplerate);
    } catch (const std::exception& e) {
      std::cerr << e.what() << "\n";
      return 1;
    }
    std::cout << "Wrote " << filename << "\n";
  }
}

