// Minimal WAV file reading and writing, so the example needs no audio library.
// SINGLE-PRECISION (fp32) build: samples are returned as `real` (float).
//
// Reads PCM 8/16/24/32-bit and IEEE float 32/64-bit files (also the
// WAVE_FORMAT_EXTENSIBLE variant), little-endian, any number of channels.
// Samples are returned in [-1, 1], interleaved (n_samples, n_chan).
// Writes 16-bit PCM.
//
// Note on precision: the conversion itself is done in double and only then
// narrowed to float. 32-bit PCM input has 32 significant bits, more than
// fp32's 24-bit mantissa, so that input loses a little precision here -- for
// 16-bit input (the usual case, and what this project writes) fp32 is
// exact, since 16 bits fit comfortably in a 24-bit mantissa.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "piva_online/common.hpp"

namespace piva_online {

struct WavData {
  int samplerate = 0;
  size_t n_chan = 0;
  size_t n_samples = 0;
  std::vector<real> samples;  // interleaved, (n_samples, n_chan)
};

namespace detail {

inline uint32_t read_u32(const unsigned char* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
inline uint16_t read_u16(const unsigned char* p) { return uint16_t(p[0] | (p[1] << 8)); }

inline void write_u32(std::ofstream& out, uint32_t v) {
  for (int i = 0; i < 4; i++) out.put(char((v >> (8 * i)) & 0xff));
}
inline void write_u16(std::ofstream& out, uint16_t v) {
  out.put(char(v & 0xff));
  out.put(char((v >> 8) & 0xff));
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
  size_t data_size = 0;

  // walk the chunks
  size_t pos = 12;
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
  bool is_pcm = format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32);
  bool is_float = format == 3 && (bits == 32 || bits == 64);
  if (!is_pcm && !is_float)
    throw std::runtime_error(filename + ": unsupported WAV format " + std::to_string(format) +
                             " with " + std::to_string(bits) + " bits");

  size_t bytes_per_sample = bits / 8;
  WavData wav;
  wav.samplerate = int(samplerate);
  wav.n_chan = n_chan;
  wav.n_samples = data_size / (bytes_per_sample * n_chan);
  wav.samples.resize(wav.n_samples * n_chan);

  for (size_t i = 0; i < wav.samples.size(); i++) {
    const unsigned char* p = data + i * bytes_per_sample;
    double v = 0.0;
    if (is_float && bits == 32) {
      float f;
      std::memcpy(&f, p, 4);
      v = f;
    } else if (is_float) {
      std::memcpy(&v, p, 8);
    } else if (bits == 8) {
      v = (double(p[0]) - 128.0) / 128.0;
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

// samples: interleaved (n_samples, n_chan) in [-1, 1]; clipped to 16-bit PCM
inline void write_wav(const std::string& filename, const std::vector<real>& samples,
                      size_t n_chan, int samplerate) {
  std::ofstream out(filename, std::ios::binary);
  if (!out) throw std::runtime_error("Could not write " + filename);

  uint32_t data_size = uint32_t(samples.size() * 2);
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
    // same scale as reading (x / 32768), so 16-bit files round-trip exactly.
    // The rounding is done in double so that values near a .5 boundary land
    // the same way they do in the reference double build.
    long s = std::clamp(std::lround(double(v) * 32768.0), -32768L, 32767L);
    detail::write_u16(out, uint16_t(int16_t(s)));
  }
}

}  // namespace piva_online
