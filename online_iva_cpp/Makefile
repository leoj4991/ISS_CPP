CXX ?= c++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -Iinclude

# Homebrew location of FFTW on macOS, if present
BREW := $(shell command -v brew 2>/dev/null && brew --prefix)
ifneq ($(BREW),)
  PREFIX := $(lastword $(BREW))
  CXXFLAGS += -I$(PREFIX)/include
  LDFLAGS += -L$(PREFIX)/lib -Wl,-rpath,$(PREFIX)/lib
endif

HEADERS = $(wildcard include/piva_online/*.hpp)

separate_online: src/separate_online.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -lfftw3 -o $@

# separate the bundled test mixture with both algorithms
run: separate_online
	mkdir -p output
	./separate_online audio/mixture.wav output/online_ip ip 1024 512
	./separate_online audio/mixture.wav output/online_iss iss 1024 512

clean:
	rm -rf separate_online output

.PHONY: run clean
