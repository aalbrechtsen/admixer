CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra
BLAS     ?= -lopenblas
LIBS     := -lz
PREFIX   ?= /usr/local

admixer: src/admixer.cpp src/io.hpp src/log.hpp src/linalg.hpp src/model.hpp src/fit.hpp src/evaladmix.hpp
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ src/admixer.cpp $(BLAS) $(LIBS)

install: admixer
	install -m 755 admixer $(PREFIX)/bin/admixer

clean:
	rm -f admixer

.PHONY: install clean
