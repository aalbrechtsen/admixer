CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra
BLAS     ?= -lopenblas
LIBS     := -lz
PREFIX   ?= /usr/local

admixer: src/admixer.cpp src/io.hpp src/log.hpp src/linalg.hpp src/model.hpp src/fit.hpp src/evaladmix.hpp src/beagle.hpp src/multistart.hpp
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ src/admixer.cpp $(BLAS) $(LIBS)

tests/unit: tests/unit.cpp src/io.hpp src/beagle.hpp src/linalg.hpp src/model.hpp src/multistart.hpp
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ tests/unit.cpp $(BLAS) $(LIBS)

tests/qdist: tests/qdist.cpp src/multistart.hpp
	$(CXX) $(CXXFLAGS) -o $@ tests/qdist.cpp

tools/simgl: tools/simgl.cpp src/io.hpp
	$(CXX) $(CXXFLAGS) -o $@ tools/simgl.cpp $(LIBS)

test: admixer
	tests/run_tests.sh

install: admixer
	install -m 755 admixer $(PREFIX)/bin/admixer

clean:
	rm -f admixer tests/unit tests/qdist tools/simgl

.PHONY: install clean test
