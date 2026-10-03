CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -mprefer-vector-width=512 -std=c++17 -Wall -Wextra
BLAS     ?= -lopenblas
LIBS     := -lz
# Always added: without it GCC turns the selects in the per-entry loops (io.hpp, beagle.hpp) into branches
# around arithmetic that could raise a floating-point exception, and does not vectorise them (about 1.5-2x
# slower per entry). No effect on results: admixer never reads the floating-point exception flags.
FPFLAGS  := -fno-trapping-math
PREFIX   ?= /usr/local

admixer: src/admixer.cpp src/io.hpp src/kernel.hpp src/log.hpp src/linalg.hpp src/smallk.hpp src/model.hpp src/fit.hpp src/evaladmix.hpp src/beagle.hpp src/multistart.hpp
	$(CXX) $(CXXFLAGS) $(FPFLAGS) -fopenmp -o $@ src/admixer.cpp $(BLAS) $(LIBS)

tests/unit: tests/unit.cpp src/io.hpp src/kernel.hpp src/beagle.hpp src/linalg.hpp src/smallk.hpp src/model.hpp src/multistart.hpp src/evaladmix.hpp
	$(CXX) $(CXXFLAGS) $(FPFLAGS) -fopenmp -o $@ tests/unit.cpp $(BLAS) $(LIBS)

tests/qdist: tests/qdist.cpp src/multistart.hpp
	$(CXX) $(CXXFLAGS) $(FPFLAGS) -o $@ tests/qdist.cpp

tools/prof_kernels: tools/prof_kernels.cpp src/io.hpp src/kernel.hpp src/beagle.hpp src/linalg.hpp src/smallk.hpp src/model.hpp
	$(CXX) $(CXXFLAGS) $(FPFLAGS) -fopenmp -o $@ tools/prof_kernels.cpp $(BLAS) $(LIBS)

tools/simgl: tools/simgl.cpp src/io.hpp src/kernel.hpp
	$(CXX) $(CXXFLAGS) $(FPFLAGS) -fopenmp -o $@ tools/simgl.cpp $(LIBS)

test: admixer
	tests/run_tests.sh

# Portable binary (see build-static.sh): x86-64-v2 code with AVX-512 versions of the hot kernels (kernel.hpp);
# OpenBLAS (DYNAMIC_ARCH), zlib and the C++/OpenMP runtimes linked statically: only glibc is needed at run time.
REL_ARCH    ?= -march=x86-64-v2 -mtune=generic
OPENBLAS_A  ?= /usr/lib/x86_64-linux-gnu/libopenblas.a
ZLIB_A      ?= /usr/lib/x86_64-linux-gnu/libz.a
DEP_INC     ?=
release: src/admixer.cpp src/io.hpp src/kernel.hpp src/log.hpp src/linalg.hpp src/smallk.hpp src/model.hpp src/fit.hpp src/evaladmix.hpp src/beagle.hpp src/multistart.hpp
	$(CXX) -O3 $(REL_ARCH) -std=c++17 -Wall -Wextra $(FPFLAGS) -fopenmp $(DEP_INC) -DADMIXER_MULTIARCH -c -o admixer.o src/admixer.cpp
	$(CXX) -o admixer admixer.o $(OPENBLAS_A) $(ZLIB_A) $$($(CXX) -print-file-name=libgomp.a) \
	  -static-libstdc++ -static-libgcc -pthread -ldl -lm
	rm -f admixer.o

install: admixer
	install -m 755 admixer $(PREFIX)/bin/admixer

clean:
	rm -f admixer tests/unit tests/qdist tools/simgl tools/prof_kernels

.PHONY: install clean test release
