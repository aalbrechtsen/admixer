// admixer: CPU dispatch for the per-entry tile kernels (io.hpp, beagle.hpp).
#pragma once

// The portable release build (make release) compiles the per-entry tile kernels three times, for AVX-512
// (x86-64-v4), AVX2 (x86-64-v3) and the baseline target, and picks one at program start (GCC function
// multiversioning); these loops are where wider vectors pay (AVX2 about 1.6x over the baseline's SSE, needs
// -fno-trapping-math, see the Makefile). Other builds (-march=native) leave it empty.
#if defined(ADMIXER_MULTIARCH) && defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
#define ADMIXER_KERNEL __attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
#else
#define ADMIXER_KERNEL
#endif
