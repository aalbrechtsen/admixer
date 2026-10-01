// admixer: CPU dispatch for the per-entry tile kernels (io.hpp, beagle.hpp).
#pragma once

// The portable release build (make release) compiles the per-entry tile kernels twice, for AVX-512
// (x86-64-v4) and for the baseline target, and picks one at program start (GCC function multiversioning);
// these loops are where AVX-512 pays (about 2x). Other builds (-march=native) leave it empty.
#if defined(ADMIXER_MULTIARCH) && defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
#define ADMIXER_KERNEL __attribute__((target_clones("arch=x86-64-v4", "default")))
#else
#define ADMIXER_KERNEL
#endif
