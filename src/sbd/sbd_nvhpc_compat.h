// This code is a Qiskit project.
//
// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// Force-included via -include for the gpu_omp_offload backend only.
// Upstream SBD calls __builtin_ffsl / __builtin_popcountl from inside
// #pragma omp declare target regions. clang/LLVM lowers those to device
// intrinsics fine, but nvc++ emits a call to __blt_pgi_ffsl which only
// exists in libnvc.so (host) — nvlink then fails with
// "Undefined reference to __blt_pgi_ffsl".
//
// We supply a portable bit-twiddling implementation inside declare target
// and shadow the builtins with macros so the headers in correlation.h /
// omp_offload.h pick up our version BEFORE the compiler intrinsifies them.
//
// Guarded on __NVCOMPILER so the shim is a no-op for other compilers.
#ifndef SBD_NVHPC_COMPAT_H
#define SBD_NVHPC_COMPAT_H

#ifdef __NVCOMPILER

#pragma omp begin declare target
static inline int sbd_portable_ffsl(long x) {
    if (x == 0) return 0;
    int p = 1;
    long y = x;
    while ((y & 1L) == 0L) { y >>= 1; ++p; }
    return p;
}
static inline int sbd_portable_popcountl(unsigned long x) {
    int c = 0;
    while (x) { x &= (x - 1UL); ++c; }
    return c;
}
static inline int sbd_portable_popcountll(unsigned long long x) {
    int c = 0;
    while (x) { x &= (x - 1ULL); ++c; }
    return c;
}
#pragma omp end declare target

#define __builtin_ffsl(x)        sbd_portable_ffsl((long)(x))
#define __builtin_popcountl(x)   sbd_portable_popcountl((unsigned long)(x))
#define __builtin_popcountll(x)  sbd_portable_popcountll((unsigned long long)(x))

#endif // __NVCOMPILER
#endif // SBD_NVHPC_COMPAT_H
