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

/**
 * @file python/bindings.cpp
 * @brief Python bindings for SBD TPB diagonalization using pybind11
 *
 * This file is compiled once per backend, with different module names + flags:
 * - _core_cpu                : CPU backend (host OpenMP via -fopenmp)
 * - _core_gpu_thrust         : Thrust GPU backend  (with -DSBD_THRUST,    nvc++ -cuda)
 * - _core_gpu_omp_offload    : OpenMP-offload GPU  (with -DUSE_OMP_OFFLOAD), built by
 *                              nvc++ -mp=gpu on NVIDIA, or amdclang++
 *                              --offload-arch=gfx* on AMD. ONE module and one
 *                              'gpu-omp' device string serve both vendors --
 *                              same source, same macros, different compiler --
 *                              with the target recorded as
 *                              __sbd_offload_target__ (see SBD_OFFLOAD_TARGET).
 *
 * The module name is controlled by the SBD_MODULE_NAME macro.
 */

// SBD's mpi_utility.h uses std::cout without including <iostream>.
// Linux/libstdc++ pulls it in transitively; macOS/libc++ doesn't.
// Force-include here before any SBD header so the patch stays in our
// repo rather than in the vendored upstream submodule.
#include <iostream>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <complex>
#include <type_traits>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <mpi4py/mpi4py.h>
#include <mpi.h>

#include "sbd/sbd.h"

#ifdef USE_OMP_OFFLOAD
#include <omp.h>
#endif

namespace py = pybind11;

/**
 * How many offload devices this process can see, or 0 on a CPU build.
 *
 * Pure query: no device is selected and no context created, so this is safe
 * to call before anything else has touched the GPU.
 */
static int sbd_device_count() {
#if defined(SBD_THRUST) && defined(__CUDACC__)
    int n = 0; cudaGetDeviceCount(&n); return n;
#elif defined(SBD_THRUST)
    int n = 0; hipGetDeviceCount(&n); return n;
#elif defined(USE_OMP_OFFLOAD)
    return omp_get_num_devices();
#else
    return 0;
#endif
}


/**
 * Helper function to convert mpi4py communicator to MPI_Comm
 */
MPI_Comm get_mpi_comm(py::object py_comm) {
    PyObject* py_comm_ptr = py_comm.ptr();
    MPI_Comm* comm_ptr = PyMPIComm_Get(py_comm_ptr);
    if (!comm_ptr) {
        throw std::runtime_error("Invalid MPI communicator");
    }
    return *comm_ptr;
}

#ifdef USE_OMP_OFFLOAD
/**
 * Pin this rank to one offload device: device = mpi_rank % n_devices.
 *
 * Note: when this .so is loaded via Python dlopen, the symbol
 * omp_get_num_devices binds to libomp.so's stub (which returns 0 because libomp
 * itself doesn't manage offload devices) instead of libomptarget's working
 * version. omp_set_default_device IS shared between the two, so once we know the
 * count we can still set the device correctly. So fall back to counting the
 * entries in the vendor's device-visibility variable when the count reads 0.
 *
 * The variable to read is vendor-specific, and the order is decided at COMPILE
 * time from the target this module was built for rather than by probing all of
 * them: that keeps the NVIDIA build reading exactly CUDA_VISIBLE_DEVICES first,
 * as it always has. The other vendor's names are still listed as a fallback,
 * which costs nothing (they are unset on a single-vendor host) and helps on an
 * oddly-configured node.
 */
static void sbd_pin_offload_device(int mpi_rank) {
    int n_dev = omp_get_num_devices();
    if (n_dev <= 0) {
        static const char* const kVisibleVars[] = {
#ifdef SBD_OFFLOAD_VENDOR_AMD
            "ROCR_VISIBLE_DEVICES",   // AMD: honoured by the ROCm OMP runtime
            "HIP_VISIBLE_DEVICES",    // AMD: HIP-level equivalent
            "CUDA_VISIBLE_DEVICES",
#else
            "CUDA_VISIBLE_DEVICES",   // NVIDIA
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
#endif
        };
        for (const char* var : kVisibleVars) {
            const char* val = std::getenv(var);
            if (val && *val) {
                n_dev = 1;
                for (const char* p = val; *p; ++p) {
                    if (*p == ',') ++n_dev;
                }
                break;
            }
        }
    }
    if (n_dev > 0) {
        omp_set_default_device(mpi_rank % n_dev);
    }
}
#endif

// Module name is set by compiler flag, e.g.
//   -DSBD_MODULE_NAME=_core_cpu | _core_gpu_thrust | _core_gpu_omp_offload
#ifndef SBD_MODULE_NAME
#define SBD_MODULE_NAME _core
#endif

// ---------------------------------------------------------------------------
// GDB distributed-basis helpers
//
// GDB's matvec is systolic over b_comm: each rank owns the bra rows of its basis
// block, and the ket vector plus its index map rotate around the ring
// (gdb/mult.h:39-41 for each rank's initial offset, :196-202 for the hops). Two
// consequences drive everything below.
//
//   1. b_comm is the ONLY dimension that shards memory. h_comm is a row stride
//      within a block (mult.h:70) closed by an allreduce (:208), and MakeHelpers
//      ignores it, so every h-rank builds the full excitation lookup.
//   2. The ring has exactly mpi_size_b stations and one "task" is one station, so
//      t_comm_size <= b_comm_size is structural.
//
// Upstream checks neither, nor that t*b divides the rank count. The failures are
// a segfault and silently ragged communicators respectively, so they are checked
// here instead.
// ---------------------------------------------------------------------------

/** GDB determinant placement over b_comm. The six schemes of the upstream app. */
enum class SbdDetDistribution {
    input, equal_bra_a, count, count_sorted, grid_cyclic, grid_cyclic_balanced
};

/**
 * Resolve a scheme name, accepting underscores for dashes as the app does.
 *
 * Empty means the default. The app reaches its default through three legacy
 * booleans with a priority rule (do_redist_alpha_eq, which defaults true, then
 * do_sort_det, then do_redist_det -- main.cc:33-56); those are deliberately not
 * exposed here, since they encode a four-way choice as three flags where two are
 * unreachable unless the first is explicitly zeroed. The name says it directly.
 */
static SbdDetDistribution sbd_resolve_det_distribution(const std::string& raw) {
    std::string name = raw;
    std::replace(name.begin(), name.end(), '_', '-');
    if (name.empty() || name == "equal-bra-a") return SbdDetDistribution::equal_bra_a;
    if (name == "input")                 return SbdDetDistribution::input;
    if (name == "count")                 return SbdDetDistribution::count;
    if (name == "count-sorted")          return SbdDetDistribution::count_sorted;
    if (name == "grid-cyclic")           return SbdDetDistribution::grid_cyclic;
    if (name == "grid-cyclic-balanced")  return SbdDetDistribution::grid_cyclic_balanced;
    throw std::invalid_argument(
        "unknown determinant_distribution '" + raw + "'; expected one of: input, "
        "equal-bra-a (default), count, count-sorted, grid-cyclic, grid-cyclic-balanced");
}

static const char* sbd_det_distribution_name(SbdDetDistribution d) {
    switch (d) {
        case SbdDetDistribution::input:                return "input";
        case SbdDetDistribution::equal_bra_a:          return "equal-bra-a";
        case SbdDetDistribution::count:                return "count";
        case SbdDetDistribution::count_sorted:         return "count-sorted";
        case SbdDetDistribution::grid_cyclic:          return "grid-cyclic";
        case SbdDetDistribution::grid_cyclic_balanced: return "grid-cyclic-balanced";
    }
    return "unknown";
}

/**
 * True on every rank iff local_ok holds on every rank of comm.
 *
 * Every per-rank check below routes through this. Throwing on one rank only
 * would leave the others inside the next collective, so the verdict has to be
 * unanimous before anyone raises.
 */
static bool sbd_all_ranks_ok(bool local_ok, MPI_Comm comm) {
    int local = local_ok ? 1 : 0;
    int all = 0;
    MPI_Allreduce(&local, &all, 1, MPI_INT, MPI_LAND, comm);
    return all != 0;
}

/**
 * The problem the lowest-numbered rank found, on every rank; empty if none did.
 *
 * For checks only some ranks can make (the one that reads a file), so every rank
 * raises the same, informative message rather than "another rank failed".
 */
static std::string sbd_first_problem(const std::string& local, MPI_Comm comm) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int mine = local.empty() ? size : rank, first = size;
    MPI_Allreduce(&mine, &first, 1, MPI_INT, MPI_MIN, comm);
    if (first == size) return std::string();
    int length = rank == first ? int(local.size()) : 0;
    MPI_Bcast(&length, 1, MPI_INT, first, comm);
    std::string message(rank == first ? local : std::string(size_t(length), ' '));
    MPI_Bcast(message.data(), length, MPI_CHAR, first, comm);
    return message;
}

#ifndef USE_OMP_OFFLOAD
/**
 * Read a CAOP Hamiltonian file's coefficients on every rank, as upstream's loader
 * would, and refuse together if any cannot be read.
 *
 * Upstream reads the file on h-rank 0 only and sends the terms on
 * (caop/basic/loadmodel.h:92-96), so a line it cannot read raises on that rank while
 * the others wait for terms: the job hangs on more than one rank. This uses
 * upstream's line rules (blank and '#' lines skipped, the first other line the
 * statistics, then one term per line, coefficient first). A real call reads a
 * coefficient as a double, so "(re,im)" is refused there with a pointer to
 * complex=True.
 */
struct CaopFileFacts {
    bool fermion = false;     // statistics zero or negative, as upstream's loader reads it
    size_t max_mode = 0;      // highest mode index any term uses
};

static CaopFileFacts caop_check_hamfile(const std::string& hamfile, bool complex,
                                        MPI_Comm comm, const char* where) {
    std::string problem;
    CaopFileFacts facts;
    std::ifstream ifs(hamfile);
    if (!ifs) {
        problem = "cannot open the Hamiltonian file " + hamfile;
    } else {
        std::string line;
        bool statistics_read = false;
        size_t number = 0;
        while (problem.empty() && std::getline(ifs, line)) {
            ++number;
            const auto first = line.find_first_not_of(" \t");
            if (first == std::string::npos || line[first] == '#') continue;
            std::istringstream iss(line);
            if (!statistics_read) {
                int statistics;
                if (!(iss >> statistics)) {
                    problem = "line " + std::to_string(number) + " of " + hamfile
                        + ": the first line must be the statistics (an integer: positive "
                          "for hard-core bosons or spins, zero or negative for fermions)";
                }
                facts.fermion = statistics <= 0;
                statistics_read = true;
                continue;
            }
            bool read;
            if (complex) {
                std::complex<double> c;
                read = static_cast<bool>(iss >> c);
            } else {
                double c;
                read = static_cast<bool>(iss >> c);
            }
            // The operator tokens, each followed by its mode, up to a '#' comment.
            std::string op;
            size_t mode = 0;
            while (read && iss >> op && op[0] != '#') {
                if (iss >> mode) facts.max_mode = std::max(facts.max_mode, mode);
                else problem = "line " + std::to_string(number) + " of " + hamfile
                         + ": operator " + op + " has no mode index after it";
            }
            if (!read) {
                const bool looks_complex = line[first] == '(';
                problem = "line " + std::to_string(number) + " of " + hamfile
                    + ": cannot read the coefficient '"
                    + line.substr(first, line.find_first_of(" \t", first) - first) + "'"
                    + (looks_complex && !complex
                        ? ". It is written as a complex number (re,im): use complex=True "
                          "(--complex in the drivers)"
                        : ". A coefficient is a real number, or (re,im) with complex=True");
            }
        }
        if (problem.empty() && !statistics_read) {
            problem = hamfile + " has no statistics line";
        }
    }
    const std::string first = sbd_first_problem(problem, comm);
    if (!first.empty()) {
        throw std::invalid_argument(std::string(where) + ": " + first);
    }
    return facts;
}

/**
 * Refuse, on every rank, a CAOP file whose terms use a mode at or beyond sites.
 *
 * Upstream checks max_index() on the h-sharded operator, so with h_comm_size > 1 only
 * the shard holding the high mode throws and the others hang.
 */
static void caop_check_modes(const std::string& file, const CaopFileFacts& facts,
                             size_t sites, const char* where) {
    if (facts.max_mode >= sites) {
        throw std::invalid_argument(
            std::string(where) + ": " + file + " uses mode " + std::to_string(facts.max_mode)
            + ", beyond sites=" + std::to_string(sites) + " (modes are 0 .. sites-1)");
    }
}
#endif

#ifndef USE_OMP_OFFLOAD
/**
 * Check a CAOP state saved by SaveWavefunction against the scalar type and the basis
 * files it will be loaded onto, on every rank, before upstream's LoadWavefunction.
 *
 * The file is [n][words][n x words determinant words][n amplitudes] with no type tag,
 * and LoadWavefunction neither checks its size nor reports what it drops: it maps the
 * saved amplitudes onto the basis by determinant and normalizes the result. So a real
 * file read as complex (or the reverse), an unnormalized state, or a basis that is
 * not the saved state's all give a wrong answer with no error. Rank r < shards reads
 * shard r, as the loader does. The basis is compared by count and an
 * order-independent hash of its determinants, with duplicates refused, which upstream
 * otherwise aborts on.
 */
static void caop_check_state(const std::string& loadname, size_t shards,
                             const std::vector<std::string>& basisfiles, size_t sites,
                             size_t bit_length, bool complex, MPI_Comm comm,
                             const char* where) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    const size_t words = (sites + bit_length - 1) / bit_length;
    const size_t elem = complex ? sizeof(std::complex<double>) : sizeof(double);
    auto mix = [](uint64_t x) {            // splitmix64: a well-spread 64-bit hash
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    };
    auto hash_det = [&](const uint64_t* w) {
        uint64_t h = 0;
        for (size_t k = 0; k < words; ++k) h = mix(h ^ w[k]);
        return h;
    };
    std::string problem;
    double norm2 = 0.0;
    uint64_t state_count = 0, state_hash = 0, basis_count = 0, basis_hash = 0;
    if (size_t(rank) < shards) {
        std::ostringstream name;
        name << loadname << std::setw(6) << std::setfill('0') << rank << ".bin";
        std::ifstream in(name.str(), std::ios::binary | std::ios::ate);
        if (!in) {
            problem = "cannot read the saved state " + name.str();
        } else {
            const auto bytes = size_t(in.tellg());
            in.seekg(0);
            uint64_t n = 0, w = 0;
            in.read(reinterpret_cast<char*>(&n), sizeof n);
            in.read(reinterpret_cast<char*>(&w), sizeof w);
            const size_t real_bytes = 16 + n * (w * 8 + sizeof(double));
            const size_t complex_bytes = 16 + n * (w * 8 + sizeof(std::complex<double>));
            if (!in || w != words) {
                problem = name.str() + ": its determinants are " + std::to_string(w)
                    + " words, where sites=" + std::to_string(sites) + " and bit_length="
                    + std::to_string(bit_length) + " give " + std::to_string(words)
                    + "; use the run's own system_size and bit_length";
            } else if (bytes != (complex ? complex_bytes : real_bytes)) {
                problem = name.str() + (bytes == (complex ? real_bytes : complex_bytes)
                    ? std::string(" holds ") + (complex ? "real" : "complex")
                      + " amplitudes: pass complex=" + (complex ? "False" : "True")
                      + ", as the state was solved with"
                    : " is truncated or not a saved CAOP state");
            } else {
                std::vector<uint64_t> det(n * words);
                in.read(reinterpret_cast<char*>(det.data()), det.size() * 8);
                std::vector<double> amp(n * elem / sizeof(double));
                in.read(reinterpret_cast<char*>(amp.data()), amp.size() * sizeof(double));
                for (double a : amp) norm2 += a * a;      // |c|^2: re^2 + im^2 if complex
                std::vector<uint64_t> hashes(n);
                for (size_t i = 0; i < n; ++i) hashes[i] = hash_det(&det[i * words]);
                for (uint64_t h : hashes) state_hash += h;
                state_count = n;
            }
        }
        // The basis lines this rank's loader would read: every file, as the loader
        // gives each loading rank its share of them; count and hash all of them on
        // rank 0 only, so the total is not counted once per loading rank.
        if (rank == 0) {
            std::vector<uint64_t> hashes;
            for (const auto& file : basisfiles) {
                std::ifstream bf(file);
                if (!bf) {
                    if (problem.empty()) problem = "cannot open the basis file " + file;
                    break;
                }
                std::string line;
                std::vector<uint64_t> wv(words);
                while (std::getline(bf, line)) {
                    if (line.empty()) continue;
                    std::fill(wv.begin(), wv.end(), 0);
                    const size_t len = std::min(line.size(), sites);
                    for (size_t j = 0; j < len; ++j) {
                        if (line[j] == '1') {
                            const size_t bit = sites - 1 - j;
                            wv[bit / bit_length] |= uint64_t(1) << (bit % bit_length);
                        }
                    }
                    hashes.push_back(hash_det(wv.data()));
                }
            }
            std::sort(hashes.begin(), hashes.end());
            if (problem.empty()
                && std::adjacent_find(hashes.begin(), hashes.end()) != hashes.end()) {
                problem = "the basis files list a determinant more than once";
            }
            basis_count = hashes.size();
            for (uint64_t h : hashes) basis_hash += h;
        }
    }
    const std::string first = sbd_first_problem(problem, comm);
    if (!first.empty()) {
        throw std::invalid_argument(std::string(where) + ": " + first);
    }
    uint64_t local[4] = {state_count, state_hash, basis_count, basis_hash};
    uint64_t total[4] = {0, 0, 0, 0};
    MPI_Allreduce(local, total, 4, MPI_UINT64_T, MPI_SUM, comm);
    double norm2_total = 0.0;
    MPI_Allreduce(&norm2, &norm2_total, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (total[0] != total[2] || total[1] != total[3]) {
        throw std::invalid_argument(
            std::string(where) + ": the basis files (" + std::to_string(total[2])
            + " determinants) are not the saved state's (" + std::to_string(total[0])
            + "); give the basis files the state was solved over");
    }
    if (!(std::abs(norm2_total - 1.0) <= 1e-8)) {
        throw std::invalid_argument(
            std::string(where) + ": the saved state's norm squared is "
            + std::to_string(norm2_total) + ", not 1; the evaluator needs a normalized "
              "state, as the solver saves it");
    }
}
#endif

/**
 * Refuse a Thrust solve with one OpenMP thread, before upstream aborts the job.
 *
 * Upstream's Thrust builds keep the Davidson subspace and part of the setup on the
 * host and call MPI_Abort when omp_get_max_threads() == 1 (tpb/sbdiag.h,
 * basic/subspace_thrust.h, caop/basic/sbdiag.h), which ends the Python process with
 * no exception to catch. Raise ValueError on every rank first instead. CPU and
 * OpenMP-offload builds are not affected.
 */
static void sbd_require_openmp_threads(const char* where, MPI_Comm comm) {
#ifdef SBD_THRUST
    if (!sbd_all_ranks_ok(omp_get_max_threads() > 1, comm)) {
        throw std::invalid_argument(
            std::string(where) + " on the gpu (Thrust) backend needs more than one "
            "OpenMP thread per rank: upstream SBD does part of the work on the host "
            "and aborts the job otherwise. Set OMP_NUM_THREADS to 2 or more, and give "
            "each rank that many cores (for example srun --cpus-per-task)");
    }
#else
    (void)where;
    (void)comm;
#endif
}

/**
 * Order-insensitive-after-sort 64-bit fingerprint of a shard (FNV-1a).
 *
 * Used to compare the replicas that ranks sharing a b_comm position must pass:
 * the in-memory diag never broadcasts the determinant list (contrast the
 * file-based overload's MpiBcast(det,0,h_comm) at gdb/sbdiag.h:764), so a
 * divergence there is silently wrong rather than detected. Taken after the local
 * sort, so a caller that supplies the same set in a different order still agrees.
 */
static uint64_t sbd_shard_fingerprint(const sbd::det_vector<size_t>& det) {
    uint64_t h = 1469598103934665603ULL;
    const auto& flat = det.cflat();
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(flat.data());
    const size_t n = flat.size() * sizeof(size_t);
    for (size_t i = 0; i < n; ++i) {
        h ^= static_cast<uint64_t>(bytes[i]);
        h *= 1099511628211ULL;
    }
    // Fold in the row count so an empty shard and a zero-filled one differ.
    h ^= static_cast<uint64_t>(det.size()) + 0x9e3779b97f4a7c15ULL;
    return h;
}

PYBIND11_MODULE(SBD_MODULE_NAME, m) {
    // Set module docstring based on backend
#ifdef SBD_THRUST
    m.doc() = "Python bindings for SBD (Selected Basis Diagonalization) library - GPU backend";
#else
    m.doc() = "Python bindings for SBD (Selected Basis Diagonalization) library - CPU backend";
#endif

    // Which GPU target this module was compiled for:
    //   gpu-omp on AMD     "amdgcn-amd-amdhsa:gfx90a"
    //   gpu-omp on NVIDIA  "nvptx64-nvidia-cuda:cc90"
    //   gpu (Thrust)       "cuda:cc90"
    //   cpu                None
    // For the OMP-offload backend this is the only way to tell the vendor apart,
    // since one module and one device string ('gpu-omp') serve both. For Thrust
    // the vendor is never in doubt but the architecture is -- and an arch
    // mismatch is the usual reason a module built elsewhere will not run here.
#ifdef SBD_OFFLOAD_TARGET
    m.attr("__sbd_offload_target__") = py::str(SBD_OFFLOAD_TARGET);
#else
    m.attr("__sbd_offload_target__") = py::none();
#endif

    // Initialize mpi4py
    if (import_mpi4py() < 0) {
        throw std::runtime_error("Failed to import mpi4py");
    }

    // ========================================================================
    // Bind FCIDump structure
    // ========================================================================
    py::class_<sbd::FCIDump>(m, "FCIDump", py::module_local(), "FCIDUMP data structure")
        .def(py::init<>())
        .def_readwrite("header", &sbd::FCIDump::header,
                      "Header information as dictionary (map<string, string>)")
        .def_readwrite("integrals", &sbd::FCIDump::integrals,
                      "Integral data as list of tuples (value, i, j, k, l)");

    // ========================================================================
    // Bind TPB SBD configuration structure
    // ========================================================================
        // h_comm_size is deliberately NOT exposed. Upstream declares the field
        // (sbdiag.h) but never reads it: diag() shadows it with a local
        //     h_comm_size = mpi_size / (task_comm_size * base_comm_size)
        // and passes THAT to DetBasisCommunicator. So the attribute could only
        // ever report 1 and ignore whatever you assigned, which is worse than
        // absent -- see issue #22. The helper dimension is derived; to change it,
        // change the rank count or the other three sizes.
    py::class_<sbd::tpb::SBD>(m, "TPB_SBD", py::module_local(), "Configuration for TPB diagonalization")
        .def(py::init<>())
        .def_readwrite("task_comm_size", &sbd::tpb::SBD::task_comm_size,
                      "Task communicator size")
        .def_readwrite("adet_comm_size", &sbd::tpb::SBD::adet_comm_size,
                      "Alpha determinant communicator size")
        .def_readwrite("bdet_comm_size", &sbd::tpb::SBD::bdet_comm_size,
                      "Beta determinant communicator size")
        .def_readwrite("method", &sbd::tpb::SBD::method,
                      "Diagonalization method (0=Davidson, 1=Davidson+Ham, 2=Lanczos, 3=Lanczos+Ham)")
        .def_readwrite("max_it", &sbd::tpb::SBD::max_it,
                      "Maximum number of iterations")
        .def_readwrite("max_nb", &sbd::tpb::SBD::max_nb,
                      "Maximum number of basis vectors")
        .def_readwrite("eps", &sbd::tpb::SBD::eps,
                      "Convergence tolerance")
        .def_readwrite("max_time", &sbd::tpb::SBD::max_time,
                      "Maximum time in seconds")
        .def_readwrite("init", &sbd::tpb::SBD::init,
                      "Initialization method")
        .def_readwrite("seed", &sbd::tpb::SBD::seed,
                      "Seed for the random initial vector (init = 1)")
        .def_readwrite("initial_adeterminant_bitstring",
                      &sbd::tpb::SBD::initial_adeterminant_bitstring,
                      "Start Davidson with unit weight on this alpha string (NORB "
                      "characters, character NORB-1-i is orbital i, so the rightmost is orbital 0) times "
                      "initial_bdeterminant_bitstring, instead of the init policy. "
                      "Empty (the default) uses init. Each string must occur exactly "
                      "once in its list, or the solve raises ValueError")
        .def_readwrite("initial_bdeterminant_bitstring",
                      &sbd::tpb::SBD::initial_bdeterminant_bitstring,
                      "Beta half of the explicit start. Empty uses the alpha string "
                      "for beta; it cannot be set without the alpha one")
#ifdef SBD_THRUST
        .def_readwrite("cpu_subspace", &sbd::tpb::SBD::cpu_subspace,
                      "gpu (Thrust) backend only: keep the Davidson/Lanczos subspace "
                      "vectors in host memory instead of GPU memory, so the GPU memory "
                      "they use no longer grows with max_nb; H is still applied on the "
                      "GPU. Energies agree with the default to within rounding")
#endif
        .def_readwrite("do_shuffle", &sbd::tpb::SBD::do_shuffle,
                      "Shuffle determinants flag")
        .def_readwrite("do_rdm", &sbd::tpb::SBD::do_rdm,
                      "Calculate RDM flag (0=density only, 1=full RDM)")
        .def_readwrite("carryover_type", &sbd::tpb::SBD::carryover_type,
                      "Carryover determinant selection type")
        .def_readwrite("ratio", &sbd::tpb::SBD::ratio,
                      "Carryover ratio")
        .def_readwrite("threshold", &sbd::tpb::SBD::threshold,
                      "Carryover threshold")
        .def_readwrite("bit_length", &sbd::tpb::SBD::bit_length,
                      "Bit length for determinant representation")
        .def_readwrite("dump_matrix_form_wf", &sbd::tpb::SBD::dump_matrix_form_wf,
                      "Path to write the full adet x bdet amplitude matrix to, "
                      "gathered onto one rank. A .bin path writes a bare "
                      "row-major float64 array with no header; .dat/.txt write "
                      "a labelled text table. Distinct from tpb_diag's "
                      "savename, which writes per-rank blocks with a header "
                      "for restart -- see the tpb_diag docstring.")
#ifdef SBD_THRUST
        .def_readwrite("use_precalculated_dets", &sbd::tpb::SBD::use_precalculated_dets,
                      "Use precalculated determinants (THRUST)")
        .def_readwrite("max_memory_gb_for_determinants", &sbd::tpb::SBD::max_memory_gb_for_determinants,
                      "Maximum memory in GB for determinants (THRUST)")
        .def_readwrite("thrust_collapse_loops", &sbd::tpb::SBD::thrust_collapse_loops,
                      "Run each task's bra x excitation loops as one flattened GPU "
                      "loop (THRUST; default true). Only used when "
                      "use_precalculated_dets is on; false uses nested loops, which "
                      "need less GPU memory. Upstream turns it off by itself when "
                      "the flattened size would overflow. Energies are unchanged.")
#endif
        ;

    // ========================================================================
    // Bind GDB SBD configuration structure
    //
    // GDB spans the subspace with an explicit list of full determinants, rather
    // than with the Cartesian product of alpha and beta determinants that TPB
    // uses. It therefore has one determinant list instead of two, and one basis
    // communicator (b_comm) instead of the adet/bdet pair.
    // ========================================================================
        // h_comm_size is deliberately NOT exposed. Upstream declares the field
        // (sbdiag.h) but never reads it: diag() shadows it with a local
        //     h_comm_size = mpi_size / (task_comm_size * base_comm_size)
        // and passes THAT to DetBasisCommunicator. So the attribute could only
        // ever report 1 and ignore whatever you assigned, which is worse than
        // absent -- see issue #22. The helper dimension is derived; to change it,
        // change the rank count or the other three sizes.
    py::class_<sbd::gdb::SBD>(m, "GDB_SBD", py::module_local(), "Configuration for GDB diagonalization")
        .def(py::init<>())
        .def_readwrite("t_comm_size", &sbd::gdb::SBD::t_comm_size,
                      "Task communicator size")
        .def_readwrite("b_comm_size", &sbd::gdb::SBD::b_comm_size,
                      "Basis communicator size")
        // GDB implements Davidson only -- there is no Lanczos anywhere under
        // include/sbd/chemistry/gdb/, so TPB's methods 2 and 3 do not exist
        // here. gdb_diag rejects them rather than passing them through: see the
        // check in gdb_diag for what upstream does with an out-of-range value.
        .def_readwrite("method", &sbd::gdb::SBD::method,
                      "Diagonalization method (0=Davidson, 1=Davidson+Ham). GDB has "
                      "no Lanczos; TPB's 2 and 3 are not available")
        .def_readwrite("max_it", &sbd::gdb::SBD::max_it,
                      "Maximum number of iterations")
        .def_readwrite("max_nb", &sbd::gdb::SBD::max_nb,
                      "Maximum number of basis vectors")
        .def_readwrite("eps", &sbd::gdb::SBD::eps,
                      "Convergence tolerance")
        .def_readwrite("max_time", &sbd::gdb::SBD::max_time,
                      "Maximum time in seconds")
        .def_readwrite("init", &sbd::gdb::SBD::init,
                      "Initialization method")
        .def_readwrite("seed", &sbd::gdb::SBD::seed,
                      "Seed for the initial vector")
        .def_readwrite("initial_determinant_bitstring",
                      &sbd::gdb::SBD::initial_determinant_bitstring,
                      "Start Davidson with unit weight on this full determinant "
                      "(2*NORB characters; from the right, bit 2*i is alpha orbital i and bit "
                      "2*i+1 beta orbital i, as in from_strings), instead of the init policy. Empty (the default) "
                      "uses init. It must occur exactly once across all shards, or "
                      "the solve raises ValueError")
        // do_shuffle is deliberately NOT exposed, for the same reason as
        // h_comm_size: GDB declares the field (gdb/sbdiag.h:25), parses it from
        // argv (:94) and copies it into a local (:219), then never reads it
        // again -- the only other "shuffle" under gdb/ is an unrelated
        // warp-shuffle comment in mult_thrust.h. TPB does use it
        // (tpb/sbdiag.h:888-900), which is why it stays on TPB_SBD. An
        // attribute that silently does nothing is worse than an absent one,
        // and determinant placement is controlled by
        // gdb_diag's determinant_distribution argument instead.
        .def_readwrite("do_rdm", &sbd::gdb::SBD::do_rdm,
                      "Calculate RDM flag (0=density only, 1=full RDM)")
        .def_readwrite("carryover_type", &sbd::gdb::SBD::carryover_type,
                      "Carryover determinant selection type (0=off, 1=weight truncation, "
                      "2/3=heatbath expansion)")
        .def_readwrite("ratio", &sbd::gdb::SBD::ratio,
                      "Carryover ratio")
        .def_readwrite("threshold", &sbd::gdb::SBD::threshold,
                      "Carryover threshold")
        .def_readwrite("heatbath_cutoff", &sbd::gdb::SBD::heatbath_cutoff,
                      "Heatbath expansion cutoff")
        .def_readwrite("heatbath_truncation", &sbd::gdb::SBD::heatbath_truncation,
                      "Weight truncation threshold applied before heatbath expansion")
        .def_readwrite("heatbath_batch_size", &sbd::gdb::SBD::heatbath_batch_size,
                      "Heatbath expansion batch size")
#ifdef SBD_THRUST
        .def_readwrite("cpu_subspace", &sbd::gdb::SBD::cpu_subspace,
                      "gpu (Thrust) backend only: keep the Davidson subspace vectors in "
                      "host memory instead of GPU memory, so the GPU memory they use no "
                      "longer grows with max_nb; H is still applied on the GPU. Energies "
                      "agree with the default to within rounding")
#endif
        .def_readwrite("bit_length", &sbd::gdb::SBD::bit_length,
                      "Bit length for determinant representation")
        ;

    // ========================================================================
    // Utility functions
    // ========================================================================
    
    m.def("LoadFCIDump", &sbd::LoadFCIDump,
          "Load FCIDUMP file and return FCIDump object",
          py::arg("filename"));

    m.def("LoadAlphaDets",
          [](const std::string& filename, size_t bit_length, size_t total_bit_length) {
              // Upstream (r-ccs-cms/sbd PR#71) migrated alpha-det containers to
              // det_vector<size_t, det_kind::half>; unpack to lists for Python.
              sbd::det_vector<size_t, sbd::det_kind::half> dets;
              sbd::LoadAlphaDets(filename, dets, bit_length, total_bit_length);
              std::vector<std::vector<size_t>> out;
              for (const auto& r : dets) out.emplace_back(r.begin(), r.end());
              return out;
          },
          "Load alpha determinants from file",
          py::arg("filename"),
          py::arg("bit_length"),
          py::arg("total_bit_length"));

    // Upstream 93ebabe made makestring a template (const DetT&), so its address
    // is no longer a single function pointer. Instantiate for the type the
    // Python API passes -- a list of ints -- which keeps the signature as it was.
    m.def("makestring", &sbd::makestring<std::vector<size_t>>,
          "Convert bitstring to string representation",
          py::arg("config"),
          py::arg("bit_length"),
          py::arg("total_bit_length"));

    m.def("from_string", &sbd::from_string,
          "Convert binary string to determinant format",
          py::arg("s"),
          py::arg("bit_length"),
          py::arg("total_bit_length"));

    // Packing one determinant per call from Python is the bottleneck at scale:
    // issue #31 measures it at tens of thousands of strings, and a sampled
    // subspace is far larger. Loop in C++ and hand back one array instead.
    m.def("from_strings",
          [](const std::vector<std::string>& strings, size_t bit_length,
             size_t total_bit_length) {
              const size_t words =
                  (total_bit_length + bit_length - 1) / bit_length;
              const std::vector<py::ssize_t> shape{
                  static_cast<py::ssize_t>(strings.size()),
                  static_cast<py::ssize_t>(words)};
              py::array_t<size_t> out(shape);
              size_t* data = out.mutable_data();
              std::memset(data, 0, strings.size() * words * sizeof(size_t));
              for (size_t r = 0; r < strings.size(); ++r) {
                  const std::string& s = strings[r];
                  if (s.size() != total_bit_length) {
                      throw std::invalid_argument(
                          "from_strings: expected every bitstring to be "
                          + std::to_string(total_bit_length) + " characters, got "
                          + std::to_string(s.size()) + " at index "
                          + std::to_string(r));
                  }
                  size_t* row = data + r * words;
                  for (size_t i = 0; i < total_bit_length; ++i) {
                      if (s[total_bit_length - 1 - i] == '1') {
                          row[i / bit_length] |=
                              (static_cast<size_t>(1) << (i % bit_length));
                      }
                  }
              }
              return out;
          },
          "Pack many bitstrings into one (n, words) array, as from_string does "
          "for a single determinant",
          py::arg("strings"),
          py::arg("bit_length"),
          py::arg("total_bit_length"));

    // Array in, array out, so a sharded determinant list can be put in canonical
    // order without a round trip through Python lists. Deduplicates locally, as
    // sbd::sort_bitarray does.
    m.def("sort_bitarray_array",
          [](py::array_t<size_t, py::array::c_style | py::array::forcecast> arr)
                 -> py::array_t<size_t> {
              if (arr.ndim() != 2) {
                  // An empty shard is legal and carries no width.
                  if (arr.ndim() <= 1 && arr.size() == 0) {
                      const std::vector<py::ssize_t> empty_shape{
                          static_cast<py::ssize_t>(0),
                          static_cast<py::ssize_t>(0)};
                      return py::array_t<size_t>(empty_shape);
                  }
                  throw std::invalid_argument(
                      "sort_bitarray_array expects a 2-D (ndets, words) array");
              }
              const size_t n = static_cast<size_t>(arr.shape(0));
              const size_t words = static_cast<size_t>(arr.shape(1));
              std::vector<std::vector<size_t>> rows(n, std::vector<size_t>(words));
              const size_t* in = arr.data();
              for (size_t r = 0; r < n; ++r) {
                  std::memcpy(rows[r].data(), in + r * words,
                              words * sizeof(size_t));
              }
              sbd::sort_bitarray(rows);
              const std::vector<py::ssize_t> shape{
                  static_cast<py::ssize_t>(rows.size()),
                  static_cast<py::ssize_t>(words)};
              py::array_t<size_t> out(shape);
              size_t* data = out.mutable_data();
              for (size_t r = 0; r < rows.size(); ++r) {
                  std::memcpy(data + r * words, rows[r].data(),
                              words * sizeof(size_t));
              }
              return out;
          },
          "Sort packed determinants into canonical order, removing duplicates",
          py::arg("dets"));

    m.def("sort_bitarray",
          [](std::vector<std::vector<size_t>>& dets) {
              sbd::sort_bitarray(dets);
              return dets;
          },
          "Sort determinant array in canonical order (required before diag)",
          py::arg("dets"));

    // ========================================================================
    // Main TPB diagonalization function (data structure version)
    // ========================================================================
    
    m.def("tpb_diag",
        [](py::object py_comm,
           const sbd::tpb::SBD& sbd_data,
           const sbd::FCIDump& fcidump,
           const std::vector<std::vector<size_t>>& adet,
           const std::vector<std::vector<size_t>>& bdet,
           const std::string& loadname,
           const std::string& savename) {
            
            // Convert MPI communicator
            MPI_Comm comm = get_mpi_comm(py_comm);
            sbd_require_openmp_threads("tpb_diag", comm);
            
            // Get MPI rank for GPU assignment
            int mpi_rank;
            MPI_Comm_rank(comm, &mpi_rank);

            // An explicit start must be in the lists. Upstream checks it locally deep
            // inside diag, after its setup collectives, and throws on the ranks that
            // fail; that is unanimous today only because every rank holds the same
            // lists. Check it here first, on every rank, with upstream's own packing,
            // and raise on all of them together. GDB needs no such check: its upstream
            // lookup (BasisInitVectorFromDeterminant) reduces over b_comm before it
            // throws, so every rank already raises together there.
            if (loadname.empty() && !sbd_data.initial_adeterminant_bitstring.empty()) {
                std::string problem;
                try {
                    size_t norb = 0;
                    for (const auto& kv : fcidump.header) {
                        if (kv.first == std::string("NORB")) {
                            norb = static_cast<size_t>(std::atoi(kv.second.c_str()));
                        }
                    }
                    if (norb == 0) {
                        throw std::invalid_argument(
                            "tpb_diag could not read NORB from the FCIDUMP header");
                    }
                    const std::string& b_string =
                        sbd_data.initial_bdeterminant_bitstring.empty()
                            ? sbd_data.initial_adeterminant_bitstring
                            : sbd_data.initial_bdeterminant_bitstring;
                    const auto want_a = sbd::from_string_checked(
                        sbd_data.initial_adeterminant_bitstring, sbd_data.bit_length, norb);
                    const auto want_b = sbd::from_string_checked(
                        b_string, sbd_data.bit_length, norb);
                    const auto na = std::count(adet.begin(), adet.end(), want_a);
                    const auto nb = std::count(bdet.begin(), bdet.end(), want_b);
                    const char* spin = na != 1 ? "alpha" : (nb != 1 ? "beta" : nullptr);
                    if (spin) {
                        problem = std::string("initial ") + spin + " determinant "
                            + ((na != 1 ? na : nb) == 0 ? "is not present in"
                                                        : "occurs more than once in")
                            + " the basis";
                    }
                } catch (const std::exception& e) {
                    problem = e.what();
                }
                if (!sbd_all_ranks_ok(problem.empty(), comm)) {
                    throw std::invalid_argument(problem.empty()
                        ? "the initial determinant is invalid on another rank"
                        : problem);
                }
            }
            
#ifdef SBD_THRUST
            // Assign GPU device based on MPI rank
            int numDevices, myDevice;
#ifdef __CUDACC__
            cudaGetDeviceCount(&numDevices);
            myDevice = mpi_rank % numDevices;
            cudaSetDevice(myDevice);
#else
            hipGetDeviceCount(&numDevices);
            myDevice = mpi_rank % numDevices;
            hipSetDevice(myDevice);
#endif
#endif
#ifdef USE_OMP_OFFLOAD
            // Assign OMP-offload device based on MPI rank.
            sbd_pin_offload_device(mpi_rank);
#endif
            
            // Output variables. Since upstream PR#71 the TPB det lists are
            // det_vector<size_t, det_kind::half>; pack/unpack to lists here.
            double energy;
            std::vector<double> density;
            sbd::det_vector<size_t, sbd::det_kind::half> co_adet;
            sbd::det_vector<size_t, sbd::det_kind::half> co_bdet;
            std::vector<std::vector<size_t>> co_adet_vvs;
            std::vector<std::vector<size_t>> co_bdet_vvs;
            std::vector<std::vector<double>> one_p_rdm;
            std::vector<std::vector<double>> two_p_rdm;

            // Release GIL for long computation
            py::gil_scoped_release release;

            // Pack the Python-provided det lists into det_vector<...half>.
            sbd::det_vector<size_t, sbd::det_kind::half> adet_p(adet.begin(), adet.end());
            sbd::det_vector<size_t, sbd::det_kind::half> bdet_p(bdet.begin(), bdet.end());

            // Call C++ function
            sbd::tpb::diag(comm, sbd_data, fcidump, adet_p, bdet_p,
                          loadname, savename, energy, density,
                          co_adet, co_bdet, one_p_rdm, two_p_rdm);

            // Unpack carryover det_vectors back to lists of lists.
            for (const auto& r : co_adet) co_adet_vvs.emplace_back(r.begin(), r.end());
            for (const auto& r : co_bdet) co_bdet_vvs.emplace_back(r.begin(), r.end());

            // Reacquire GIL for Python object creation
            py::gil_scoped_acquire acquire;

            // Return results as dictionary
            py::dict results;
            results["energy"] = energy;
            results["density"] = density;
            results["carryover_adet"] = co_adet_vvs;
            results["carryover_bdet"] = co_bdet_vvs;
            results["one_p_rdm"] = one_p_rdm;
            results["two_p_rdm"] = two_p_rdm;
            
            return results;
        },
        "Perform TPB diagonalization with pre-loaded data structures",
        py::arg("comm"),
        py::arg("sbd_data"),
        py::arg("fcidump"),
        py::arg("adet"),
        py::arg("bdet"),
        py::arg("loadname") = "",
        py::arg("savename") = "");

    // ========================================================================
    // Main GDB diagonalization function (data structure version)
    //
    // The determinant list IS the subspace: unlike TPB, it is not the Cartesian
    // product of two half-determinant lists, so an arbitrary sparse set can be
    // diagonalized.
    //
    // THE SHARD CONTRACT. The list may be distributed:
    //
    //   b_comm_size == 1  -> every rank passes the WHOLE basis.
    //   b_comm_size >  1  -> every rank passes ITS OWN SHARD, and the union over
    //                        b_comm positions is the basis.
    //
    // Sharding is what makes GDB's memory scale: b_comm is the only dimension
    // that divides the basis (and hence the excitation lookup), so at
    // b_comm_size == 1 every rank stores everything no matter how many ranks
    // there are. Ranks sharing a b_comm position -- i.e. one h_comm, which is
    // world_rank % b_comm_size -- must pass IDENTICAL shards, because the
    // in-memory gdb::diag never broadcasts the list the way the file-based
    // overload does (gdb/sbdiag.h:764). That is checked, not trusted.
    //
    // Input shards must be globally sorted and disjoint. This is stricter than
    // upstream (grid-cyclic tolerates arbitrary shards), but it is what slicing a
    // globally sorted list gives you for free, and it lets a caller's partition
    // be checked rather than silently producing a wrong subspace. Completeness --
    // that the union is the basis you MEANT -- cannot be checked here, so the
    // global dimension is returned for the caller to assert on.
    // ========================================================================

    m.def("gdb_diag",
        [](py::object py_comm,
           const sbd::gdb::SBD& sbd_data,
           const sbd::FCIDump& fcidump,
           py::array_t<size_t, py::array::c_style | py::array::forcecast> det_in,
           const std::string& loadname,
           const std::string& savename,
           const std::string& determinant_distribution,
           int determinant_grid_a,
           int determinant_grid_b) {

#ifdef USE_OMP_OFFLOAD
            // Upstream has no OpenMP-offload GDB code -- there is no `omp target`
            // anywhere under include/sbd/chemistry/gdb/ -- so this build would run
            // GDB entirely on the host while the caller asked for a GPU. Refuse
            // rather than do that. Every rank loads the same module, so every rank
            // raises here and none is left waiting in a collective.
            throw std::invalid_argument(
                "gdb_diag is not supported on the OpenMP-offload backend ('gpu-omp'): "
                "upstream SBD has no OpenMP-offload GDB kernels, so it would run on "
                "the CPU. Use device='cpu', or device='gpu' (NVHPC Thrust, NVIDIA GPUs "
                "only; there is no GDB GPU path for AMD GPUs).");
#endif

            MPI_Comm comm = get_mpi_comm(py_comm);
            sbd_require_openmp_threads("gdb_diag", comm);
            int mpi_rank; MPI_Comm_rank(comm, &mpi_rank);
            int mpi_size; MPI_Comm_size(comm, &mpi_size);

            const int b_comm_size = sbd_data.b_comm_size;
            const int t_comm_size = sbd_data.t_comm_size;

            // ---- grid shape -------------------------------------------------
            // These depend only on the config, which every rank passes
            // identically, so they can throw directly without a vote.
            if (b_comm_size < 1 || t_comm_size < 1) {
                throw std::invalid_argument(
                    "gdb_diag requires b_comm_size >= 1 and t_comm_size >= 1");
            }
            // GDB runs one task per basis-ring station and there are exactly
            // b_comm_size of them, so a larger t_comm_size starves a rank and
            // upstream's MakeHelpers then dereferences an empty lookup
            // (gdb/helper.h:736-739 then :761) -- a segfault, not an error.
            if (t_comm_size > b_comm_size) {
                throw std::invalid_argument(
                    "gdb_diag requires t_comm_size <= b_comm_size: GDB runs one task "
                    "per basis-ring station and there are exactly b_comm_size of them, "
                    "so a larger t_comm_size starves a rank and upstream's MakeHelpers "
                    "then dereferences an empty lookup (segfault, not an error)");
            }
            // gdb::diag dispatches with `if (method == 0) {...} else if (method == 1)
            // {...}` and no else, assigning `energy` only inside those branches
            // (gdb/sbdiag.h:418, :501). A method of 2 or 3 -- valid for TPB, where
            // they select Lanczos -- runs no diagonalization at all and leaves
            // `energy` uninitialized. The Thrust build masks the value first
            // (`method &= 1`, sbdiag.h:210-211), so it is the CPU and OMP-offload
            // backends that would return garbage. Reject it on every backend.
            if (sbd_data.method != 0 && sbd_data.method != 1) {
                throw std::invalid_argument(
                    "gdb_diag requires method 0 or 1 (Davidson, or Davidson storing "
                    "the Hamiltonian); GDB has no Lanczos, so TPB's methods 2 and 3 "
                    "do not exist here and would return an uninitialized energy");
            }
            const long long named_grid =
                static_cast<long long>(b_comm_size) * static_cast<long long>(t_comm_size);
            // h_comm_size is derived by integer division upstream with no check,
            // so a rank count that is not a multiple of b*t silently yields
            // ragged communicators: unequal h_comm sizes in one run, and a rank
            // alone in its own b_comm believing the ring has one station.
            if (named_grid > mpi_size || mpi_size % named_grid != 0) {
                throw std::invalid_argument(
                    "gdb_diag requires t_comm_size * b_comm_size to divide the rank "
                    "count exactly (the helper dimension is the quotient); got "
                    + std::to_string(t_comm_size) + " * " + std::to_string(b_comm_size)
                    + " against " + std::to_string(mpi_size) + " ranks");
            }
            const int h_comm_size = static_cast<int>(mpi_size / named_grid);
#ifdef SBD_THRUST
            // gdb/mult_thrust.h:310-314 throws on every kernel launch unless the
            // helper dimension is trivial. Say so before any work happens.
            if (h_comm_size != 1) {
                throw std::invalid_argument(
                    "GDB on the Thrust backend requires h_comm_size == 1, i.e. "
                    "t_comm_size * b_comm_size == ranks; got a helper dimension of "
                    + std::to_string(h_comm_size) + ". Spend every rank on "
                    "b_comm_size (and t_comm_size <= b_comm_size) instead");
            }
#endif

            // ---- placement scheme -------------------------------------------
            const SbdDetDistribution distribution =
                sbd_resolve_det_distribution(determinant_distribution);
            const bool uses_grid =
                distribution == SbdDetDistribution::grid_cyclic ||
                distribution == SbdDetDistribution::grid_cyclic_balanced;
            int grid_a = determinant_grid_a;
            int grid_b = determinant_grid_b;
            if (uses_grid) {
                if (grid_a < 0 || grid_b < 0) {
                    throw std::invalid_argument(
                        "determinant grid dimensions must be positive");
                }
                if ((grid_a == 0) != (grid_b == 0)) {
                    throw std::invalid_argument(
                        "specify both determinant grid dimensions or neither");
                }
                if (grid_a == 0) {
                    // The factor pair of b_comm_size nearest square, as main.cc does.
                    grid_a = static_cast<int>(std::sqrt(static_cast<double>(b_comm_size)));
                    while (grid_a > 1 && b_comm_size % grid_a != 0) --grid_a;
                    if (grid_a < 1) grid_a = 1;
                    grid_b = b_comm_size / grid_a;
                }
                if (static_cast<long long>(grid_a) * grid_b != b_comm_size) {
                    throw std::invalid_argument(
                        "determinant grid dimensions must multiply to b_comm_size ("
                        + std::to_string(b_comm_size) + ")");
                }
            } else if (grid_a != 0 || grid_b != 0) {
                throw std::invalid_argument(
                    "determinant grid dimensions require a grid-cyclic distribution");
            }

            // ---- orbital count and packing width ----------------------------
            size_t norb = 0;
            for (const auto& kv : fcidump.header) {
                if (kv.first == std::string("NORB")) {
                    norb = static_cast<size_t>(std::atoi(kv.second.c_str()));
                }
            }
            if (norb == 0) {
                throw std::invalid_argument(
                    "gdb_diag could not read NORB from the FCIDUMP header");
            }
            const size_t total_bits = 2 * norb;
            const size_t bit_length = sbd_data.bit_length;
            if (bit_length == 0) {
                throw std::invalid_argument("gdb_diag requires bit_length >= 1");
            }
            const size_t expected_words = (total_bits + bit_length - 1) / bit_length;

            // ---- the determinant buffer -------------------------------------
            // An empty shard is legal (a rank may own nothing), and numpy gives a
            // 1-D zero-length array for an empty list, which carries no width --
            // hence the agreement step below rather than trusting shape(1).
            const py::ssize_t ndim = det_in.ndim();
            size_t n_local = 0;
            size_t local_words = 0;
            bool shape_ok = true;
            if (ndim == 2) {
                n_local = static_cast<size_t>(det_in.shape(0));
                local_words = static_cast<size_t>(det_in.shape(1));
                if (n_local > 0 && local_words == 0) shape_ok = false;
            } else if (ndim <= 1 && det_in.size() == 0) {
                n_local = 0;
                local_words = 0;
            } else {
                shape_ok = false;
            }
            if (!sbd_all_ranks_ok(shape_ok, comm)) {
                throw std::invalid_argument(
                    "gdb_diag expects det as a 2-D (ndets, words) array of packed "
                    "determinants, as from_strings() returns; an empty shard may be "
                    "an empty array");
            }

            unsigned long long words_local = static_cast<unsigned long long>(local_words);
            unsigned long long words_max = 0;
            MPI_Allreduce(&words_local, &words_max, 1, MPI_UNSIGNED_LONG_LONG,
                          MPI_MAX, comm);
            if (words_max == 0) {
                throw std::invalid_argument(
                    "gdb_diag requires at least one determinant somewhere on the "
                    "communicator");
            }
            const size_t words = static_cast<size_t>(words_max);
            if (!sbd_all_ranks_ok(local_words == 0 || local_words == words, comm)) {
                throw std::invalid_argument(
                    "gdb_diag requires every rank's determinants to have the same "
                    "number of packed words");
            }
            if (words != expected_words) {
                throw std::invalid_argument(
                    "gdb_diag got determinants packed into " + std::to_string(words)
                    + " word(s), but NORB=" + std::to_string(norb) + " with bit_length="
                    + std::to_string(bit_length) + " implies "
                    + std::to_string(expected_words)
                    + "; pack with the same bit_length the config carries");
            }

            // det_vector's row width is a process-global property fixed by the
            // first container built, so set it while the GIL is held and report a
            // mismatch before any work is done. The half-determinant width is set
            // too, matching the upstream app: the grid-cyclic path builds
            // half-determinant containers before any diagonalization runs.
            try {
                sbd::det_vector<size_t>::init_elem_size(words);
                sbd::det_vector<size_t, sbd::det_kind::half>::init_elem_size(
                    (norb + bit_length - 1) / bit_length);
            } catch (const std::length_error&) {
                throw std::invalid_argument(
                    "gdb_diag was already called in this process with a different "
                    "number of words per determinant, which SBD fixes for the "
                    "lifetime of the process. Keep norb and bit_length fixed, or "
                    "run the new problem in a fresh process.");
            }

            sbd::det_vector<size_t> det;
            det.resize(n_local);
            if (n_local > 0) {
                std::memcpy(det.flat().data(), det_in.data(),
                            n_local * words * sizeof(size_t));
            }

            // SBD indexes the subspace with binary searches, so each shard must be
            // in canonical order; an unsorted list silently yields a wrong energy.
            // sort_bitarray also removes duplicates, which would leave part of the
            // subspace unreachable, so reject a shrink rather than dropping them.
            const size_t n_before = det.size();
            sbd::sort_bitarray(det);
            if (!sbd_all_ranks_ok(det.size() == n_before, comm)) {
                throw std::invalid_argument(
                    "gdb_diag requires distinct determinants");
            }

            // ---- communicators ----------------------------------------------
            // Built here, before diag builds its own set, because the placement
            // schemes redistribute over b_comm. With both named dimensions
            // trivial, h_comm is the whole communicator, so skip the split.
            const bool split_comms = (b_comm_size > 1 || t_comm_size > 1);
            MPI_Comm h_comm = comm, b_comm = MPI_COMM_NULL, t_comm = MPI_COMM_NULL;
            if (split_comms) {
                sbd::gdb::DetBasisCommunicator(comm, h_comm_size, b_comm_size,
                                               t_comm_size, h_comm, b_comm, t_comm);
            }
            struct CommGuard {
                bool owns; MPI_Comm *h, *b, *t;
                ~CommGuard() {
                    if (!owns) return;
                    if (*h != MPI_COMM_NULL) MPI_Comm_free(h);
                    if (*b != MPI_COMM_NULL) MPI_Comm_free(b);
                    if (*t != MPI_COMM_NULL) MPI_Comm_free(t);
                }
            } comm_guard{split_comms, &h_comm, &b_comm, &t_comm};

            // ---- shard agreement across h_comm ------------------------------
            {
                int h_size = 1;
                MPI_Comm_size(h_comm, &h_size);
                if (h_size > 1) {
                    unsigned long long fp = sbd_shard_fingerprint(det);
                    unsigned long long lo = 0, hi = 0;
                    MPI_Allreduce(&fp, &lo, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, h_comm);
                    MPI_Allreduce(&fp, &hi, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, h_comm);
                    if (!sbd_all_ranks_ok(lo == hi, comm)) {
                        throw std::invalid_argument(
                            "gdb_diag requires ranks sharing a b_comm position to pass "
                            "identical determinants: the shard index is "
                            "world_rank % b_comm_size, and the in-memory path does not "
                            "broadcast the list, so a divergence would silently "
                            "diagonalize different subspaces");
                    }
                }
            }

            // ---- shards are globally sorted and disjoint ---------------------
            if (b_comm_size > 1) {
                int rank_b = 0, size_b = 0;
                MPI_Comm_rank(b_comm, &rank_b);
                MPI_Comm_size(b_comm, &size_b);
                std::vector<size_t> my_last(words, 0), prev_last(words, 0);
                const bool have_mine = det.size() > 0;
                if (have_mine) {
                    std::memcpy(my_last.data(), &det.cflat()[(det.size() - 1) * words],
                                words * sizeof(size_t));
                }
                int send_n = have_mine ? 1 : 0;
                int recv_n = 0;
                const int dst = (rank_b + 1 < size_b) ? rank_b + 1 : MPI_PROC_NULL;
                const int srcr = (rank_b > 0) ? rank_b - 1 : MPI_PROC_NULL;
                MPI_Sendrecv(&send_n, 1, MPI_INT, dst, 91,
                             &recv_n, 1, MPI_INT, srcr, 91, b_comm, MPI_STATUS_IGNORE);
                MPI_Sendrecv(my_last.data(), static_cast<int>(words),
                             SBD_MPI_SIZE_T, dst, 92,
                             prev_last.data(), static_cast<int>(words),
                             SBD_MPI_SIZE_T, srcr, 92, b_comm, MPI_STATUS_IGNORE);
                // Only comparable when both sides hold something; an empty shard
                // in between is skipped rather than treated as a failure.
                bool ordered = true;
                if (recv_n == 1 && have_mine) {
                    std::vector<size_t> my_first(words, 0);
                    std::memcpy(my_first.data(), det.cflat().data(),
                                words * sizeof(size_t));
                    const bool duplicate =
                        std::memcmp(my_first.data(), prev_last.data(),
                                    words * sizeof(size_t)) == 0;
                    ordered = !duplicate && !sbd::less_from_back(my_first, prev_last);
                }
                if (!sbd_all_ranks_ok(ordered, comm)) {
                    throw std::invalid_argument(
                        "gdb_diag requires the input shards to be globally sorted and "
                        "disjoint: shard i must hold a strictly lower range than shard "
                        "i+1, where the shard index is world_rank % b_comm_size. Slice "
                        "a globally sorted determinant list to get this");
                }
            }

            // ---- schemes that cannot take an empty shard --------------------
            // redistribution/reordering index config[0] unconditionally
            // (framework/bit_manipulation.h:566, :577) and SaveWavefunction does
            // basis[0].size() (caop/basic/restart.h:35), all UB when a rank owns
            // nothing. redistribution_bitarray and the grid-cyclic path are safe.
            const bool needs_nonempty =
                distribution == SbdDetDistribution::count ||
                distribution == SbdDetDistribution::count_sorted ||
                !savename.empty();
            if (needs_nonempty && !sbd_all_ranks_ok(det.size() > 0, comm)) {
                throw std::invalid_argument(
                    std::string("gdb_diag with determinant_distribution='")
                    + sbd_det_distribution_name(distribution)
                    + "'" + (savename.empty() ? "" : " or a savename")
                    + " requires a non-empty shard on every rank: upstream indexes the "
                      "first determinant unconditionally on those paths");
            }

            // ---- redistribute over b_comm -----------------------------------
            // Nothing to place with a single basis block, and equal-bra-a there
            // would still pay an allgather of every alpha string, so skip it.
            if (b_comm_size > 1 && distribution != SbdDetDistribution::input) {
                py::gil_scoped_release release;
                switch (distribution) {
                    case SbdDetDistribution::equal_bra_a:
                        sbd::redistribution_equal_bra_a(det, bit_length, total_bits,
                                                        b_comm);
                        break;
                    case SbdDetDistribution::count:
                        sbd::redistribution(det, bit_length, total_bits, b_comm);
                        break;
                    case SbdDetDistribution::count_sorted:
                        sbd::redistribution(det, bit_length, total_bits, b_comm);
                        sbd::reordering(det, bit_length, total_bits, b_comm);
                        break;
                    case SbdDetDistribution::grid_cyclic:
                    case SbdDetDistribution::grid_cyclic_balanced:
                        sbd::gdb::redistribution_grid_bra_ab_cyclic(
                            det, bit_length, total_bits,
                            static_cast<size_t>(grid_a), static_cast<size_t>(grid_b),
                            b_comm);
                        if (distribution == SbdDetDistribution::grid_cyclic_balanced) {
                            sbd::redistribution_bitarray(det, b_comm);
                        }
                        // Mandatory: the grid-cyclic output is locally unsorted,
                        // and MakeHelpers requires each shard in canonical order.
                        sbd::sort_bitarray(det);
                        break;
                    case SbdDetDistribution::input:
                        break;
                }
            }

#ifdef SBD_THRUST
            // ---- Thrust: every b shard must hold something, after placement -------
            // Upstream's Thrust mult never returns when a b shard is empty: the ring
            // exchange (MpiSlider::Sync, gdb/helper_thrust.h) waits forever, while the
            // CPU backend handles the same input. A seed smaller than b_comm_size (one
            // Hartree-Fock determinant, say) is the usual way to get there, and
            // placement can empty a shard the input filled: equal-bra-a deals out
            // whole alpha strings, so fewer distinct alpha strings than b_comm_size
            // leaves ranks with nothing.
            if (b_comm_size > 1 && !sbd_all_ranks_ok(det.size() > 0, comm)) {
                throw std::invalid_argument(
                    std::string("gdb_diag on the gpu (Thrust) backend needs determinants "
                                "on every b_comm shard after placement (determinant_"
                                "distribution='")
                    + sbd_det_distribution_name(distribution)
                    + "'), but at least one is empty: upstream's Thrust solve then waits "
                      "forever. A subspace smaller than b_comm_size, such as a single "
                      "starting determinant, cannot fill every shard. Use fewer shards "
                      "(a smaller b_comm_size), determinant_distribution='input' with a "
                      "non-empty slice per rank, or device='cpu'");
            }
#endif

            // ---- global dimension -------------------------------------------
            // Summed over b_comm, since h_comm and t_comm hold replicas.
            unsigned long long local_dim = static_cast<unsigned long long>(det.size());
            unsigned long long global_dim = local_dim;
            if (b_comm_size > 1) {
                MPI_Allreduce(&local_dim, &global_dim, 1, MPI_UNSIGNED_LONG_LONG,
                              MPI_SUM, b_comm);
            }

#ifdef SBD_THRUST
            // Assign GPU device based on MPI rank
            int numDevices, myDevice;
#ifdef __CUDACC__
            cudaGetDeviceCount(&numDevices);
            myDevice = mpi_rank % numDevices;
            cudaSetDevice(myDevice);
#else
            hipGetDeviceCount(&numDevices);
            myDevice = mpi_rank % numDevices;
            hipSetDevice(myDevice);
#endif
#endif
#ifdef USE_OMP_OFFLOAD
            // Assign OMP-offload device based on MPI rank.
            sbd_pin_offload_device(mpi_rank);
#endif

            double energy = 0.0;
            std::vector<double> density;
            sbd::det_vector<size_t> co_det;
            std::vector<std::vector<double>> one_p_rdm;
            std::vector<std::vector<double>> two_p_rdm;

            {
                py::gil_scoped_release release;

                sbd::gdb::diag(comm, sbd_data, fcidump, det,
                               loadname, savename, energy, density,
                               co_det, one_p_rdm, two_p_rdm);

                // With do_rdm == 0 the occupation density is computed only where
                // mpi_rank_t == 0 (gdb/sbdiag.h:525) and left untouched
                // elsewhere, so ranks off that plane would see an empty list.
                // Fill them in so the Python contract does not depend on t.
                if (t_comm_size > 1 && t_comm != MPI_COMM_NULL) {
                    unsigned long long n_den = density.size(), n_den_max = 0;
                    MPI_Allreduce(&n_den, &n_den_max, 1, MPI_UNSIGNED_LONG_LONG,
                                  MPI_MAX, t_comm);
                    if (n_den_max > 0) {
                        density.resize(static_cast<size_t>(n_den_max), 0.0);
                        MPI_Bcast(density.data(), static_cast<int>(n_den_max),
                                  MPI_DOUBLE, 0, t_comm);
                    }
                }
            }

            // Carryover comes back as a shard, not a gathered list: for
            // carryover_type 1 it is split over b_comm AND duplicated across
            // h_comm (gdb/carryover.h), and for 2/3 it is split over the world
            // communicator with no duplication (gdb/expansion.h:821-822). Handing
            // back the raw shard keeps the memory win; the caller gathers if it
            // wants the whole list, taking one representative per b_comm position
            // for type 1.
            const size_t n_co = co_det.size();
            // Explicit shape vector: a braced list here is ambiguous under g++
            // (array_t's ShapeContainer constructor versus its copy/move), which
            // nvc++ and clang both accepted -- so the Linux CPU build broke while
            // the macOS and Thrust builds passed.
            const std::vector<py::ssize_t> co_shape{
                static_cast<py::ssize_t>(n_co),
                static_cast<py::ssize_t>(words)};
            py::array_t<size_t> co_out(co_shape);
            if (n_co > 0) {
                std::memcpy(co_out.mutable_data(), co_det.cflat().data(),
                            n_co * words * sizeof(size_t));
            }

            py::dict results;
            results["energy"] = energy;
            results["density"] = density;
            results["carryover_det"] = co_out;
            results["one_p_rdm"] = one_p_rdm;
            results["two_p_rdm"] = two_p_rdm;
            results["local_dim"] = local_dim;
            results["global_dim"] = global_dim;
            results["determinant_distribution"] =
                std::string(sbd_det_distribution_name(distribution));

            return results;
        },
        "Perform GDB diagonalization over an explicit list of determinants, which "
        "may be sharded across b_comm",
        py::arg("comm"),
        py::arg("sbd_data"),
        py::arg("fcidump"),
        py::arg("det"),
        py::arg("loadname") = "",
        py::arg("savename") = "",
        py::arg("determinant_distribution") = "",
        py::arg("determinant_grid_a") = 0,
        py::arg("determinant_grid_b") = 0);

    // ========================================================================
    // Stochastic external variance and Epstein-Nesbet PT2 (upstream's
    // apps/gdb-stat-evaluator, as a library call)
    //
    // Not in the Thrust build: upstream's gdb/inc_all.h includes the evaluator
    // only #ifndef SBD_THRUST. It always runs on CPU anyway, so sbd.analysis uses
    // a module that has it (the CPU or OpenMP-offload one), whatever the device
    // the state was diagonalized on.
    // ========================================================================
#ifndef SBD_THRUST
    m.attr("__sbd_has_evaluator__") = true;
    m.def("external_observables",
        [](py::object py_comm,
           const sbd::FCIDump& fcidump,
           py::array_t<size_t, py::array::c_style | py::array::forcecast> det_in,
           py::array_t<double, py::array::c_style | py::array::forcecast> coef_in,
           size_t bit_length,
           double reference_energy,
           size_t sample_count,
           size_t batch_count,
           uint64_t seed,
           uint64_t first_batch_id,
           double heatbath_cutoff,
           const std::string& observable,
           size_t expansion_batch_size,
           double minimum_abs_denominator) {

            MPI_Comm comm = get_mpi_comm(py_comm);
            int mpi_size = 0;
            MPI_Comm_size(comm, &mpi_size);

            sbd::stat_evaluator::Observable which;
            try {
                which = sbd::stat_evaluator::parse_observable(observable);
            } catch (const std::exception&) {
                throw std::invalid_argument(
                    "observable must be 'both', 'variance' or 'pt2', got '" + observable + "'");
            }
            if (sample_count < 2)
                throw std::invalid_argument("sample_count must be at least 2");
            if (batch_count < 1)
                throw std::invalid_argument("batch_count must be at least 1");
            if (bit_length < 1 || bit_length > 64)
                throw std::invalid_argument("bit_length must be in [1, 64]");
            if (!(heatbath_cutoff >= 0.0) || !std::isfinite(heatbath_cutoff))
                throw std::invalid_argument("heatbath_cutoff must be finite and >= 0");
            if (which != sbd::stat_evaluator::Observable::variance
                && !std::isfinite(reference_energy))
                throw std::invalid_argument("PT2 needs a finite reference_energy");

            // Every rank passes the same FCIDump; the integrals are replicated.
            int norb_i = 0, nelec = 0;
            double scalar_integral = 0.0;
            sbd::oneInt<double> one_integrals;
            sbd::twoInt<double> two_integrals;
            sbd::SetupIntegrals(fcidump, norb_i, nelec, scalar_integral,
                                one_integrals, two_integrals);
            if (norb_i <= 0)
                throw std::invalid_argument("FCIDUMP has no positive NORB");
            const size_t norb = static_cast<size_t>(norb_i);
            const size_t words = (2 * norb + bit_length - 1) / bit_length;

            if (det_in.ndim() != 2)
                throw std::invalid_argument("det must be an (n, words) array");
            if (coef_in.ndim() != 1)
                throw std::invalid_argument("coef must be a 1-D array");
            const size_t n_local = static_cast<size_t>(det_in.shape(0));
            const size_t local_words = n_local ? static_cast<size_t>(det_in.shape(1)) : words;
            if (!sbd_all_ranks_ok(static_cast<size_t>(coef_in.shape(0)) == n_local, comm))
                throw std::invalid_argument("det and coef have different lengths on some rank");
            if (!sbd_all_ranks_ok(local_words == words, comm))
                throw std::invalid_argument(
                    "determinants are packed into " + std::to_string(local_words)
                    + " word(s), but NORB=" + std::to_string(norb) + " with bit_length="
                    + std::to_string(bit_length) + " implies " + std::to_string(words));

            try {
                sbd::det_vector<size_t>::init_elem_size(words);
                sbd::det_vector<size_t, sbd::det_kind::half>::init_elem_size(
                    (norb + bit_length - 1) / bit_length);
            } catch (const std::length_error&) {
                throw std::invalid_argument(
                    "this process already used a different number of words per "
                    "determinant, which SBD fixes for the process lifetime. Keep norb "
                    "and bit_length fixed, or evaluate in a fresh process.");
            }

            sbd::det_vector<size_t> parents;
            parents.resize(n_local);
            if (n_local > 0)
                std::memcpy(parents.flat().data(), det_in.data(), n_local * words * sizeof(size_t));
            std::vector<double> coefficients(coef_in.data(), coef_in.data() + n_local);

            // The evaluator assumes a normalized state (upstream normalizes in
            // LoadWavefunction, which this path does not use), so normalize over
            // all ranks here.
            double local_norm2 = 0.0;
            for (double c : coefficients) local_norm2 += c * c;
            double norm2 = 0.0;
            MPI_Allreduce(&local_norm2, &norm2, 1, MPI_DOUBLE, MPI_SUM, comm);
            if (!(norm2 > 0.0) || !std::isfinite(norm2))
                throw std::invalid_argument("the state has zero or non-finite norm");
            const double inv_norm = 1.0 / std::sqrt(norm2);
            for (double& c : coefficients) c *= inv_norm;

            sbd::gdb::StatEvaluatorOptions<double> options;
            options.observable = which;
            options.bit_length = bit_length;
            options.norb = norb;
            options.sample_count = sample_count;
            options.heatbath_cutoff = heatbath_cutoff;
            options.reference_energy = reference_energy;
            options.base_seed = seed;
            options.expansion_batch_size = expansion_batch_size;
            options.minimum_abs_denominator = minimum_abs_denominator;

            std::vector<sbd::gdb::BatchRequest> batches(batch_count);
            for (size_t k = 0; k < batch_count; ++k) {
                batches[k].calculation_id = "sbd-python";
                batches[k].batch_id = first_batch_id + k;
            }

            std::vector<double> variance, pt2, seconds;
            std::vector<uint64_t> batch_ids;
            {
                py::gil_scoped_release release;
                sbd::gdb::evaluate_statistical_batches<double, double>(
                    parents, coefficients, scalar_integral, one_integrals, two_integrals,
                    options, batches, comm,
                    [&](const sbd::gdb::EvaluatedBatch& evaluated) {
                        const auto& r = evaluated.record;
                        batch_ids.push_back(r.batch_id);
                        variance.push_back(r.has_variance ? r.variance_estimate : NAN);
                        pt2.push_back(r.has_pt2 ? r.pt2_estimate : NAN);
                        seconds.push_back(r.elapsed_seconds);
                    });
            }

            py::dict out;
            out["variance"] = variance;
            out["pt2"] = pt2;
            out["batch_id"] = batch_ids;
            out["seconds"] = seconds;
            out["mpi_size"] = mpi_size;
            out["norm_before"] = std::sqrt(norm2);
            return out;
        },
        "Stochastic external variance and Epstein-Nesbet PT2 of a GDB state "
        "(upstream's statistical evaluator). Internal: use sbd.analysis.variance.",
        py::arg("comm"),
        py::arg("fcidump"),
        py::arg("det"),
        py::arg("coef"),
        py::arg("bit_length"),
        py::arg("reference_energy"),
        py::arg("sample_count") = 20000,
        py::arg("batch_count") = 10,
        py::arg("seed") = 0,
        py::arg("first_batch_id") = 0,
        py::arg("heatbath_cutoff") = 0.0,
        py::arg("observable") = "both",
        py::arg("expansion_batch_size") = 1000000,
        py::arg("minimum_abs_denominator") = 0.0);
#else
    m.attr("__sbd_has_evaluator__") = false;
#endif

    // ========================================================================
    // Main TPB diagonalization function (file-based version)
    // ========================================================================
    
    m.def("tpb_diag_from_files",
        [](py::object py_comm,
           const sbd::tpb::SBD& sbd_data,
           const std::string& fcidumpfile,
           const std::string& adetfile,
           const std::string& loadname,
           const std::string& savename) {
            
            // Convert MPI communicator
            MPI_Comm comm = get_mpi_comm(py_comm);
            sbd_require_openmp_threads("tpb_diag_from_files", comm);
            
            // Get MPI rank for GPU assignment
            int mpi_rank;
            MPI_Comm_rank(comm, &mpi_rank);
            
#ifdef SBD_THRUST
            // Assign GPU device based on MPI rank
            int numDevices, myDevice;
#ifdef __CUDACC__
            cudaGetDeviceCount(&numDevices);
            myDevice = mpi_rank % numDevices;
            cudaSetDevice(myDevice);
#else
            hipGetDeviceCount(&numDevices);
            myDevice = mpi_rank % numDevices;
            hipSetDevice(myDevice);
#endif
#endif
#ifdef USE_OMP_OFFLOAD
            // Assign OMP-offload device based on MPI rank.
            sbd_pin_offload_device(mpi_rank);
#endif
            
            // Output variables. co_adet/co_bdet are det_vector<...half> since
            // upstream PR#71; unpack to lists for Python.
            double energy;
            std::vector<double> density;
            sbd::det_vector<size_t, sbd::det_kind::half> co_adet;
            sbd::det_vector<size_t, sbd::det_kind::half> co_bdet;
            std::vector<std::vector<size_t>> co_adet_vvs;
            std::vector<std::vector<size_t>> co_bdet_vvs;
            std::vector<std::vector<double>> one_p_rdm;
            std::vector<std::vector<double>> two_p_rdm;

            // Release GIL for long computation
            py::gil_scoped_release release;

            // Call file-based C++ function
            sbd::tpb::diag(comm, sbd_data, fcidumpfile, adetfile,
                          loadname, savename, energy, density,
                          co_adet, co_bdet, one_p_rdm, two_p_rdm);

            // Unpack carryover det_vectors back to lists of lists.
            for (const auto& r : co_adet) co_adet_vvs.emplace_back(r.begin(), r.end());
            for (const auto& r : co_bdet) co_bdet_vvs.emplace_back(r.begin(), r.end());

            // Reacquire GIL for Python object creation
            py::gil_scoped_acquire acquire;

            // Return results as dictionary
            py::dict results;
            results["energy"] = energy;
            results["density"] = density;
            results["carryover_adet"] = co_adet_vvs;
            results["carryover_bdet"] = co_bdet_vvs;
            results["one_p_rdm"] = one_p_rdm;
            results["two_p_rdm"] = two_p_rdm;
            
            return results;
        },
        "Perform TPB diagonalization from files (convenience function)",
        py::arg("comm"),
        py::arg("sbd_data"),
        py::arg("fcidumpfile"),
        py::arg("adetfile"),
        py::arg("loadname") = "",
        py::arg("savename") = "");

    // ========================================================================
    // Cleanup/Finalization functions
    // ========================================================================
    
    m.def("planned_device_id",
        [](py::object py_comm) {
            MPI_Comm comm = get_mpi_comm(py_comm);
            int mpi_rank;
            MPI_Comm_rank(comm, &mpi_rank);
            // Same rank % count rule the diag entry points apply. Deliberately
            // a second copy of that one-liner rather than a refactor of the
            // existing selection sites -- tpb_diag, gdb_diag,
            // tpb_diag_from_files, and the offload pin all three share -- so
            // exposing the value cannot change how any existing path selects
            // its device. The copies must be kept in step, or this query
            // starts lying.
            const int n = sbd_device_count();
            return n > 0 ? mpi_rank % n : -1;
        },
        "Device index this rank will use for diagonalization, or -1 if there "
        "is none (CPU build, or no devices visible).");

    m.def("cleanup_device",
        []() {
#ifdef SBD_THRUST
            // Synchronize GPU device but do NOT reset
            // cudaDeviceReset() can interfere with CUDA-aware MPI (UCX)
            // which may still have active CUDA events/streams
#ifdef __CUDACC__
            cudaDeviceSynchronize();
            // Note: cudaDeviceReset() intentionally NOT called to avoid
            // conflicts with CUDA-aware MPI cleanup
#else
            hipDeviceSynchronize();
            // Note: hipDeviceReset() intentionally NOT called to avoid
            // conflicts with ROCm-aware MPI cleanup
#endif
#endif
        },
        "Synchronize GPU device (GPU backend only). "
        "Note: Does not call cudaDeviceReset() to avoid conflicts with CUDA-aware MPI. "
        "GPU resources are freed automatically when the process exits.");

    m.def("finalize_mpi",
        []() {
            // Check if MPI is initialized before finalizing
            int initialized, finalized;
            MPI_Initialized(&initialized);
            MPI_Finalized(&finalized);
            
            if (initialized && !finalized) {
                MPI_Finalize();
            }
        },
        "Finalize MPI. Only call this if you initialized MPI yourself. "
        "If using mpi4py, MPI finalization is handled automatically at exit.");

    // ========================================================================
    // Version information
    // ========================================================================
    
    // ========================================================================
    // CAOP: selected-basis diagonalization of a creation/annihilation-operator
    // Hamiltonian (upstream apps/caop_selected_basis_diagonalization)
    // ========================================================================
    //
    // Upstream's CAOP code has CPU and Thrust paths and no OpenMP-offload one, so the
    // offload module does not bind it; __sbd_has_caop__ says which modules do. The
    // library is templated on the scalar type, so both real and complex
    // Hamiltonians are bound here, chosen per call; upstream's app picks one at
    // compile time with -D_COMPLEX.
#ifndef USE_OMP_OFFLOAD
    m.attr("__sbd_has_caop__") = true;

    py::class_<sbd::caop::SBD>(m, "CAOP_SBD", py::module_local(),
                               "Configuration for CAOP diagonalization")
        .def(py::init<>())
        .def_readwrite("t_comm_size", &sbd::caop::SBD::t_comm_size,
                      "Task communicator size (column-wise Hamiltonian application)")
        .def_readwrite("b_comm_size", &sbd::caop::SBD::b_comm_size,
                      "Basis communicator size: how many shards the basis is split into")
        // h_comm_size is deliberately NOT exposed, as on TPB_SBD and GDB_SBD.
        // Upstream declares the field but never reads it: caop::diag derives
        //     h_comm_size = mpi_size / (t_comm_size * b_comm_size)
        // (caop/basic/sbdiag.h:263, :564). An attribute that silently does nothing
        // is worse than an absent one; change the rank count or the other two
        // sizes instead.
        .def_readwrite("method", &sbd::caop::SBD::method,
                      "0/1: Davidson, Hamiltonian on the fly / stored; "
                      "2/3: Lanczos, on the fly / stored")
        .def_readwrite("max_it", &sbd::caop::SBD::max_it, "Restart cycles")
        .def_readwrite("max_nb", &sbd::caop::SBD::max_nb,
                      "Largest reduced-space dimension before a restart")
        .def_readwrite("max_iv", &sbd::caop::SBD::max_iv,
                      "Initial vectors (initial subspace dimension) for Davidson")
        .def_readwrite("eps", &sbd::caop::SBD::eps, "Convergence threshold on the residual")
        .def_readwrite("init", &sbd::caop::SBD::init,
                      "Initial vector; only 0 (random) is supported upstream")
        .def_readwrite("initial_determinant_bitstring",
                      &sbd::caop::SBD::initial_determinant_bitstring,
                      "Start with unit weight on this basis bitstring (system_size "
                      "characters) instead of init. It must occur exactly once in the "
                      "basis. A loadname takes precedence")
        .def_readwrite("seed", &sbd::caop::SBD::seed, "Seed for the random initial vector")
        .def_readwrite("system_size", &sbd::caop::SBD::system_size,
                      "Number of modes, i.e. the length of each basis bitstring")
        .def_readwrite("bit_length", &sbd::caop::SBD::bit_length,
                      "Bits stored per 64-bit word of a packed bitstring")
        .def_readwrite("do_sort_basis", &sbd::caop::SBD::do_sort_basis,
                      "Sort the basis across ranks after loading (for overlapping files)")
        .def_readwrite("do_redist_basis", &sbd::caop::SBD::do_redist_basis,
                      "Redistribute the basis evenly over b_comm after loading")
        .def_readwrite("carryover_type", &sbd::caop::SBD::carryover_type,
                      "0: none; 1: keep the largest-weight fraction; 2: deterministic "
                      "heat-bath expansion")
        .def_readwrite("ratio", &sbd::caop::SBD::ratio,
                      "carryover_type 1: fraction of basis states kept")
        .def_readwrite("heatbath_cutoff", &sbd::caop::SBD::heatbath_cutoff,
                      "carryover_type 2: accept a term when |a_k c_j| exceeds this")
        .def_readwrite("heatbath_truncation", &sbd::caop::SBD::heatbath_truncation,
                      "carryover_type 2: drop a parent unless |c_j|^2 exceeds this")
        .def_readwrite("heatbath_batch_size", &sbd::caop::SBD::heatbath_batch_size,
                      "carryover_type 2: candidate batch size per rank")
        .def_readwrite("heatbath_parent_distribution",
                      &sbd::caop::SBD::heatbath_parent_distribution,
                      "carryover_type 2: 0 keeps the basis layout for expansion, 1 "
                      "deals parents out by weight")
        .def("__repr__", [](const sbd::caop::SBD& s) {
            return "CAOP_SBD(method=" + std::to_string(s.method)
                + ", system_size=" + std::to_string(s.system_size)
                + ", bit_length=" + std::to_string(s.bit_length)
                + ", b_comm_size=" + std::to_string(s.b_comm_size) + ")";
        });

    m.def("caop_diag_from_files",
        [](py::object py_comm,
           const sbd::caop::SBD& sbd_data,
           const std::string& hamfile,
           const std::vector<std::string>& basisfiles,
           const std::string& loadname,
           const std::string& savename,
           bool complex) {
            MPI_Comm comm = get_mpi_comm(py_comm);
            sbd_require_openmp_threads("caop_diag_from_files", comm);
            int mpi_rank;
            MPI_Comm_rank(comm, &mpi_rank);
            int mpi_size;
            MPI_Comm_size(comm, &mpi_size);

            // ---- what upstream does not check ---------------------------------
            // These depend only on the config and the rank count, which every rank
            // sees identically, so every rank throws together without a vote. Each
            // was observed upstream to return a wrong energy with exit 0, or hang.
            const int t_comm_size = sbd_data.t_comm_size;
            const int b_comm_size = sbd_data.b_comm_size;
            if (t_comm_size < 1 || b_comm_size < 1) {
                throw std::invalid_argument(
                    "caop_diag_from_files requires t_comm_size >= 1 and b_comm_size >= 1");
            }
            // caop::diag derives h_comm_size = ranks / (t * b) by integer division
            // (caop/basic/sbdiag.h:263) and setup_communicator then splits ranks into
            // groups of t * b (helper.h:22-27). With a remainder, the last group is
            // short, so ranks disagree on the grid (of 3 ranks with b = 2, one sees
            // b = 1): a wrong energy, with exit 0. With fewer ranks than t * b the
            // grid is consistent but smaller than asked (2 ranks, b = 4 run b = 2).
            if (mpi_size % (t_comm_size * b_comm_size) != 0) {
                throw std::invalid_argument(
                    "caop_diag_from_files: the rank count (" + std::to_string(mpi_size)
                    + ") must be a multiple of t_comm_size * b_comm_size ("
                    + std::to_string(t_comm_size) + " * " + std::to_string(b_comm_size)
                    + " = " + std::to_string(t_comm_size * b_comm_size)
                    + "). Upstream derives the helper size as ranks / (t * b) and "
                      "returns a wrong energy otherwise");
            }
            // caop::diag's dispatch is `if (method == 0 || method == 2) {...} else
            // {...}`, so any other value silently runs a stored-Hamiltonian method.
            if (sbd_data.method < 0 || sbd_data.method > 3) {
                throw std::invalid_argument(
                    "caop_diag_from_files: method must be 0-3 (0/1 Davidson, 2/3 "
                    "Lanczos; odd stores the Hamiltonian), got "
                    + std::to_string(sbd_data.method));
            }
            // Davidson (methods 0 and 1) starts from max_iv vectors and grows the
            // reduced space up to max_nb before a restart, so it needs room to grow:
            // max_iv == max_nb returns a wrong energy, and max_iv > max_nb or 0
            // hangs. Lanczos (methods 2 and 3) does not use max_iv.
            if ((sbd_data.method == 0 || sbd_data.method == 1)
                && (sbd_data.max_iv < 1 || sbd_data.max_iv >= sbd_data.max_nb)) {
                throw std::invalid_argument(
                    "caop_diag_from_files: Davidson (method 0 or 1) needs "
                    "1 <= max_iv < max_nb, got max_iv=" + std::to_string(sbd_data.max_iv)
                    + ", max_nb=" + std::to_string(sbd_data.max_nb)
                    + ". Upstream returns a wrong energy at max_iv == max_nb and hangs "
                      "above it or at 0");
            }
#ifdef SBD_THRUST
            // make_slide deals the b ring stations out over the t task ranks
            // (caop/basic/helper.h:41-55), so with t > b some ranks get none. The CPU
            // mult guards an empty slide (mult.h:19); the Thrust one returns early
            // (mult_thrust.h:583) and skips the Allreduce the other ranks wait in, so
            // the job hangs; the stored-Hamiltonian methods run the CPU mult. The
            // same t <= b holds for GDB on every backend.
            if ((sbd_data.method == 0 || sbd_data.method == 2) && t_comm_size > b_comm_size) {
                throw std::invalid_argument(
                    "caop_diag_from_files on the gpu (Thrust) backend with method "
                    + std::to_string(sbd_data.method) + " needs t_comm_size <= b_comm_size, "
                    "got t_comm_size=" + std::to_string(t_comm_size) + ", b_comm_size="
                    + std::to_string(b_comm_size) + ": a task rank with no basis-ring "
                    "station hangs upstream's GPU kernel. Raise b_comm_size, lower "
                    "t_comm_size, or use device='cpu'");
            }
#endif
#ifdef SBD_THRUST
            int numDevices, myDevice;
#ifdef __CUDACC__
            cudaGetDeviceCount(&numDevices);
            myDevice = mpi_rank % numDevices;
            cudaSetDevice(myDevice);
#else
            hipGetDeviceCount(&numDevices);
            myDevice = mpi_rank % numDevices;
            hipSetDevice(myDevice);
#endif
#endif
            if (basisfiles.empty()) {
                throw std::invalid_argument("caop_diag_from_files needs at least one basis file");
            }
            // Upstream reads the hamfile on h-rank 0 only and sends the terms on
            // (caop/basic/loadmodel.h:92-96), so a file it cannot read hangs the other
            // ranks; see caop_check_hamfile.
            // Upstream's solve does not check the terms' modes against system_size.
            caop_check_modes(hamfile,
                             caop_check_hamfile(hamfile, complex, comm,
                                                "caop_diag_from_files"),
                             sbd_data.system_size, "caop_diag_from_files");
            // As upstream's caop-spectrum app: the stored Hamiltonian (methods 1/3) keeps
            // per-thread slots that need a fixed OpenMP team (caop/basic/mkham.h).
            omp_set_dynamic(0);
            double energy = 0.0;
            sbd::det_vector<size_t> co_basis;
            {
                py::gil_scoped_release release;
                if (complex) {
                    sbd::caop::diag<std::complex<double>>(comm, sbd_data, hamfile, basisfiles,
                                                          loadname, savename, energy, co_basis);
                } else {
                    sbd::caop::diag<double>(comm, sbd_data, hamfile, basisfiles,
                                            loadname, savename, energy, co_basis);
                }
            }
            const size_t words = co_basis.empty() ? 0 : co_basis.elem_size();
            py::array_t<size_t> co_out(std::vector<py::ssize_t>{
                static_cast<py::ssize_t>(co_basis.size()), static_cast<py::ssize_t>(words)});
            if (!co_basis.empty()) {
                std::memcpy(co_out.mutable_data(), co_basis.cflat().data(),
                            co_basis.size() * words * sizeof(size_t));
            }
            py::dict results;
            results["energy"] = energy;
            results["carryover_basis"] = co_out;
            return results;
        },
        "CAOP diagonalization of a Hamiltonian file over a basis given as files, "
        "as upstream's caop_selected_basis_diagonalization app",
        py::arg("comm"),
        py::arg("sbd_data"),
        py::arg("hamfile"),
        py::arg("basisfiles"),
        py::arg("loadname") = "",
        py::arg("savename") = "",
        py::arg("complex") = false);

    // CAOP spectrum: a Green's function of a saved CAOP ground state by block Lanczos
    // (upstream apps/caop-spectrum). Upstream's pipeline, run in one call: read the
    // checkpoint and the extra basis, load H, build the response coefficients with
    // sbd::caop::spectrum::spectrum, then evaluate G(omega) on a grid.
    m.def("caop_spectrum",
        [](py::object py_comm,
           const std::string& loadname,
           size_t shards,
           const std::string& hamfile,
           const std::vector<size_t>& orbitals,
           bool addition,
           size_t sites,
           size_t bit_length,
           double reference_energy,
           const std::vector<std::string>& extra_basis,
           const std::vector<double>& omegas,
           double eta,
           size_t max_steps,
           size_t h_comm_size,
           size_t t_comm_size,
           bool store_hamiltonian,
           bool complex,
           bool wavefunction_complex,
           const std::string& save_coefficients,
           const std::vector<std::string>& operator_files,
           double rank_atol,
           double rank_rtol) {
            namespace cs = sbd::caop::spectrum;
            const bool general = !operator_files.empty();
            if (wavefunction_complex && !complex) {
                throw std::invalid_argument(
                    "caop_spectrum: a complex saved state needs complex=True");
            }
            MPI_Comm comm = get_mpi_comm(py_comm);
            sbd_require_openmp_threads("caop_spectrum", comm);
            int world = 1, rank = 0;
            MPI_Comm_size(comm, &world);
            MPI_Comm_rank(comm, &rank);
            if (h_comm_size == 0 || t_comm_size == 0 || world % (h_comm_size * t_comm_size)) {
                throw std::invalid_argument(
                    "caop_spectrum: h_comm_size * t_comm_size must divide the rank count");
            }
            if (general && !orbitals.empty()) {
                throw std::invalid_argument(
                    "caop_spectrum: give orbitals (single-particle c/cdag) or operator "
                    "files (general operators), not both");
            }
            if (!general && orbitals.empty()) {
                throw std::invalid_argument(
                    "caop_spectrum needs at least one orbital or operator file");
            }
            if (!(rank_atol >= 0) || !(rank_rtol >= 0)) {
                throw std::invalid_argument(
                    "caop_spectrum: rank_atol and rank_rtol must be non-negative");
            }
            if (sites == 0 || bit_length == 0 || bit_length > 8 * sizeof(size_t)) {
                throw std::invalid_argument("caop_spectrum: invalid sites or bit_length");
            }
            // Upstream reads the hamfile on h-rank 0 only, as for the solve; see
            // caop_check_hamfile. The operator files are read the same way.
            const auto ham_facts = caop_check_hamfile(hamfile, complex, comm, "caop_spectrum");
            caop_check_modes(hamfile, ham_facts, sites, "caop_spectrum");
            // Upstream compares each operator's statistics with H on the h=t=0 ranks
            // only, so a mismatch there would hang the rest: refuse it here, everywhere.
            for (const auto& file : operator_files) {
                const auto facts = caop_check_hamfile(file, complex, comm, "caop_spectrum");
                caop_check_modes(file, facts, sites, "caop_spectrum");
                if (facts.fermion != ham_facts.fermion) {
                    throw std::invalid_argument(
                        "caop_spectrum: operator file " + file
                        + " has different statistics from the Hamiltonian");
                }
            }
#ifdef SBD_THRUST
            // make_slide deals the b ring stations over the t task ranks
            // (caop/basic/helper.h:41-55); with t > b some get none, and upstream's
            // Thrust mult returns early on an empty slide (mult_thrust.h:583), skipping
            // the Allreduce the other ranks wait in. The stored mode runs the CPU mult.
            if (!store_hamiltonian && t_comm_size > world / (h_comm_size * t_comm_size)) {
                throw std::invalid_argument(
                    "caop_spectrum on the gpu (Thrust) backend with the on-the-fly "
                    "Hamiltonian needs t_comm_size <= the b ranks (ranks / (h_comm_size * "
                    "t_comm_size)): a task rank with no basis-ring station hangs "
                    "upstream's GPU kernel. Lower t_comm_size, store the Hamiltonian, or "
                    "use device='cpu'");
            }
#endif
            // As upstream's app: the stored Hamiltonian's per-thread slots need the
            // same OpenMP team in every parallel region (caop/basic/mkham.h).
            omp_set_dynamic(0);
#ifdef SBD_THRUST
            {
                int numDevices = 1;
#ifdef __CUDACC__
                cudaGetDeviceCount(&numDevices);
                cudaSetDevice(rank % numDevices);
#else
                hipGetDeviceCount(&numDevices);
                hipSetDevice(rank % numDevices);
#endif
            }
#endif
            std::vector<std::complex<double>> g;
            size_t basis_size = 0, steps = 0;
            std::string stop_reason;
            auto run = [&](auto tag) {
                using ElemT = decltype(tag);
                sbd::det_vector<size_t>::init_elem_size((sites + bit_length - 1) / bit_length);
                const int b_size = int(world / h_comm_size / t_comm_size);
                MPI_Comm h_comm, b_comm, t_comm;
                sbd::setup_communicator(comm, int(h_comm_size), b_size, int(t_comm_size),
                                        h_comm, b_comm, t_comm);
                int h_rank = 0, t_rank = 0;
                MPI_Comm_rank(h_comm, &h_rank);
                MPI_Comm_rank(t_comm, &t_rank);
                cs::Wavefunction<ElemT> state;
                cs::Basis extra;
                if (h_rank == 0 && t_rank == 0) {
                    // Read the checkpoint as the type it was saved with (real and complex
                    // files are not interchangeable), then promote a real state into a
                    // complex calculation, as upstream's app does.
                    if (wavefunction_complex) {
                        if constexpr (std::is_same_v<ElemT, std::complex<double>>) {
                            state = cs::ReadWavefunctionCheckpoint<ElemT>(
                                loadname, shards, sites, bit_length, b_comm);
                        }
                    } else {
                        auto real_state = cs::ReadWavefunctionCheckpoint<double>(
                            loadname, shards, sites, bit_length, b_comm);
                        state.basis = std::move(real_state.basis);
                        state.coefficients.assign(real_state.coefficients.begin(),
                                                  real_state.coefficients.end());
                    }
                    if (!extra_basis.empty()) {
                        extra = cs::LoadBasis(extra_basis, sites, bit_length, b_comm);
                    }
                }
                sbd::GeneralOp<ElemT> h;
                bool fermion = false;
                sbd::load_GeneralOp_from_file(hamfile, h, fermion, h_comm, b_comm, t_comm);
                sbd::sparse_solver::LanczosOptions options;
                options.max_steps = max_steps;
                options.rank_atol = rank_atol;
                options.rank_rtol = rank_rtol;
                // General operators, as upstream's app: each file is a CAOP operator,
                // read on the h=t=0 ranks with the statistics of H, observables 0..n-1,
                // and the addition-sign resolvent <psi0| O_a^dag (w+E0-H+i eta)^-1 O_b |psi0>.
                std::vector<sbd::GeneralOp<ElemT>> applied;
                std::vector<size_t> observables = orbitals;
                if (general) {
                    observables.resize(operator_files.size());
                    std::iota(observables.begin(), observables.end(), size_t{0});
                    if (h_rank == 0 && t_rank == 0) {
                        applied.resize(operator_files.size());
                        for (size_t v = 0; v < applied.size(); ++v) {
                            bool op_fermion = false;
                            sbd::load_GeneralOp_from_file(operator_files[v], applied[v],
                                                          op_fermion, MPI_COMM_SELF, b_comm,
                                                          MPI_COMM_SELF);
                            if (op_fermion != fermion) {
                                throw std::invalid_argument(
                                    "caop_spectrum: operator file " + operator_files[v]
                                    + " has different statistics from the Hamiltonian");
                            }
                        }
                    }
                }
                auto result = cs::spectrum<ElemT>(state, h, fermion, observables,
                                                  general || addition, sites, bit_length,
                                                  extra, reference_energy, h_comm, b_comm,
                                                  t_comm, options,
                                                  general ? &applied : nullptr,
                                                  store_hamiltonian);
                result.operator_files = operator_files;
                MPI_Comm_free(&t_comm);
                MPI_Comm_free(&b_comm);
                MPI_Comm_free(&h_comm);
                basis_size = result.basis_size;
                steps = result.coefficients.A.size();
                stop_reason = result.coefficients.stop_reason;
                if (rank == 0 && !save_coefficients.empty()) {
                    cs::save_coefficients(save_coefficients, result, options);
                }
                const size_t n = observables.size();
                g.assign(omegas.size() * n * n, 0.0);
                for (size_t k = 0; k < omegas.size(); ++k) {
                    const auto gk = cs::green_function(result, omegas[k], eta);
                    std::copy(gk.begin(), gk.end(), g.begin() + k * n * n);
                }
            };
            {
                py::gil_scoped_release release;
                if (complex) run(std::complex<double>{});
                else run(double{});
            }
            const size_t n = general ? operator_files.size() : orbitals.size();
            py::array_t<std::complex<double>> out(std::vector<py::ssize_t>{
                static_cast<py::ssize_t>(omegas.size()), static_cast<py::ssize_t>(n),
                static_cast<py::ssize_t>(n)});
            std::copy(g.begin(), g.end(), out.mutable_data());
            py::dict results;
            results["green_function"] = out;
            results["basis_size"] = basis_size;
            results["steps"] = steps;
            results["stop_reason"] = stop_reason;
            return results;
        },
        "Green's function of a saved CAOP ground state by block Lanczos, as upstream's "
        "caop-spectrum app",
        py::arg("comm"), py::arg("loadname"), py::arg("shards"), py::arg("hamfile"),
        py::arg("orbitals"), py::arg("addition"), py::arg("sites"), py::arg("bit_length"),
        py::arg("reference_energy"), py::arg("extra_basis") = std::vector<std::string>{},
        py::arg("omegas") = std::vector<double>{}, py::arg("eta") = 0.1,
        py::arg("max_steps") = 100, py::arg("h_comm_size") = 1, py::arg("t_comm_size") = 1,
        py::arg("store_hamiltonian") = true, py::arg("complex") = false,
        py::arg("wavefunction_complex") = false, py::arg("save_coefficients") = "",
        py::arg("operator_files") = std::vector<std::string>{},
        py::arg("rank_atol") = 1e-13, py::arg("rank_rtol") = 1e-12);

    // Upstream's --read-coefficients: evaluate saved block-Lanczos coefficients on a new
    // frequency grid or broadening, without the Hamiltonian or the state. File-only and
    // rank-local; the scalar type comes from the file's header.
    m.def("caop_spectrum_from_coefficients",
        [](const std::string& path, const std::vector<double>& omegas, double eta) {
            namespace cs = sbd::caop::spectrum;
            std::vector<std::complex<double>> g;
            size_t n = 0, basis_size = 0, steps = 0;
            bool addition = true;
            std::string stop_reason;
            std::vector<size_t> labels;
            std::vector<std::string> operator_files;
            auto evaluate = [&](auto tag) {
                using ElemT = decltype(tag);
                auto result = cs::load_coefficients<ElemT>(path);
                n = result.orbitals.size();
                labels = result.orbitals;
                operator_files = result.operator_files;
                basis_size = result.basis_size;
                steps = result.coefficients.A.size();
                stop_reason = result.coefficients.stop_reason;
                addition = result.addition;
                g.assign(omegas.size() * n * n, 0.0);
                for (size_t k = 0; k < omegas.size(); ++k) {
                    const auto gk = cs::green_function(result, omegas[k], eta);
                    std::copy(gk.begin(), gk.end(), g.begin() + k * n * n);
                }
            };
            if (cs::complex_coefficient_file(path)) evaluate(std::complex<double>{});
            else evaluate(double{});
            py::array_t<std::complex<double>> out(std::vector<py::ssize_t>{
                static_cast<py::ssize_t>(omegas.size()), static_cast<py::ssize_t>(n),
                static_cast<py::ssize_t>(n)});
            std::copy(g.begin(), g.end(), out.mutable_data());
            py::dict results;
            results["green_function"] = out;
            results["basis_size"] = basis_size;
            results["steps"] = steps;
            results["stop_reason"] = stop_reason;
            results["addition"] = addition;
            results["orbitals"] = labels;
            results["operator_files"] = operator_files;
            return results;
        },
        "Green's function from coefficients saved by caop_spectrum, on a new grid, as "
        "upstream's caop-spectrum --read-coefficients",
        py::arg("path"), py::arg("omegas"), py::arg("eta"));
    // CAOP stochastic external variance and Epstein-Nesbet PT2 (upstream's
    // apps/caop-stat-evaluator, as a library call). It reads the state as upstream's
    // CLI does: the basis files, then the saved amplitudes onto them with
    // LoadWavefunction, on one rank per saved shard. It runs on the CPU, like the
    // GDB evaluator, whichever module calls it.
    m.def("caop_external_observables",
        [](py::object py_comm,
           const std::string& hamfile,
           const std::vector<std::string>& basisfiles,
           const std::string& loadname,
           size_t shards,
           size_t sites,
           size_t bit_length,
           double reference_energy,
           size_t sample_count,
           size_t batch_count,
           uint64_t seed,
           uint64_t first_batch_id,
           double heatbath_cutoff,
           const std::string& observable,
           size_t expansion_batch_size,
           double minimum_abs_denominator,
           size_t h_comm_size,
           bool complex) {
            MPI_Comm world = get_mpi_comm(py_comm);
            int rank = 0, size = 1;
            MPI_Comm_rank(world, &rank);
            MPI_Comm_size(world, &size);
            if (h_comm_size == 0 || size % h_comm_size) {
                throw std::invalid_argument(
                    "caop_external_observables: h_comm_size must divide the rank count");
            }
            sbd::stat_evaluator::Observable which;
            try {
                which = sbd::stat_evaluator::parse_observable(observable);
            } catch (const std::exception&) {
                throw std::invalid_argument(
                    "observable must be 'both', 'variance' or 'pt2', got '" + observable + "'");
            }
            if (sample_count < 2) {
                throw std::invalid_argument("sample_count must be at least 2");
            }
            if (batch_count == 0) {
                throw std::invalid_argument("batch_count must be at least 1");
            }
            const int b_size = size / int(h_comm_size);
            const size_t load_b_size = shards == 0 ? size_t(b_size) : shards;
            if (load_b_size > size_t(b_size)) {
                throw std::invalid_argument(
                    "caop_external_observables: the state has more shards than b ranks");
            }
            // Upstream loads the Hamiltonian, the basis and the state on some ranks only;
            // check what would otherwise hang the others or give a wrong answer silently.
            caop_check_hamfile(hamfile, complex, world, "caop_external_observables");
            caop_check_state(loadname, load_b_size, basisfiles, sites, bit_length, complex,
                             world, "caop_external_observables");
            std::vector<double> variance, pt2, seconds;
            std::vector<uint64_t> batch_ids;
            auto run = [&](auto tag) {
                using ElemT = decltype(tag);
                const int h_rank = rank / b_size;
                MPI_Comm h_comm, b_comm, load_b_comm = MPI_COMM_NULL;
                MPI_Comm_split(world, rank % b_size, h_rank, &h_comm);
                MPI_Comm_split(world, h_rank, rank % b_size, &b_comm);
                const bool load_rank = size_t(rank) < load_b_size;
                MPI_Comm_split(world, load_rank ? 0 : MPI_UNDEFINED, rank, &load_b_comm);
                sbd::GeneralOp<ElemT> hamiltonian;
                bool sign = false;
                sbd::load_GeneralOp_from_file(hamfile, hamiltonian, sign,
                                              h_comm, b_comm, MPI_COMM_SELF);
                if (size_t(hamiltonian.max_index()) >= sites) {
                    throw std::invalid_argument("a Hamiltonian mode exceeds sites");
                }
                sbd::det_vector<size_t>::init_elem_size((sites + bit_length - 1) / bit_length);
                sbd::det_vector<size_t> parents;
                std::vector<ElemT> coefficients;
                if (load_rank) {
                    sbd::load_basis_from_files(basisfiles, parents, bit_length, sites,
                                               load_b_comm);
                    sbd::LoadWavefunction(loadname, parents, MPI_COMM_SELF, load_b_comm,
                                          MPI_COMM_SELF, coefficients);
                    MPI_Comm_free(&load_b_comm);
                }
                sbd::caop::stat::StatEvaluatorOptions<double> options;
                options.observable = which;
                options.bit_length = bit_length;
                options.sites = sites;
                options.sample_count = sample_count;
                options.heatbath_cutoff = heatbath_cutoff;
                options.reference_energy = reference_energy;
                options.base_seed = seed;
                options.expansion_batch_size = expansion_batch_size;
                options.minimum_abs_denominator = minimum_abs_denominator;
                std::vector<sbd::caop::stat::BatchRequest> batches(batch_count);
                for (size_t k = 0; k < batch_count; ++k) {
                    batches[k].calculation_id = "sbd-python";
                    batches[k].batch_id = first_batch_id + k;
                }
                sbd::caop::stat::evaluate_statistical_batches<ElemT, double>(
                    parents, coefficients, hamiltonian, sign, options, batches, world,
                    [&](const sbd::caop::stat::EvaluatedBatch& evaluated) {
                        const auto& r = evaluated.record;
                        batch_ids.push_back(r.batch_id);
                        variance.push_back(r.has_variance ? r.variance_estimate : NAN);
                        pt2.push_back(r.has_pt2 ? r.pt2_estimate : NAN);
                        seconds.push_back(r.elapsed_seconds);
                    },
                    h_comm, b_comm);
                MPI_Comm_free(&b_comm);
                MPI_Comm_free(&h_comm);
            };
            {
                py::gil_scoped_release release;
                if (complex) run(std::complex<double>{});
                else run(double{});
            }
            py::dict out;
            out["variance"] = variance;
            out["pt2"] = pt2;
            out["batch_id"] = batch_ids;
            out["seconds"] = seconds;
            out["mpi_size"] = size;
            return out;
        },
        "Stochastic external variance and EN-PT2 of a saved CAOP state, as upstream's "
        "caop-stat-evaluator",
        py::arg("comm"), py::arg("hamfile"), py::arg("basisfiles"), py::arg("loadname"),
        py::arg("shards"), py::arg("sites"), py::arg("bit_length"),
        py::arg("reference_energy"), py::arg("sample_count") = 20000,
        py::arg("batch_count") = 10, py::arg("seed") = 0, py::arg("first_batch_id") = 0,
        py::arg("heatbath_cutoff") = 0.0, py::arg("observable") = "both",
        py::arg("expansion_batch_size") = 1000000,
        py::arg("minimum_abs_denominator") = 0.0, py::arg("h_comm_size") = 1,
        py::arg("complex") = false);
#else
    m.attr("__sbd_has_caop__") = false;
#endif

    m.attr("__version__") = "1.2.0";
}

// Made with Bob
