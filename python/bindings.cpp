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
#include <cstring>

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
                      "Filename to dump wavefunction in matrix form")
#ifdef SBD_THRUST
        .def_readwrite("use_precalculated_dets", &sbd::tpb::SBD::use_precalculated_dets,
                      "Use precalculated determinants (THRUST)")
        .def_readwrite("max_memory_gb_for_determinants", &sbd::tpb::SBD::max_memory_gb_for_determinants,
                      "Maximum memory in GB for determinants (THRUST)")
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

            MPI_Comm comm = get_mpi_comm(py_comm);
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
    
    m.attr("__version__") = "1.2.0";
}

// Made with Bob
