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
 * This file is compiled three times with different module names + flags:
 * - _core_cpu                : CPU backend (host OpenMP via -fopenmp)
 * - _core_gpu_thrust         : Thrust GPU backend  (with -DSBD_THRUST,    nvc++ -cuda)
 * - _core_gpu_omp_offload    : OpenMP-offload GPU  (with -DUSE_OMP_OFFLOAD, nvc++ -mp=gpu)
 *
 * The module name is controlled by the SBD_MODULE_NAME macro.
 */

// SBD's mpi_utility.h uses std::cout without including <iostream>.
// Linux/libstdc++ pulls it in transitively; macOS/libc++ doesn't.
// Force-include here before any SBD header so the patch stays in our
// repo rather than in the vendored upstream submodule.
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <mpi4py/mpi4py.h>
#include <mpi.h>

#include "sbd/sbd.h"

#ifdef USE_OMP_OFFLOAD
#include <omp.h>
#endif

namespace py = pybind11;

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

// Module name is set by compiler flag, e.g.
//   -DSBD_MODULE_NAME=_core_cpu | _core_gpu_thrust | _core_gpu_omp_offload
#ifndef SBD_MODULE_NAME
#define SBD_MODULE_NAME _core
#endif

PYBIND11_MODULE(SBD_MODULE_NAME, m) {
    // Set module docstring based on backend
#ifdef SBD_THRUST
    m.doc() = "Python bindings for SBD (Selected Basis Diagonalization) library - GPU backend";
#else
    m.doc() = "Python bindings for SBD (Selected Basis Diagonalization) library - CPU backend";
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
    py::class_<sbd::tpb::SBD>(m, "TPB_SBD", py::module_local(), "Configuration for TPB diagonalization")
        .def(py::init<>())
        .def_readwrite("task_comm_size", &sbd::tpb::SBD::task_comm_size,
                      "Task communicator size")
        .def_readwrite("adet_comm_size", &sbd::tpb::SBD::adet_comm_size,
                      "Alpha determinant communicator size")
        .def_readwrite("bdet_comm_size", &sbd::tpb::SBD::bdet_comm_size,
                      "Beta determinant communicator size")
        .def_readwrite("h_comm_size", &sbd::tpb::SBD::h_comm_size,
                      "Helper communicator size")
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
    py::class_<sbd::gdb::SBD>(m, "GDB_SBD", py::module_local(), "Configuration for GDB diagonalization")
        .def(py::init<>())
        .def_readwrite("t_comm_size", &sbd::gdb::SBD::t_comm_size,
                      "Task communicator size")
        .def_readwrite("b_comm_size", &sbd::gdb::SBD::b_comm_size,
                      "Basis communicator size")
        .def_readwrite("h_comm_size", &sbd::gdb::SBD::h_comm_size,
                      "Helper communicator size")
        .def_readwrite("method", &sbd::gdb::SBD::method,
                      "Diagonalization method (0=Davidson, 1=Davidson+Ham, 2=Lanczos, 3=Lanczos+Ham)")
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
        .def_readwrite("do_shuffle", &sbd::gdb::SBD::do_shuffle,
                      "Shuffle determinants flag")
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

    m.def("makestring", &sbd::makestring,
          "Convert bitstring to string representation",
          py::arg("config"),
          py::arg("bit_length"),
          py::arg("total_bit_length"));

    m.def("from_string", &sbd::from_string,
          "Convert binary string to determinant format",
          py::arg("s"),
          py::arg("bit_length"),
          py::arg("total_bit_length"));

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
            //
            // Note: when this .so is loaded via Python dlopen, the symbol
            // omp_get_num_devices binds to libomp.so's stub (which returns 0
            // because libomp itself doesn't manage offload devices) instead
            // of libomptarget's working version. omp_set_default_device
            // IS shared between the two, so once we know the count we can
            // still set the device correctly. Fall back to parsing
            // CUDA_VISIBLE_DEVICES when omp_get_num_devices reports 0.
            {
                int n_dev = omp_get_num_devices();
                if (n_dev <= 0) {
                    const char* cvd = std::getenv("CUDA_VISIBLE_DEVICES");
                    if (cvd && *cvd) {
                        n_dev = 1;
                        for (const char* p = cvd; *p; ++p) {
                            if (*p == ',') ++n_dev;
                        }
                    }
                }
                if (n_dev > 0) {
                    omp_set_default_device(mpi_rank % n_dev);
                }
            }
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
    // The determinant list is the subspace itself: unlike TPB, the subspace is
    // not the Cartesian product of two half-determinant lists, so an arbitrary
    // sparse set of determinants can be diagonalized.
    // ========================================================================

    m.def("gdb_diag",
        [](py::object py_comm,
           const sbd::gdb::SBD& sbd_data,
           const sbd::FCIDump& fcidump,
           const std::vector<std::vector<size_t>>& det,
           const std::string& loadname,
           const std::string& savename) {

            // Convert MPI communicator
            MPI_Comm comm = get_mpi_comm(py_comm);

            int mpi_rank;
            MPI_Comm_rank(comm, &mpi_rank);

            // Every rank passes the whole determinant list, which is what
            // sbd::gdb::diag expects only when the basis is not split over
            // b_comm. Splitting it would require distributing the determinants
            // over b_comm first, as the file-based entry point does.
            if (sbd_data.b_comm_size != 1) {
                throw std::invalid_argument(
                    "gdb_diag requires b_comm_size == 1; distribute work over "
                    "t_comm_size and h_comm_size instead");
            }
            if (det.empty()) {
                throw std::invalid_argument("gdb_diag requires at least one determinant");
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

            // Output variables
            double energy;
            std::vector<double> density;
            sbd::det_vector<size_t> co_det;
            std::vector<std::vector<double>> one_p_rdm;
            std::vector<std::vector<double>> two_p_rdm;

            // det_vector packs each determinant into a fixed number of words,
            // which is a process-global property of the type: it is fixed by the
            // first determinant container built in the process and cannot be
            // changed afterwards. Set it here, while the GIL is still held, so
            // that a mismatch is reported before any work is done.
            try {
                sbd::det_vector<size_t>::init_elem_size(det[0].size());
            } catch (const std::length_error&) {
                throw std::invalid_argument(
                    "gdb_diag was already called in this process with a different "
                    "number of words per determinant, which SBD fixes for the "
                    "lifetime of the process. Keep norb and bit_length fixed, or "
                    "run the new problem in a fresh process.");
            }

            // SBD indexes the subspace with binary searches, so the determinants
            // must be in canonical order; an unsorted list silently yields a
            // wrong energy. Sort here rather than asking the caller to; the
            // exposed sort_bitarray reproduces this order for a caller that
            // needs it. sort_bitarray also removes duplicates, which would leave
            // part of the subspace unreachable, so reject them instead of
            // dropping them silently.
            std::vector<std::vector<size_t>> det_sorted(det);
            sbd::sort_bitarray(det_sorted);
            if (det_sorted.size() != det.size()) {
                throw std::invalid_argument(
                    "gdb_diag requires distinct determinants");
            }

            std::vector<std::vector<size_t>> co_det_vvs;

            {
                // Release GIL for long computation
                py::gil_scoped_release release;

                sbd::det_vector<size_t> det_packed(det_sorted.begin(), det_sorted.end());

                sbd::gdb::diag(comm, sbd_data, fcidump, det_packed,
                               loadname, savename, energy, density,
                               co_det, one_p_rdm, two_p_rdm);

                for (const auto& row : co_det) {
                    co_det_vvs.emplace_back(row.begin(), row.end());
                }
            }

            // Return results as dictionary
            py::dict results;
            results["energy"] = energy;
            results["density"] = density;
            results["carryover_det"] = co_det_vvs;
            results["one_p_rdm"] = one_p_rdm;
            results["two_p_rdm"] = two_p_rdm;

            return results;
        },
        "Perform GDB diagonalization over an explicit list of determinants",
        py::arg("comm"),
        py::arg("sbd_data"),
        py::arg("fcidump"),
        py::arg("det"),
        py::arg("loadname") = "",
        py::arg("savename") = "");

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
            //
            // Note: when this .so is loaded via Python dlopen, the symbol
            // omp_get_num_devices binds to libomp.so's stub (which returns 0
            // because libomp itself doesn't manage offload devices) instead
            // of libomptarget's working version. omp_set_default_device
            // IS shared between the two, so once we know the count we can
            // still set the device correctly. Fall back to parsing
            // CUDA_VISIBLE_DEVICES when omp_get_num_devices reports 0.
            {
                int n_dev = omp_get_num_devices();
                if (n_dev <= 0) {
                    const char* cvd = std::getenv("CUDA_VISIBLE_DEVICES");
                    if (cvd && *cvd) {
                        n_dev = 1;
                        for (const char* p = cvd; *p; ++p) {
                            if (*p == ',') ++n_dev;
                        }
                    }
                }
                if (n_dev > 0) {
                    omp_set_default_device(mpi_rank % n_dev);
                }
            }
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
