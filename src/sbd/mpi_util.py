# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Run an MPI program so that a failure on any one rank ends the whole job.

An SBD solve puts every rank in collective operations. If one rank fails and exits
while the others wait in one of them, the job hangs until its time limit. Wrap the
program's entry point in :func:`run`::

    import sys
    from sbd.mpi_util import run

    def main():
        ...                     # parse arguments, call sbd, report
        return 0

    if __name__ == "__main__":
        sys.exit(run(main))

The example drivers all start this way.
"""

from __future__ import annotations

import sys
import time

__all__ = ["run", "let_output_through"]


def let_output_through(seconds=0.5):
    """Pause before ``MPI_Abort``, so the messages just written reach the launcher.

    Abort kills every rank at once, this one included, and output a rank has flushed
    can still be lost on the way to ``mpirun``: without a pause, the error that
    explains the abort went missing in about one run in four or five.
    """
    time.sleep(seconds)


def run(main):
    """Call ``main()``; on a failure on ANY rank, abort the whole MPI job.

    A rank that raises, is interrupted, or returns a non-zero code prints its own
    message, tagged with its rank (it may not be rank 0), then aborts the job, so the
    other ranks do not wait for it in a collective. In a single process it returns
    or raises as ``main`` does.

    Args:
        main: The program, a function of no arguments returning an exit code (0 or
            ``None`` for success).

    Returns:
        ``main``'s exit code, when every rank succeeds or there is one rank.
    """
    try:
        code = main()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        from mpi4py import MPI
        world = MPI.COMM_WORLD
        sys.stderr.write(f"[rank {world.Get_rank()}] interrupted\n")
        sys.stderr.flush()
        if world.Get_size() > 1:
            let_output_through()
            world.Abort(130)
        return 130
    except BaseException:
        import traceback
        from mpi4py import MPI
        world = MPI.COMM_WORLD
        sys.stderr.write(f"[rank {world.Get_rank()}] unhandled error:\n"
                         + traceback.format_exc())
        sys.stderr.flush()
        if world.Get_size() > 1:
            let_output_through()
            world.Abort(1)
        raise
    if code:
        from mpi4py import MPI
        world = MPI.COMM_WORLD
        if world.Get_size() > 1:
            sys.stderr.write(f"[rank {world.Get_rank()}] exiting with status "
                             f"{code}; aborting the MPI job\n")
            sys.stderr.flush()
            let_output_through()
            world.Abort(code)
    return code
