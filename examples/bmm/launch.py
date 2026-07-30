# Kernel launch descriptor for the BMM (matrix multiply) kernel.
# Mirrors mm-baremetal-examples/1CR/BMM/launch.py -- see that file for the
# kernel source (mm.c) this descriptor drives.
#
# Unlike Fib, this exercises the buffer `init` path: the two input matrices
# are generated host-side and written via kernel_launch's buf_init/ mechanism
# (see BUF_INIT_DIRNAME in kernel_launch.py) rather than left uninitialized.
import random
import struct
from pathlib import Path
from kernel_launch import IN, KernelLaunch, scalar, buffer, OUT

BMM_DIR = Path(__file__).resolve().parents[4] / "mm-baremetal-examples" / "1CR" / "BMM"

SQUARE_MATRIX_DIM = 16


def random_matrix_bytes(dim: int, low: int = 0, high: int = 9) -> bytes:
    """Pack dim*dim random int32 values (matches kernel's `int *` element type)."""
    values = [random.randint(low, high) for _ in range(dim * dim)]
    return struct.pack(f"<{dim * dim}i", *values)


random.seed(0)  # reproducible test vectors across regenerations

LAUNCHES = [
    KernelLaunch(
        elf=str(BMM_DIR / "BMM.elf"),
        entry="bmm_start",
        grid=(1, 1, 4),  # 1x1 CR grid, 4 CEs
        args=[  # order must match argument order in kernel entry function
            buffer(SQUARE_MATRIX_DIM * SQUARE_MATRIX_DIM * 4, IN,
                   init=random_matrix_bytes(SQUARE_MATRIX_DIM), name="p"),  # p: input matrix
            buffer(SQUARE_MATRIX_DIM * SQUARE_MATRIX_DIM * 4, IN,
                   init=random_matrix_bytes(SQUARE_MATRIX_DIM), name="q"),  # q: input matrix
            buffer(SQUARE_MATRIX_DIM * SQUARE_MATRIX_DIM * 4, OUT, name="r"),  # r: output matrix
            scalar("i32", SQUARE_MATRIX_DIM, name="n"),  # n: matrix dimension
        ],
    )
]
