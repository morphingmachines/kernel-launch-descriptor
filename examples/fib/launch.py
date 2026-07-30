# Kernel launch descriptor for the Fib kernel.
# Mirrors mm-baremetal-examples/1CR/Fib/launch.py -- see that file for the
# kernel source (fib.c) this descriptor drives.
from pathlib import Path
from kernel_launch import KernelLaunch, scalar, buffer, OUT

FIB_DIR = Path(__file__).resolve().parents[4] / "mm-baremetal-examples" / "1CR" / "Fib"

LAUNCHES = [
    KernelLaunch(
        elf=str(FIB_DIR / "Fib.elf"),
        entry="fibStart",
        grid=(1, 1, 4),  # 1x1 CR grid, 4 CEs
        args=[  # order must match argument order in kernel entry function
            scalar("i32", 3, name="n"),     # n: compute fib(3)
            buffer(4, OUT, name="fib_n"),   # fib_n: 4-byte output
        ],
    )
]
