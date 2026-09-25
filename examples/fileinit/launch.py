# Exercises file-backed buffer init with *relative* paths.
#
# init="p.bin" / init="lut.bin" are resolved against this file's directory
# (not the cwd gen_launches_json.py runs from, and not the directory the
# launches JSON is written to), so the JSON ends up holding absolute paths to
# the .bin files sitting next to this launch.py. The ctest generates the JSON
# from a different cwd into a third directory to prove that.
#
# The ELF path is illustrative (like pipeline/); check_launch only needs the
# string.
from pathlib import Path
from kernel_launch import IN, KernelLaunch, SharedBuffer, buffer, scalar, OUT

N = 16  # int32 elements per buffer -> 64 bytes

lut = SharedBuffer(N * 4, init="lut.bin")  # relative -> <this dir>/lut.bin

LAUNCHES = [
    KernelLaunch(
        elf=str(Path(__file__).parent / "FileInit.elf"),
        entry="fileinitStart",
        grid=(1, 1, 4),
        args=[
            buffer(N * 4, IN, init="p.bin", name="p"),        # relative file init
            buffer(N * 4, IN, init=bytes(N * 4), name="q"),   # bytes init -> buf_init/
            lut.as_input(),
            buffer(N * 4, OUT, name="r"),
            scalar("i32", N, name="n"),
        ],
    )
]
