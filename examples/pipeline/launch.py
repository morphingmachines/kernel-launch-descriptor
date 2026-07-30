# Synthetic 3-kernel pipeline: Producer -> Filter -> Consumer.
#
# Demonstrates a multi-kernel LAUNCHES list where SharedBuffer instances
# chain data between kernels (Producer's output is Filter's input; Filter's
# output is Consumer's input), alongside a private per-kernel output buffer.
#
# Purely illustrative -- unlike fib/ and bmm/, these ELF paths don't point
# at real kernels. gen_launches_json.py / parse_launches / check_launch only
# need the path string, so this still exercises the full descriptor pipeline;
# only actually running it on a host driver would require real ELFs.
from pathlib import Path
from kernel_launch import KernelLaunch, SharedBuffer, scalar, buffer, OUT

STAGE_DIR = Path(__file__).parent

stage1_out = SharedBuffer(256)  # Producer -> Filter
stage2_out = SharedBuffer(256)  # Filter -> Consumer

LAUNCHES = [
    KernelLaunch(
        elf=str(STAGE_DIR / "Producer.elf"),
        entry="producerStart",
        grid=(1, 1, 4),
        args=[
            scalar("i32", 10, name="n"),
            stage1_out.as_output(),
        ],
    ),
    KernelLaunch(
        elf=str(STAGE_DIR / "Filter.elf"),
        entry="filterStart",
        grid=(1, 1, 4),
        args=[
            stage1_out.as_input(),
            scalar("u32", 0xFF, name="mask"),
            stage2_out.as_output(),
        ],
    ),
    KernelLaunch(
        elf=str(STAGE_DIR / "Consumer.elf"),
        entry="consumerStart",
        grid=(2, 1, 4),
        args=[
            stage2_out.as_input(),
            buffer(128, OUT, name="out"),
        ],
    ),
]
