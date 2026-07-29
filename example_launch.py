from pathlib import Path
from kernel_launch import KernelLaunch, SharedBuffer, scalar, buffer, IN, OUT

# Absolute path to the mm-baremetal-examples directory.
# Adjust this to match your local build output location.
MM_EXAMPLES = Path(__file__).resolve().parents[2] / "mm-baremetal-examples"

result = SharedBuffer(256)

LAUNCHES = [
    KernelLaunch(
        elf=str(MM_EXAMPLES / "1CR/Producer/Producer.elf"),
        entry="producerStart",
        grid=(1, 1, 4),
        args=[
            scalar("i32", 10, name="n"),
            result.as_output(),
        ]
    ),
    KernelLaunch(
        elf=str(MM_EXAMPLES / "1CR/Consumer/Consumer.elf"),
        entry="consumerStart",
        grid=(2, 1, 4),
        args=[
            result.as_input(),
            buffer(128, OUT, name="out"),
        ]
    ),
]
