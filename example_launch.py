from kernel_launch import KernelLaunch, SharedBuffer, scalar, buffer, IN, OUT

result = SharedBuffer(256)

LAUNCHES = [
    KernelLaunch(
        elf="1CR/Producer/Producer.elf",
        grid=(1, 1, 4),
        args=[
            scalar("i32", 10),
            result.as_output(),
        ]
    ),
    KernelLaunch(
        elf="1CR/Consumer/Consumer.elf",
        grid=(2, 1, 4),
        args=[
            result.as_input(),
            buffer(128, OUT),
        ]
    ),
]
